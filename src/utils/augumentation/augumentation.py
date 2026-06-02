import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import copy
from utils.utils_methods import format_example
from utils.augumentation.methods import (
    aug_shift, aug_flip, random_brightness, random_contrast, aug_rotation_angle
)


# -------------------- 包装成单帧接口 -------------------- #
def _rotate_single(img, angle):
    """输入 [H,W,C] 输出 [H,W,C]"""
    h, w, c = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

def _shift_single(img, tx=0, ty=0):
    """输入 [H,W,C] 输出 [H,W,C]"""
    h, w, c = img.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

def _brightness_single(img, delta=0.1):
    """输入 [H,W,C] uint8 输出 [H,W,C] uint8"""
    delta = int(np.random.uniform(-delta, delta) * 255)
    img = img.astype(np.int16) + delta
    return np.clip(img, 0, 255).astype(np.uint8)

def _contrast_single(img, factor=0.1):
    """输入 [H,W,C] uint8 输出 [H,W,C] uint8"""
    f = 1.0 + np.random.uniform(-factor, factor)
    mean = img.mean()
    img = (img - mean) * f + mean
    return np.clip(img, 0, 255).astype(np.uint8)

def _rotate_batch_1or20(x, angle):
    """x: [1,H,W,C] 或 [20,H,W,C]"""
    if x.shape[0] == 1:
        return _rotate_single(x[0], angle)[None]
    return np.stack([_rotate_single(x[i], angle) for i in range(20)], axis=0)

def _shift_batch_1or20(x, tx=0, ty=0):
    if x.shape[0] == 1:
        return _shift_single(x[0], tx, ty)[None]
    return np.stack([_shift_single(x[i], tx, ty) for i in range(20)], axis=0)

def _brightness_batch(x):
    if x.shape[0] == 1:
        return _brightness_single(x[0])[None]
    return np.stack([_brightness_single(x[i]) for i in range(20)], axis=0)

def _contrast_batch(x):
    if x.shape[0] == 1:
        return _contrast_single(x[0])[None]
    return np.stack([_contrast_single(x[i]) for i in range(20)], axis=0)

# -------------------- 几何映射 -------------------- #

def _flip_sparse(idx, orig_kp_flat, img_width=128, img_height=96):
    """修正版：正确处理256x192图像的翻转坐标"""
    idx = idx.copy()

    # idx 格式假设: [N, 4] = [batch/frame_idx, y, x, channel]
    # 水平翻转只改变 x 坐标
    idx[:, 2] = (img_width - 1) - idx[:, 2]

    # 关键点格式: [x1, y1, x2, y2, ...] 展平数组
    kp = orig_kp_flat.reshape(-1, 2)
    kp[:, 0] = (img_width - 1) - kp[:, 0]  # 翻转 x

    return idx, kp.reshape(-1)

# -------------------- 统一转 bytes -------------------- #
def _to_bytes(dic):
    """增强后写入前统一转 bytes"""
    dic_out = dic.copy()

    # 处理所有需要转换的字段
    byte_fields = [
        'Ic', 'It', 'Mc', 'Mt',
        'Ic_original_keypoints', 'It_original_keypoints',
        'Ic_values', 'It_values'
    ]

    for k in byte_fields:
        if k in dic_out and dic_out[k] is not None:
            if isinstance(dic_out[k], np.ndarray):
                dic_out[k] = dic_out[k].tobytes()
            elif isinstance(dic_out[k], (list, tuple)):
                # 自动推断数据类型
                if k.endswith('_values'):
                    dic_out[k] = np.array(dic_out[k], dtype=np.float32).tobytes()
                else:
                    dic_out[k] = np.array(dic_out[k], dtype=np.int64).tobytes()

    # 特殊处理 indices
    for k in ['Ic_indices', 'It_indices']:
        if k in dic_out and dic_out[k] is not None:
            if isinstance(dic_out[k], np.ndarray):
                dic_out[k] = dic_out[k].tobytes()
            elif isinstance(dic_out[k], (list, tuple)):
                dic_out[k] = np.array(dic_out[k], dtype=np.int64).tobytes()

    return dic_out

# -------------------- 主增强流程 -------------------- #


# ------------------------------------------------------------------
# 已嵌入“去除背景”的完整函数
# ------------------------------------------------------------------
def apply_augumentation(data_tfrecord_path,
                        unprocess_dataset_iterator,
                        name_dataset,
                        len_dataset):
    name_file = f'{name_dataset}_augumentation.tfrecord'
    name_tfrecord = os.path.join(data_tfrecord_path, name_file)
    writer = tf.io.TFRecordWriter(name_tfrecord)

    cnt = 0
    sys.stdout.write(f"\nAugment {name_dataset}..\n")

    # --------------------------------------------------------------
    # 1. 工具函数
    # --------------------------------------------------------------
    def remove_background(dic):
        """掩膜 0/255 → 0/1，再去背"""
        dic['Ic'] = dic['Ic'] * (dic['Mc'] > 0).astype(np.uint8)
        dic['It'] = dic['It'] * (dic['Mt'] > 0).astype(np.uint8)
        return dic

    def update_shape_fields(dic_data):
        dic_data['shape_len_Ic_original_keypoints'] = dic_data['Ic_original_keypoints'].size
        dic_data['shape_len_It_original_keypoints'] = dic_data['It_original_keypoints'].size
        dic_data['shape_len_Ic_indices'] = len(dic_data['Ic_indices']) if isinstance(dic_data['Ic_indices'], (list, np.ndarray)) else 0
        dic_data['shape_len_It_indices'] = len(dic_data['It_indices']) if isinstance(dic_data['It_indices'], (list, np.ndarray)) else 0
        dic_data['shape_len_Ic_values']  = len(dic_data['Ic_values'])  if isinstance(dic_data['Ic_values'],  (list, np.ndarray)) else 0
        dic_data['shape_len_It_values']  = len(dic_data['It_values'])  if isinstance(dic_data['It_values'],  (list, np.ndarray)) else 0
        return dic_data

    def validate_data_shapes(dic_data, augmentation_type):
        mc_size = dic_data['Mc'].nbytes if hasattr(dic_data['Mc'], 'nbytes') else len(dic_data['Mc'])
        mt_size = dic_data['Mt'].nbytes if hasattr(dic_data['Mt'], 'nbytes') else len(dic_data['Mt'])
        expected_mc_size = 1 * 96 * 128 * 1   #
        expected_mt_size = 20 * 96 * 128 * 1  #
        return mc_size == expected_mc_size and mt_size == expected_mt_size

    def deep_copy_dic(dic):
        new_dic = {}
        for k, v in dic.items():
            if isinstance(v, np.ndarray):
                new_dic[k] = v.copy()
            elif isinstance(v, (list, dict)):
                new_dic[k] = copy.deepcopy(v)
            else:
                new_dic[k] = v
        return new_dic

    def safe_augmentation(dic_data, aug_type, aug_func, **kwargs):
        try:
            result = aug_func(dic_data, **kwargs)
            return result if validate_data_shapes(result, aug_type) else dic_data
        except Exception as e:
            print(f"错误: {aug_type} 增强失败: {e}")
            return dic_data

    # --------------------------------------------------------------
    # 2. 主循环
    # --------------------------------------------------------------
    for id_ex in range(len_dataset):
        sys.stdout.write(f'\rExample: {id_ex + 1} / {len_dataset}')
        batch = next(unprocess_dataset_iterator)

        # --- decode ---
        def _to_np(b, dtype=None):
            x = b.numpy()[0]
            return np.frombuffer(x, dtype=dtype) if dtype else x

        Ic_np = _to_np(batch[0], np.uint8).reshape(1, 96, 128, 3)
        It_np = _to_np(batch[1], np.uint8).reshape(20, 96, 128, 3)
        Mc_np = _to_np(batch[4], np.uint8).reshape(1, 96, 128, 1)
        Mt_np = _to_np(batch[5], np.uint8).reshape(20, 96, 128, 1)
        Ic_idx_np = _to_np(batch[10], np.int64).reshape(-1, 4)
        It_idx_np = _to_np(batch[11], np.int64).reshape(-1, 4)
        Ic_val_np = _to_np(batch[12], np.float32)
        It_val_np = _to_np(batch[13], np.float32)
        Ic_kp_np = _to_np(batch[14], np.int64)
        It_kp_np = _to_np(batch[15], np.int64)

        dic = {
            'pz_condition': batch[6].numpy()[0].decode('utf-8'),
            'pz_target':    batch[7].numpy()[0].decode('utf-8'),
            'Ic_image_name': batch[8].numpy()[0].decode('utf-8'),
            'It_image_name': batch[9].numpy()[0].decode('utf-8'),

            'Ic': Ic_np, 'It': It_np, 'Mc': Mc_np, 'Mt': Mt_np,
            'image_format': b'PNG', 'image_height': 96, 'image_width': 128,

            'Ic_original_keypoints': Ic_kp_np, 'It_original_keypoints': It_kp_np,
            'Ic_indices': Ic_idx_np, 'It_indices': It_idx_np,
            'Ic_values':  Ic_val_np,  'It_values':  It_val_np,
            'radius_keypoints': batch[16].numpy()[0],
        }
        dic = update_shape_fields(dic)

        # >>> 去除背景 <<<
        dic = remove_background(dic)

        if not validate_data_shapes(dic, "原始样本"):
            print("原始样本形状验证失败，跳过此样本")
            continue

        # --- 原始样本 ---
        writer.write(format_example(_to_bytes(dic)).SerializeToString())
        cnt += 1

        # --- 翻转 ---
        dic_f = deep_copy_dic(dic)
        dic_f['Ic'] = np.flip(dic_f['Ic'], axis=2); dic_f['It'] = np.flip(dic_f['It'], axis=2)
        dic_f['Mc'] = np.flip(dic_f['Mc'], axis=2); dic_f['Mt'] = np.flip(dic_f['Mt'], axis=2)
        dic_f['Ic_indices'], dic_f['Ic_original_keypoints'] = _flip_sparse(
            dic_f['Ic_indices'], dic_f['Ic_original_keypoints'])
        dic_f['It_indices'], dic_f['It_original_keypoints'] = _flip_sparse(
            dic_f['It_indices'], dic_f['It_original_keypoints'])
        if validate_data_shapes(dic_f, "翻转增强"):
            dic_f = update_shape_fields(dic_f)
            writer.write(format_example(_to_bytes(dic_f)).SerializeToString())
            cnt += 1

        # --- 旋转 4 角度 ---
        for ang in tf.random.uniform([4], -91, 91, dtype=tf.int32).numpy():
            dic_r = deep_copy_dic(dic)
            dic_r['Ic'] = _rotate_batch_1or20(dic_r['Ic'], ang); dic_r['It'] = _rotate_batch_1or20(dic_r['It'], ang)
            dic_r['Mc'] = _rotate_batch_1or20(dic_r['Mc'], ang); dic_r['Mt'] = _rotate_batch_1or20(dic_r['Mt'], ang)
            dic_r = safe_augmentation(dic_r, f"旋转增强(角度{ang})", aug_rotation_angle, angle_deegre=ang, indx_img="c")
            dic_r = safe_augmentation(dic_r, f"旋转增强(角度{ang})", aug_rotation_angle, angle_deegre=ang, indx_img="t")
            dic_r = update_shape_fields(dic_r)
            writer.write(format_example(_to_bytes(dic_r)).SerializeToString())
            cnt += 1

        # --- 平移 ---
        for tx in tf.random.uniform([2], -31, 31, dtype=tf.int32).numpy():
            dic_s = deep_copy_dic(dic)
            dic_s = aug_shift(dic_s, type="or", indx_img="c", tx=tx)
            dic_s = aug_shift(dic_s, type="or", indx_img="t", tx=tx)
            if validate_data_shapes(dic_s, f"水平平移(tx={tx})"):
                dic_s = update_shape_fields(dic_s)
                writer.write(format_example(_to_bytes(dic_s)).SerializeToString())
                cnt += 1
        for ty in tf.random.uniform([2], -11, 11, dtype=tf.int32).numpy():
            dic_s = deep_copy_dic(dic)
            dic_s = aug_shift(dic_s, type="ver", indx_img="c", ty=ty)
            dic_s = aug_shift(dic_s, type="ver", indx_img="t", ty=ty)
            if validate_data_shapes(dic_s, f"垂直平移(ty={ty})"):
                dic_s = update_shape_fields(dic_s)
                writer.write(format_example(_to_bytes(dic_s)).SerializeToString())
                cnt += 1

        # --- 亮度 & 对比度 ---
        np.random.seed(42)
        for _ in range(4):
            dic_b = deep_copy_dic(dic)
            delta_b = np.random.uniform(-0.05, 0.05)
            dic_b['Ic'] = _enhance_batch(dic_b['Ic'], _brightness_single_det, delta_b)
            dic_b['It'] = _enhance_batch(dic_b['It'], _brightness_single_det, delta_b)
            if validate_data_shapes(dic_b, "亮度增强"):
                dic_b = update_shape_fields(dic_b)
                writer.write(format_example(_to_bytes(dic_b)).SerializeToString())
                cnt += 1
        for _ in range(4):
            dic_c = deep_copy_dic(dic)
            factor_c = np.random.uniform(0.95, 1.05)
            dic_c['Ic'] = _enhance_batch(dic_c['Ic'], _contrast_single_det, factor_c)
            dic_c['It'] = _enhance_batch(dic_c['It'], _contrast_single_det, factor_c)
            if validate_data_shapes(dic_c, "对比度增强"):
                dic_c = update_shape_fields(dic_c)
                writer.write(format_example(_to_bytes(dic_c)).SerializeToString())
                cnt += 1

    writer.close()
    import gc; gc.collect()
    if tf.executing_eagerly():
        tf.keras.backend.clear_session()
    print(f"增强完成! 共生成 {cnt} 个样本")
    return name_tfrecord, cnt

def aug_sequence(dic, aug_fn, **kwargs):
    """对 It/Mt/It_indices/It_values/It_mask 逐帧做相同 aug"""
    T = dic['It'].shape[0]
    new_It, new_Mt, new_idx, new_val, new_mask = [], [], [], [], []
    for t in range(T):
        frame = {
            'It': dic['It'][t],
            'Mt': dic['Mt'][t],
            'It_indices': dic['It_indices'][t],
            'It_values':  dic['It_values'][t],
          #  'It_mask':    dic['It_mask'][t]
        }
        frame = aug_fn(frame, **kwargs)   # 你已有的单帧 aug
        new_It.append(frame['It'])
        new_Mt.append(frame['Mt'])
        new_idx.append(frame['It_indices'])
        new_val.append(frame['It_values'])
        new_mask.append(frame['It_mask'])
    # 拼回序列
    dic['It'] = np.stack(new_It, 0)
    dic['Mt'] = np.stack(new_Mt, 0)
    dic['It_indices'] = np.stack(new_idx, 0)
    dic['It_values']  = np.stack(new_val, 0)
    dic['It_mask']    = np.stack(new_mask, 0)
    return dic

def _enhance_batch(x, func, param):
    """
    x: [1,H,W,C] or [20,H,W,C]
    func: _brightness_single or _contrast_single
    param: 同一个随机参数，保证帧间一致
    """
    if x.shape[0] == 1:
        return func(x[0], param)[None]
    return np.stack([func(x[i], param) for i in range(x.shape[0])], axis=0)

def _brightness_single_det(img, delta):
    """确定性亮度调整，delta 已外部生成"""
    img = img.astype(np.int16) + int(delta * 255)
    return np.clip(img, 0, 255).astype(np.uint8)

def _contrast_single_det(img, factor):
    """确定性对比度调整，factor 已外部生成"""
    mean = img.mean()
    img = (img - mean) * factor + mean
    return np.clip(img, 0, 255).astype(np.uint8)



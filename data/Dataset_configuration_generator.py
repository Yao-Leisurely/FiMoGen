import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append("../src")
from utils import format_example, aug_flip, getSparsePose, enlarge_keypoint

T = 5 # 序列长度（帧数）


def _sparse2dense(indices, values, shape):
    """
    Create a binary mask in which only the shape of the infant takes on a value of 1
    """
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        dense[r, c, 0] = values[i]
    return dense


def _load_external_mask(image_name, pz_id, H_out, W_out):
    """
    从外部掩码目录加载掩码文件
    根据新的文件结构：external_masks/pz[ID]/[frame_number]_segmentation.png
    """
    # 提取帧号并构建掩码文件名
    frame_number = image_name.split('_')[0]  # 如 "00000"
    mask_filename = f"{frame_number}_segmentation.png"

    # 构建掩码文件路径
    mask_path = os.path.join(dir_external_masks, f"pz{pz_id}", mask_filename)

    if not os.path.exists(mask_path):
        # 如果找不到掩码文件，返回全零掩码
        print(f"Warning: Mask file not found: {mask_path}")
        return np.zeros((H_out, W_out, 1), dtype=np.uint8)

    # 读取掩码文件
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Failed to load mask: {mask_path}")
        return np.zeros((H_out, W_out, 1), dtype=np.uint8)

    # 调整大小
    mask = cv2.resize(mask, (W_out, H_out), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 0, 1, 0)  # 二值化
    return mask.astype(np.uint8)[..., np.newaxis]


def _format_data(id_pz_condition, id_pz_target,
                 Ic_annotations, It_annotations,
                 r_k, radius_keypoints_mask, r_h, dilatation):
    """
    单条件图 + 20 帧目标序列
    使用外部提供的掩码文件（新文件结构）
    """
    T = 5
    H_out, W_out = 192, 256
    scale = 2.5  # 640/128 = 480/96

    # ---------- 1. 条件图 ---------- #
    pz_condition = f'pz{id_pz_condition}'
    pz_target = f'pz{id_pz_target}'

    # 图像文件名（新格式：00000_8bit.png）
    name_img_c = Ic_annotations['image']
    img_c_path = os.path.join(dir_images, pz_condition, name_img_c)
    img_c = cv2.imread(img_c_path, cv2.IMREAD_COLOR)
    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)   ####################颜色交换成RGB

    if img_c is None:
        print(f"Error: Failed to load condition image: {img_c_path}")
        return None

    h_raw, w_raw = img_c.shape[:2]

    # 使用外部掩码（新文件结构）
    Mc = _load_external_mask(name_img_c, id_pz_condition, H_out, W_out)
    img_c = cv2.resize(img_c, (W_out, H_out), interpolation=cv2.INTER_NEAREST)

    Ic = np.expand_dims(img_c, 0)  # [1,H,W,3]
    Mc = np.expand_dims(Mc, 0)  # [1,H,W,1]

    # 条件图 14 关键点
    kps_c = [[int(x) // scale, int(y) // scale] if x != -1 and y != -1 else [-1, -1]
             for x, y in (kp.split(',') for kp in Ic_annotations[1:])]
    ic_idx, ic_val = getSparsePose(kps_c, H_out, W_out, r_k, mode='Solid')
    ic_idx = np.asarray(ic_idx, dtype=np.int64)
    ic_val = np.asarray(ic_val, dtype=np.float32)
    ic_idx = np.concatenate([np.zeros((ic_idx.shape[0], 1), dtype=np.int64), ic_idx], 1)  # [N,4]

    # ---------- 2. 目标序列（20 帧） ---------- #
    It, Mt, It_indices, It_values, kp_tgt = [], [], [], [], []
    frame_offset = [0]

    for t in range(T):
        row = It_annotations.iloc[t]
        name_img_t = row['image']
        img_t_path = os.path.join(dir_images, pz_target, name_img_t)
        img_t = cv2.imread(img_t_path, cv2.IMREAD_COLOR)
        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)  ##################################颜色交换成RGB

        if img_t is None:
            print(f"Error: Failed to load target image: {img_t_path}")
            return None

        img_t = cv2.resize(img_t, (W_out, H_out), interpolation=cv2.INTER_NEAREST)
        It.append(img_t)

        kps_t = [[int(x) // scale, int(y) // scale] if x != -1 and y != -1 else [-1, -1]
                 for x, y in (kp.split(',') for kp in row[1:])]
        kp_tgt.append(kps_t)  # 收集 14 关键点

        # 使用外部掩码（新文件结构）
        Mt_t = _load_external_mask(name_img_t, id_pz_target, H_out, W_out)
        Mt.append(Mt_t)

        # 完整稀疏场
        ind, val = getSparsePose(kps_t, H_out, W_out, r_k, mode='Solid')
        ind = np.asarray(ind, dtype=np.int64)
        val = np.asarray(val, dtype=np.float32)
        ind = np.concatenate([np.full((ind.shape[0], 1), t, dtype=np.int64), ind], 1)  # [N,4]

        It_indices.append(ind)
        It_values.append(val)
        frame_offset.append(frame_offset[-1] + ind.shape[0])

    # stack 成数组
    It = np.stack(It, axis=0)  # [T,H,W,3]
    Mt = np.stack(Mt, axis=0)  # [T,H,W,1]
    kp_tgt = np.array(kp_tgt, dtype=np.int64)  # [T,14,2]

    # 稀疏 concat
    It_indices = np.concatenate(It_indices, 0)  # [N_all,4]
    It_values = np.concatenate(It_values, 0)  # [N_all]

    # ---------- 3. 打包 ---------- #
    Ic_orig_flat = np.array(kps_c, dtype=np.int64).reshape(-1)
    It_orig_flat = kp_tgt.reshape(-1)

    dic_data = {
        'pz_condition': pz_condition,
        'pz_target': pz_target,

        'Ic_image_name': name_img_c,
        'It_image_name': It_annotations.iloc[0]['image'],

        'Ic': Ic,
        'It': It,
        'Mc': Mc.astype(np.uint8).tobytes(),
        'Mt': Mt.astype(np.uint8).tobytes(),

        'image_format': b'PNG',
        'image_height': H_out,
        'image_width': W_out,

        # 兼容旧接口
        'Ic_original_keypoints': Ic_orig_flat.tobytes(),
        'shape_len_Ic_original_keypoints': Ic_orig_flat.size,
        'It_original_keypoints': It_orig_flat.tobytes(),
        'shape_len_It_original_keypoints': It_orig_flat.size,

        # 稀疏张量（完整）
        'Ic_indices': ic_idx.astype(np.int64).tobytes(),
        'Ic_values': ic_val.astype(np.float32).tobytes(),
        'shape_len_Ic_indices': ic_idx.shape[0],
        'shape_len_Ic_values': ic_val.shape[0],

        'It_indices': It_indices.astype(np.int64).tobytes(),
        'It_values': It_values.astype(np.float32).tobytes(),
        'shape_len_It_indices': It_indices.shape[0],
        'shape_len_It_values': It_values.shape[0],
        'It_frame_offset': np.asarray(frame_offset, dtype=np.int32).tobytes(),
        'It_n_frames': T,

        'radius_keypoints': r_k,
    }

    print(f"[WRITE] Ic_nnz={ic_idx.shape[0]}  It_nnz={It_indices.shape[0]}")
    return dic_data


def _process_pair(args):
    """
    处理配对的进程函数
    """
    (id_pz_condition, id_pz_target, dir_annotations, dir_images, dir_external_masks,
     T, campionamento, r_k, radius_keypoints_mask, r_h, dilatation,
     flip, key_dict, tot_pairs_offset) = args

    path_c = os.path.join(dir_annotations, f'result_pz{id_pz_condition}.csv')
    path_t = os.path.join(dir_annotations, f'result_pz{id_pz_target}.csv')
    df_c = pd.read_csv(path_c, delimiter=';')
    df_t = pd.read_csv(path_t, delimiter=';')

    if len(df_c) < T or len(df_t) < T:
        return [], {}, 0

    examples, local_hist = [], {}
    local_cnt = 0
    max_start = len(df_c) - T

    for start in range(0, max_start + 1, campionamento):
        if start + T > len(df_t):
            continue
        Ic_ann = df_c.iloc[start]
        It_seq = df_t.iloc[start:start + T]

        dic_data = _format_data(id_pz_condition, id_pz_target,
                                Ic_ann, It_seq,
                                r_k, radius_keypoints_mask, r_h, dilatation)

        if dic_data is None:
            continue  # 跳过处理失败的数据

        ex = format_example(dic_data)
        examples.append(ex.SerializeToString())

        it_img_names = [It_seq.iloc[t]['image'] for t in range(T)]
        key = f'{key_dict}_{tot_pairs_offset + local_cnt}'
        local_hist[key] = {
            'pz_condition': f'pz{id_pz_condition}',
            'img_condition': Ic_ann['image'],
            'pz_target': f'pz{id_pz_target}',
            'img_target': it_img_names,
            'id_in_tfrecord': key
        }
        local_cnt += 1

        if flip:
            dic_flip = aug_flip(dic_data.copy())
            ex_flip = format_example(dic_flip)
            examples.append(ex_flip.SerializeToString())

            key = f'{key_dict}_{tot_pairs_offset + local_cnt}'
            local_hist[key] = {
                'pz_condition': f'pz{id_pz_condition}',
                'img_condition': Ic_ann['image'],
                'pz_target': f'pz{id_pz_target}',
                'img_target': it_img_names,
                'id_in_tfrecord': key
            }
            local_cnt += 1

    return examples, local_hist, local_cnt


def fill_tfrecord(dic_history, lista, tfrecord_writer, radius_keypoints_pose,
                  radius_keypoints_mask, radius_head_mask, dilatation,
                  campionamento, key_dict, flip=False, pairing_mode="negative"):
    """
    并行版：生成 T 帧目标序列并写入 TFRecord（负样本配对）
    """
    if pairing_mode != "negative":
        raise NotImplementedError("Only 'negative' pairing mode implemented in parallel version.")

    # 构造任务列表
    tasks = []
    for id_pz_condition in lista:
        for id_pz_target in lista:
            if id_pz_condition == id_pz_target:
                continue
            tasks.append((id_pz_condition, id_pz_target,
                          dir_annotations, dir_images, dir_external_masks,
                          T, campionamento, radius_keypoints_pose,
                          radius_keypoints_mask, radius_head_mask,
                          dilatation, flip, key_dict, 0))

    # 并行执行
    all_examples = []
    offset = 0

    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 4)) as exe:  # 限制线程数避免内存溢出
        future_map = {exe.submit(_process_pair, t): (t[0], t[1]) for t in tasks}

        for fut in tqdm(as_completed(future_map), total=len(tasks), desc=f"{key_dict} pairs"):
            id_c, id_t = future_map[fut]
            try:
                exs, local_hist, n = fut.result()
                # 更新历史记录
                for k, v in local_hist.items():
                    new_key = f"{key_dict}_{offset}"
                    v['id_in_tfrecord'] = new_key
                    dic_history[new_key] = v
                    offset += 1
                all_examples.extend(exs)
            except Exception as e:
                print(f"Error processing pair ({id_c}, {id_t}): {e}")
                continue

    # 一次性写入
    for ex in all_examples:
        tfrecord_writer.write(ex)
    tfrecord_writer.close()

    print(f'\nSET DATI TERMINATO  tot={offset}\n')
    return offset


if __name__ == '__main__':
    """
    主程序：创建数据集配置
    """
    #### CONFIG ##########
    dataset_type = "Syntetich"
    dataset_note = "complete"
    dataset_configuration = "testing configuration"

    # 数据集划分
    lista_pz_train = [101, 103, 105, 106, 107, 109, 110, 112]
    #lista_pz_train = [105, 103]
    lista_pz_valid = [102, 111]
    lista_pz_test = [108,104]

    # 参数配置
    campionamento = 10
    r_k = 2  # keypoints radius on Pose maps Pc and Pt
    radius_keypoints_mask = 1
    r_h = 40  # mask radius head
    dilatation = 35
    flip = False
    pairing_mode = "negative"

    #########################
    name_dataset = f'{dataset_type}_{dataset_note}'
    dataset_configuration = '_'.join(dataset_configuration.split(" "))

    # 根据新文件结构定义路径
    dir_dataset = os.path.join('.', '/media/jy/36623f40-fafa-4a13-879d-8d47450b13cc/jy/gan/video generation/data/Syntetich_complete')  # 数据集根目录
    dir_images = os.path.join(dir_dataset, 'images')  # 图像目录
    dir_external_masks = os.path.join(dir_dataset, 'external_masks')  # 外部掩码目录
    dir_annotations = os.path.join(dir_dataset, 'annotations')  # 注释文件目录
    dir_configuration = os.path.join(dir_dataset, "tfrecord", dataset_configuration)  # TFRecord输出目录

    keypoint_num = 14

    name_tfrecord_train = f'{dataset_type}_train.tfrecord'
    name_tfrecord_valid = f'{dataset_type}_valid.tfrecord'
    name_tfrecord_test = f'{dataset_type}_test.tfrecord'

    # 检查路径是否存在
    assert os.path.exists(dir_dataset), f"Dataset directory not found: {dir_dataset}"
    assert os.path.exists(dir_images), f"Images directory not found: {dir_images}"
    assert os.path.exists(dir_external_masks), f"External masks directory not found: {dir_external_masks}"
    assert os.path.exists(dir_annotations), f"Annotations directory not found: {dir_annotations}"

    # 检查每个pz目录和CSV文件
    for dataset_split in [lista_pz_train, lista_pz_valid, lista_pz_test]:
        for id_unique in dataset_split:
            pz_image_path = os.path.join(dir_images, f'pz{id_unique}')
            pz_mask_path = os.path.join(dir_external_masks, f'pz{id_unique}')
            csv_path = os.path.join(dir_annotations, f'result_pz{id_unique}.csv')

            assert os.path.exists(pz_image_path), f"Image directory not found: {pz_image_path}"
            assert os.path.exists(pz_mask_path), f"Mask directory not found: {pz_mask_path}"
            assert os.path.exists(csv_path), f"CSV file not found: {csv_path}"

            # 检查目录中是否有文件
            image_files = [f for f in os.listdir(pz_image_path) if f.endswith('_8bit.png')]
            mask_files = [f for f in os.listdir(pz_mask_path) if f.endswith('_segmentation.png')]

            assert len(image_files) > 0, f"No images found in: {pz_image_path}"
            assert len(mask_files) > 0, f"No masks found in: {pz_mask_path}"

            print(f"pz{id_unique}: {len(image_files)} images, {len(mask_files)} masks")

    if not os.path.exists(dir_configuration):
        os.makedirs(dir_configuration)
    assert campionamento != 0

    # TFRecord 文件路径
    output_filename_train = os.path.join(dir_configuration, name_tfrecord_train)
    output_filename_valid = os.path.join(dir_configuration, name_tfrecord_valid)
    output_filename_test = os.path.join(dir_configuration, name_tfrecord_test)

    dic_history = {}
    tot_train, tot_valid, tot_test = 0, 0, 0

    # 处理训练集
    if os.path.exists(output_filename_train):
        r_tr = input("Il tf record di train esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_tr in ["Y", "N", "y", "n"]
    else:
        r_tr = "Y"

    if r_tr in ["Y", "y"]:
        tfrecord_writer_train = tf.compat.v1.python_io.TFRecordWriter(output_filename_train)
        tot_train = fill_tfrecord(dic_history, lista_pz_train, tfrecord_writer_train, r_k, radius_keypoints_mask,
                                  r_h, dilatation, campionamento, key_dict="train", flip=flip,
                                  pairing_mode=pairing_mode)
        print("TOT TRAIN: ", tot_train)
    else:
        print("OK, non farò nulla sul train set")

    # 处理验证集
    if os.path.exists(output_filename_valid):
        r_v = input("Il tf record di valid esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_v in ["Y", "N", "y", "n"]
    else:
        r_v = "Y"

    if r_v in ["Y", "y"]:
        tfrecord_writer_valid = tf.compat.v1.python_io.TFRecordWriter(output_filename_valid)
        tot_valid = fill_tfrecord(dic_history, lista_pz_valid, tfrecord_writer_valid, r_k, radius_keypoints_mask,
                                  r_h, dilatation, campionamento, key_dict="valid", flip=flip,
                                  pairing_mode=pairing_mode)
        print("TOT VALID: ", tot_valid)
    else:
        print("OK, non farò nulla sul valid set")

    # 处理测试集
    if os.path.exists(output_filename_test):
        r_te = input("Il tf record di test esiste già. Sovrascriverlo? Yes[Y] No[N]")
        assert r_te in ["Y", "N", "y", "n"]
    else:
        r_te = "Y"

    if r_te in ["Y", "y"]:
        tfrecord_writer_test = tf.compat.v1.python_io.TFRecordWriter(output_filename_test)
        tot_test = fill_tfrecord(dic_history, lista_pz_test, tfrecord_writer_test, r_k, radius_keypoints_mask,
                                 r_h, dilatation, campionamento, key_dict="test", flip=flip, pairing_mode=pairing_mode)
        print("TOT TEST: ", tot_test)
    else:
        print("OK, non farò nulla sul test set")

    # 保存配置信息
    dic = {
        "general": {
            "campionamento": campionamento,
            "radius_keypoints_pose (r_k)": r_k,
            "radius_keypoints_mask": radius_keypoints_mask,
            "radius_head_mask (r_h)": r_h,
            "dilatation": dilatation,
            "flip": flip,
            "pairing_mode": pairing_mode,
            "file_structure": "new_structure_with_external_masks"
        },
        "train": {
            "name_file": name_tfrecord_train,
            "list_pz": lista_pz_train,
            "tot": tot_train
        },
        "valid": {
            "name_file": name_tfrecord_valid,
            "list_pz": lista_pz_valid,
            "tot": tot_valid
        },
        "test": {
            "name_file": name_tfrecord_test,
            "list_pz": lista_pz_test,
            "tot": tot_test
        }
    }

    # 保存配置文件
    set_config_path = os.path.join(dir_configuration, 'sets_config.pkl')
    with open(set_config_path, "wb") as f:
        pickle.dump(dic, f)

    dic_history_path = os.path.join(dir_configuration, 'dic_history.pkl')
    with open(dic_history_path, "wb") as f:
        pickle.dump(dic_history, f)

    print("Dataset creation completed successfully!")
    print(f"Output directory: {dir_configuration}")
    print(f"Train samples: {tot_train}")
    print(f"Valid samples: {tot_valid}")
    print(f"Test samples: {tot_test}")
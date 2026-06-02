import cv2
import math
import numpy as np
import tensorflow as tf

from utils.utils_methods import getSparsePose

# STRUCTURAL
# luminoità tra max//3 e -max//3
def _rotate_single(img, angle):
    """输入 [H,W] 或 [H,W,C] 输出同形状"""
    if img.ndim == 2:                      # 掩膜 2 维
        h, w = img.shape
        return cv2.warpAffine(img, cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0),
                              (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    # 3 维图
    h, w, c = img.shape
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0),
                          (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

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



def random_brightness(dic_data, indx_img):
    image = dic_data["I" + indx_img]
    max = np.int64(image.max())
    max_d = max // 3
    min_d = -max_d
    brightness = tf.random.uniform(shape=[1], minval=min_d, maxval=max_d, dtype=tf.dtypes.int64).numpy()

    new_image = image + brightness
    new_image = np.clip(new_image, 0, 32765)
    new_image = new_image.astype(np.uint8)
    dic_data["I" + indx_img] = new_image

    return dic_data

# contrasto tra max// e -max//2
def random_contrast(dic_data, indx_img):
    image = dic_data["I" + indx_img]
    max = np.int64(image.max())
    max_d = max // 2
    min_d = -max_d
    contrast = tf.random.uniform(shape=[1], minval=min_d, maxval=max_d, dtype=tf.dtypes.int64).numpy()

    f = (max + 4) * (contrast + max) / (max * ((max + 4) - contrast))
    alpha = f
    gamma = (max_d) * (1 - f)

    new_image = (alpha * image) + gamma
    new_image = np.clip(new_image, 0, 32765)
    new_image = new_image.astype(np.uint8)
    dic_data["I" + indx_img] = new_image

    return dic_data


def aug_shift(dic_data, type, indx_img, tx=0, ty=0):
    """平移增强 - 完整处理图像、掩膜和关键点"""
    if type == "or":
        assert (ty == 0)
    elif type == "ver":
        assert (tx == 0)

    # 检查图像数据是否存在
    img_key = "I" + indx_img
    if img_key not in dic_data:
        print(f"WARNING: {img_key} not found in dic_data, skipping shift")
        return dic_data

    img = dic_data[img_key]

    #print(f"Debug - aug_shift: 处理 {indx_img}, 输入形状 {img.shape}, 平移(tx={tx}, ty={ty})")

    # 处理图像 - 保持批量维度
    if img.ndim == 4:
        # 批量数据 [batch, H, W, C]
        batch_size, h, w, c = img.shape
        shifted_imgs = []

        for i in range(batch_size):
            img_single = img[i]  # [H, W, C]
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img_shifted = cv2.warpAffine(img_single, M, (w, h),
                                         flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_REPLICATE)
            # 确保形状正确
            if img_shifted.shape != (h, w, c):
                img_shifted = img_shifted.reshape(h, w, c)
            shifted_imgs.append(img_shifted)

        dic_data[img_key] = np.stack(shifted_imgs, axis=0)
        #print(f"Debug - aug_shift: 图像处理后形状 {dic_data[img_key].shape}")

    elif img.ndim == 3:
        # 单张图像 [H, W, C]
        h, w, c = img.shape
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img_shifted = cv2.warpAffine(img, M, (w, h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_REPLICATE)
        if img_shifted.shape != (h, w, c):
            img_shifted = img_shifted.reshape(h, w, c)
        dic_data[img_key] = img_shifted
        #print(f"Debug - aug_shift: 图像处理后形状 {dic_data[img_key].shape}")

    # 处理掩膜 - 保持批量维度
    mask_key = "M" + indx_img
    if mask_key in dic_data:
        mask = dic_data[mask_key]
        #print(f"Debug - aug_shift: 掩膜输入形状 {mask.shape}")

        if mask.ndim == 4:
            # 批量掩膜 [batch, H, W, 1]
            batch_size, h_mask, w_mask, c_mask = mask.shape
            shifted_masks = []

            for i in range(batch_size):
                mask_single = mask[i, :, :, 0]  # [H, W] - 去掉通道维度用于处理
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                mask_shifted = cv2.warpAffine(mask_single, M, (w_mask, h_mask),
                                              flags=cv2.INTER_NEAREST,
                                              borderMode=cv2.BORDER_REPLICATE)
                # 恢复形状 [H, W, 1]
                mask_shifted = mask_shifted[:, :, np.newaxis]
                shifted_masks.append(mask_shifted)

            dic_data[mask_key] = np.stack(shifted_masks, axis=0)
            #print(f"Debug - aug_shift: 掩膜处理后形状 {dic_data[mask_key].shape}")

        elif mask.ndim == 3:
            # 单张掩膜 [H, W, 1]
            h_mask, w_mask, c_mask = mask.shape
            mask_single = mask[:, :, 0]  # [H, W]
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            mask_shifted = cv2.warpAffine(mask_single, M, (w_mask, h_mask),
                                          flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_REPLICATE)
            # 恢复形状 [H, W, 1]
            mask_shifted = mask_shifted[:, :, np.newaxis]
            dic_data[mask_key] = mask_shifted
            #print(f"Debug - aug_shift: 掩膜处理后形状 {dic_data[mask_key].shape}")

    # 处理关键点
    indices_key = "I" + indx_img + "_indices"
    keypoints_key = "I" + indx_img + "_original_keypoints"

    if indices_key in dic_data:
        keypoints_shifted = []
        values_shifted = []

        original_indices = dic_data[indices_key]
        #print(f"Debug - aug_shift: 原始关键点数量: {len(original_indices)}")

        for coordinates in original_indices:
            if len(coordinates) >= 3:
                if len(coordinates) == 3:  # [y, x, id]
                    y, x, id_val = coordinates
                    frame_idx = None
                else:  # 4维 [frame, y, x, id]
                    frame_idx, y, x, id_val = coordinates

                if type == "or":
                    xs = x + tx
                    ys = y
                    if 0 <= xs < w and 0 <= ys < h:
                        if frame_idx is None:
                            keypoints_shifted.append([ys, xs, id_val])
                        else:
                            keypoints_shifted.append([frame_idx, ys, xs, id_val])
                        values_shifted.append(1)
                elif type == "ver":
                    xs = x
                    ys = y + ty
                    if 0 <= xs < w and 0 <= ys < h:
                        if frame_idx is None:
                            keypoints_shifted.append([ys, xs, id_val])
                        else:
                            keypoints_shifted.append([frame_idx, ys, xs, id_val])
                        values_shifted.append(1)

        dic_data[indices_key] = keypoints_shifted
        #print(f"Debug - aug_shift: 平移后关键点数量: {len(keypoints_shifted)}")

        # 更新 values
        values_key = "I" + indx_img + "_values"
        if values_shifted:
            dic_data[values_key] = values_shifted
        else:
            dic_data[values_key] = []
            print("Warning: 没有有效的平移后关键点")

    else:
        print(f"Warning: {indices_key} 不存在，跳过关键点平移")

    return dic_data


def rotate_keypoints(kp_array, angle_deegre, h, w):
    """旋转关键点坐标
    Args:
        kp_array: 关键点数组，可能形状为:
                  - [N*2] 扁平化的关键点 (如 [28,] 表示14个关键点)
                  - [N, 2] 标准关键点格式
                  - [T, N, 2] 多帧关键点
        angle_deegre: 旋转角度
        h: 图像高度
        w: 图像宽度
    """
    xm, ym = w // 2, h // 2
    angle_radias = math.radians(angle_deegre)

    # 确保是 numpy 数组
    kp_array = np.asarray(kp_array)



    # 处理扁平化的一维数组 [N*2]
    if kp_array.ndim == 1:
        # 重塑为 [N, 2]
        if kp_array.size % 2 != 0:
            raise ValueError(f"Cannot reshape keypoints of size {kp_array.size} to [N, 2]")
        kp_reshaped = kp_array.reshape(-1, 2)
        rot = []
        for i in range(len(kp_reshaped)):
            x, y = kp_reshaped[i]
            if y != -1 and x != -1:  # 有效关键点
                xr = (x - xm) * math.cos(angle_radias) - (y - ym) * math.sin(angle_radias) + xm
                yr = (x - xm) * math.sin(angle_radias) + (y - ym) * math.cos(angle_radias) + ym
                rot.append([int(xr), int(yr)])
            else:
                rot.append([x, y])
        # 返回扁平化的结果以保持一致性
        return np.array(rot).flatten()

    # 处理二维数组 [N, 2]
    elif kp_array.ndim == 2 and kp_array.shape[1] == 2:
        rot = []
        for i in range(len(kp_array)):
            x, y = kp_array[i]
            if y != -1 and x != -1:
                xr = (x - xm) * math.cos(angle_radias) - (y - ym) * math.sin(angle_radias) + xm
                yr = (x - xm) * math.sin(angle_radias) + (y - ym) * math.cos(angle_radias) + ym
                rot.append([int(xr), int(yr)])
            else:
                rot.append([x, y])
        return np.array(rot)

    # 处理三维数组 [T, N, 2]
    elif kp_array.ndim == 3 and kp_array.shape[2] == 2:
        rot_frames = []
        for t in range(len(kp_array)):
            rot_frame = []
            for i in range(len(kp_array[t])):
                x, y = kp_array[t, i]
                if y != -1 and x != -1:
                    xr = (x - xm) * math.cos(angle_radias) - (y - ym) * math.sin(angle_radias) + xm
                    yr = (x - xm) * math.sin(angle_radias) + (y - ym) * math.cos(angle_radias) + ym
                    rot_frame.append([int(xr), int(yr)])
                else:
                    rot_frame.append([x, y])
            rot_frames.append(rot_frame)
        return np.array(rot_frames)

    else:
        raise ValueError(f"Unsupported keypoints shape: {kp_array.shape}")


def aug_rotation_angle(dic_data, angle_deegre, indx_img):
    img = dic_data["I" + indx_img]

    #print(f"Debug - aug_rotation_angle: {indx_img}, image shape: {img.shape}")

    # 处理条件图像 (单帧) - 保持不变
    if img.ndim == 4 and img.shape[0] == 1:
        # ... 条件图像处理代码保持不变 ...
        pass

    # 处理目标图像 (多帧)
    elif img.ndim == 4:
        T, h, w, c = img.shape
        new_idx, new_val = [], []

        # 获取完整的关键点数据
        kp_key = "I" + indx_img + "_original_keypoints"
        if kp_key in dic_data:
            full_keypoints = dic_data[kp_key]
            #print(f"Debug - 完整关键点数据形状: {full_keypoints.shape}, 大小: {full_keypoints.size}")

            # 检查关键点数据是否完整
            points_per_frame = 28  # 14关键点 × 2坐标
            expected_size = T * points_per_frame

            if full_keypoints.size != expected_size:
                #print(f"WARNING: 关键点大小不匹配! 期望: {expected_size}, 实际: {full_keypoints.size}")
                return dic_data

            # 旋转每一帧
            for t in range(T):
                # 旋转图像
                dic_data["I" + indx_img][t] = _rotate_single(img[t], angle_deegre)
                dic_data["M" + indx_img][t] = _rotate_single(dic_data["M" + indx_img][t], angle_deegre)

                # 正确获取当前帧的关键点
                start_idx = t * points_per_frame
                end_idx = (t + 1) * points_per_frame
                kp_flat = full_keypoints[start_idx:end_idx]

                #print(f"Debug - 帧 {t} 关键点形状: {kp_flat.shape}, 大小: {kp_flat.size}")

                # 旋转关键点
                rot_kp = rotate_keypoints(kp_flat, angle_deegre, h, w)
                #print(f"Debug - 帧 {t} 旋转后关键点形状: {rot_kp.shape}")

                # 确保关键点是正确形状
                if rot_kp.ndim == 1:
                    rot_kp_reshaped = rot_kp.reshape(-1, 2)
                else:
                    rot_kp_reshaped = rot_kp

                #print(f"Debug - 帧 {t} 重塑后关键点形状: {rot_kp_reshaped.shape}")

                # 生成稀疏表示
                try:
                    idx, val = getSparsePose(rot_kp_reshaped, h, w,
                                             r_k=dic_data['radius_keypoints'], mode='Solid')

                    # 添加帧索引
                    new_idx.extend([[t, r, c, ch] for r, c, ch in idx])
                    new_val.extend(val)
                    #print(f"Debug - 帧 {t} 生成稀疏表示: {len(idx)} 个索引")

                except Exception as e:
                    print(f"Error generating sparse pose for frame {t}: {e}")
                    # 打印关键点内容帮助调试
                    print(f"Keypoints that caused error: {rot_kp_reshaped}")

        dic_data["I" + indx_img + "_indices"] = new_idx
        dic_data["I" + indx_img + "_values"] = new_val

    return dic_data

def aug_flip(dic_data):
    ### Flip vertical pz_condition
    mapping = {0: 0, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 10: 11, 11: 10, 9: 12, 12: 9, 8: 13, 13: 8}
    dic_data["Ic"] = cv2.flip(dic_data["Ic"], 1)
    dic_data["Ic_indices"] = [[i[0], 64 + (64 - i[1]), mapping[i[2]]] for i in dic_data["Ic_indices"]] # flip annotazioni
    dic_data["Mc"] = cv2.flip(dic_data["Mc"], 1)

    ### Flip vertical pz_target
    mapping = {0: 0, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 10: 11, 11: 10, 9: 12, 12: 9, 8: 13, 13: 8}
    dic_data["It"] = cv2.flip(dic_data["It"], 1)
    dic_data["It_indices"] = [[i[0], 64 + (64 - i[1]), mapping[i[2]]] for i in dic_data["It_indices"]]
    dic_data["Mt"] = cv2.flip(dic_data["Mt"], 1)

    return dic_data


import os
import math
import numpy as np
import importlib.util
from PIL import Image
import tensorflow as tf
import cv2
from typing import Union, Tuple, Optional

##################################################
#工具文档
###################################################

def enlarge_keypoint(y, x, id_keypoint, r_k, height, width, mode='Solid'):     #关键点处理与稀疏姿态生成
    """
    Dato un radius di r_k e un punto p di coordinate (x,y) il metodo trova tutti i punti nell'intorno [-r_k, r_k]
    di p. Le coordinate di ognuno di questi punti le salvo in indices e setto il valore 1 (visibile).
    Al termine, ciò che otteniamo è che il punto p viene ingrandito considerando un raggio di r_k.
    :param y
    :param x
    :param id_keypoint
    :param height
    :param width
    :return indices: coordinate dei punti nell'intorno (x,y)
    :return values: valori di visibilità (1) per ognuna delle coordinate definite in indices
    """
    indices = []
    values = []
    for i in range(-r_k, r_k + 1):
        for j in range(-r_k, r_k + 1):
            distance = np.sqrt(float(i ** 2 + j ** 2))
            if y + i >= 0 and y + i < height and x + j >= 0 and x + j < width:
                if 'Solid' == mode and distance <= r_k:
                    indices.append([y + i, x + j, id_keypoint])
                    values.append(1)

    return indices, values

def getSparsePose(keypoints, height, width, r_k, mode='Solid'):
    """
    Andiamo a creare una posa PT sparsa, ingrandendo ogni keypoint di un raggio r_k
    Salviamo i nuovi punti trovati nell'intorno [-r_k, r_k] in indices.
    I values sono settati ad 1 (punto visibile) ed indicano la visibilità degli indices
    I valori di k indicano gli indici di ogni keypoint:
      0 head; 1 right_hand; 2 right_elbow; 3 right_shoulder; 4 neck; 5 left_shoulder; 6 left_elbow;
      7 left_hand; 8 right_foot; 9 right_knee; 10 right_hip; 11 left_hip; 12 left_knee; 13 left_foot

    :return list indices: [ [<coordinata_x>, <coordinata_y>, <indice keypoint>], ... ]
    :return list values: [  1,1,1, ... ]
    :return list shape: [height, width, num keypoints]
    """
    indices = []
    values = []
    for id_keypoint in range(len(keypoints)):
        p = keypoints[id_keypoint]  # coordinate peak ex: "300,200"
        x = p[0]
        y = p[1]
        if x != -1 and y != -1:  # non considero le occlusioni indicate con -1
            ind, val = enlarge_keypoint(y, x, id_keypoint, r_k, height, width, mode)
            indices.extend(ind)
            values.extend(val)
    return indices, values

####################################
# Utils per la creazione dei TFRecords
####################################

def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def float_feature(values):
  """Returns a TF-Feature of float32.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

# src/utils/utils_methods.py
def bytes_feature(value):
    """兼容 str / bytes / list[str] / list[bytes]"""
    if isinstance(value, list):
        value = [v.encode('utf-8') if isinstance(v, str) else v for v in value]
    else:
        if isinstance(value, str):
            value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=value if isinstance(value, list) else [value]))


def format_example(dic):
    """

    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'pz_condition': bytes_feature(dic["pz_condition"].encode('utf-8')),
        'pz_target': bytes_feature(dic["pz_target"].encode('utf-8')),
        'Ic_image_name': bytes_feature(dic["Ic_image_name"].encode('utf-8')),
        'It_image_name': bytes_feature(dic["It_image_name"]),

        'Ic': bytes_feature(dic["Ic"] if isinstance(dic["Ic"], bytes) else dic["Ic"].tobytes()),
        'It': bytes_feature(dic["It"] if isinstance(dic["It"], bytes) else dic["It"].tobytes()),

        'image_format': bytes_feature(dic['image_format']),
        'image_height': int64_feature(dic['image_height']),
        'image_width': int64_feature(dic['image_width']),

        'Ic_original_keypoints': bytes_feature(dic["Ic_original_keypoints"]),
        'It_original_keypoints': bytes_feature(dic["It_original_keypoints"]),
        'shape_len_Ic_original_keypoints': int64_feature(dic["shape_len_Ic_original_keypoints"]),
        'shape_len_It_original_keypoints': int64_feature(dic["shape_len_It_original_keypoints"]),

        'Mc': bytes_feature(dic["Mc"] if isinstance(dic["Mc"], bytes) else dic["Mc"].tobytes()),
        'Mt': bytes_feature(dic["Mt"] if isinstance(dic["Mt"], bytes) else dic["Mt"].tobytes()),

        'Ic_indices': bytes_feature(dic["Ic_indices"]),
        'Ic_values': bytes_feature(dic["Ic_values"]),
        'It_indices': bytes_feature(dic["It_indices"]),
        'It_values': bytes_feature(dic["It_values"]),

        'shape_len_Ic_indices': int64_feature(dic["shape_len_Ic_indices"]),
        'shape_len_Ic_values': int64_feature(dic["shape_len_Ic_values"]),
        'shape_len_It_indices': int64_feature(dic["shape_len_It_indices"]),
        'shape_len_It_values': int64_feature(dic["shape_len_It_values"]),



        'radius_keypoints': int64_feature(dic['radius_keypoints']),
    }))

    return example

####################################
# Utils da utilizzare nel training
####################################
"""
Questo metodo consente di crere una griglia
"""


""""图片
def save_grid(tensor, filename, nrow=8, padding=2, normalize=False, scale_each=False):  ##将一批图像（通常是模型输出或输入）拼接成一个网格图，保存为图片文件，8bit。

    def _grid(tensor, nrow=8, padding=2, normalize=False, scale_each=False):
        nmaps = tensor.shape[0]
        xmaps = min(nrow, nmaps)  # numero di colonne
        ymaps = int(math.ceil(float(nmaps) / xmaps))  # numero di righe
        height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
        num_channels = tensor.shape[3]  # Get number of channels

        grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, num_channels],
                        dtype=np.uint8)
        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                h, h_width = y * height + 1 + padding // 2, height - padding
                w, w_width = x * width + 1 + padding // 2, width - padding
                grid[h:h + h_width, w:w + w_width, :] = tf.reshape(tensor[k], (96, 128, 3))  # Assuming 3 channels
                k = k + 1
        return grid

    ndarr = _grid(tensor, nrow=nrow, padding=padding, normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)
"""

#视频网格
def save_grid(tensor, filename, grid_shape=(2, 10), padding=2):
    """
    将20帧视频保存为2×10的网格图片
    tensor: [batch, T, H, W, 3] 或 [T, H, W, 3]
    grid_shape: (rows, cols) - 对于20帧，使用 (2, 10)
    """
    # 确保 tensor 形状正确
    if len(tensor.shape) == 5:  # [batch, T, H, W, 3]
        # 取第一个样本的所有帧
        frames_tensor = tensor[0]  # [T, H, W, 3]
    elif len(tensor.shape) == 4:  # [T, H, W, 3]
        frames_tensor = tensor
    else:
        raise ValueError(f"不支持的tensor形状: {tensor.shape}")

    T, H, W, C = frames_tensor.shape
    rows, cols = grid_shape

    # 验证帧数匹配网格
    if T != rows * cols:
        #print(f"警告: 帧数{T}不匹配网格{rows}×{cols}={rows * cols}, 自动调整网格")
        # 自动计算合适的网格
        cols = min(10, T)  # 最大10列
        rows = (T + cols - 1) // cols
        #print(f"调整网格为: {rows}×{cols}")

    #print(f"[DEBUG] 保存视频网格: {T}帧, 网格: {rows}×{cols}, 形状: {frames_tensor.shape}")

    # 转换为numpy并确保是uint8
    frames_np = frames_tensor.numpy() if hasattr(frames_tensor, 'numpy') else frames_tensor
    frames_np = frames_np.astype(np.uint8)

    # 计算网格尺寸
    grid_height = rows * H + (rows + 1) * padding
    grid_width = cols * W + (cols + 1) * padding

    # 创建网格图像
    grid = np.zeros([grid_height, grid_width, C], dtype=np.uint8)

    # 填充白色背景（可选）
    grid.fill(255)  # 白色背景

    # 将帧排列到网格中
    for i in range(T):
        row = i // cols
        col = i % cols

        # 计算在网格中的位置
        y_start = row * (H + padding) + padding
        y_end = y_start + H
        x_start = col * (W + padding) + padding
        x_end = x_start + W

        # 放置帧
        grid[y_start:y_end, x_start:x_end, :] = frames_np[i]

        # 可选：添加帧编号标签
        # 这里可以添加文字标注，但需要PIL的ImageDraw

    # 保存图像
    im = Image.fromarray(grid)
    im.save(filename)
    #print(f"视频网格已保存: {filename} (网格: {rows}×{cols})")


# utils.py 里追加 -------------------------------------------------


Tensor = Union[np.ndarray, 'torch.Tensor']   # 兼容 torch / numpy

def tensor2vid(frames: np.ndarray,
               save_path: str,
               fps: int = 10,
               codec: str = 'mp4v',
               is_rgb: bool = True):
    """[T,H,W,3] uint8 -> .mp4"""
    T, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H), True)
    if not out.isOpened():
        raise RuntimeError(f'Cannot open {save_path} for video writing')
    for t in range(T):
        img = frames[t]
        if is_rgb:                      # PIL(RGB) -> OpenCV(BGR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)
    out.release()


def save_grid_with_video(tensor: Tensor,
                         img_path: str,
                         video_path: str,
                         grid_shape: Tuple[int, int] = (2, 10),
                         padding: int = 2,
                         fps: int = 10):
    """
    同时保存：
    1) 网格预览图 -> img_path
    2) 原始帧视频 -> video_path
    """
    # 1. 先保存网格图（复用老代码）
    #save_grid(tensor, img_path, grid_shape, padding)

    # 2. 再写视频
    if len(tensor.shape) == 5:          # [B,T,H,W,C]
        tensor = tensor[0]
    assert tensor.ndim == 4 and tensor.shape[-1] == 3
    frames = (tensor.numpy() if hasattr(tensor, 'numpy') else tensor).astype(np.uint8)
    os.makedirs(os.path.dirname(video_path) or '.', exist_ok=True)
    tensor2vid(frames, video_path, fps=fps)



def import_module(path, name_module):
    """
    Questo metodo mi consente di caricare in maniera dinamica i vari moduli di riferimento per G1, G2, D, Syntetich.
    Ad esempio: models/mono/G1.py
    Ad esempio: dataset/Syntetich.py

    :param str path: path relativo oassoluto di dove rintracciare name_module
    :param str name_module: nome del modulo
    :return python modulo
    """

    spec = importlib.util.spec_from_file_location(name_module, os.path.join(path, name_module + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

import tensorflow as tf

#NORMALIZATION_VALUE_IMAGE = 32765.5  #用于图像数据的归一化处理，以确保像素值在某个特定范围内，16bit
NORMALIZATION_VALUE_IMAGE = 127.5  #用于图像数据的归一化处理，以确保像素值在某个特定范围内,8bit
T = 5
##########################
# PROCESSAMENTO IMMAGINE
##########################

"""""
def process_image(image, mean_pixel, norm=NORMALIZATION_VALUE_IMAGE):  #用于处理图像数据
    return (image - mean_pixel) / norm

def unprocess_image(image, mean_pixel, norm=NORMALIZATION_VALUE_IMAGE):  #用于将处理后的图像还原为原始图像
    return image * norm + mean_pixel

def unprocess_image(image, mean_pixel):
    image = tf.cast(image, tf.float32)
    mean_pixel = tf.cast(mean_pixel, tf.float32)

    # 0. 先保证 mean_pixel 是 [1,1,1,1,3] 或 [1,1,1,3]
    mean_pixel = tf.reshape(mean_pixel, [-1])[:3]
    if len(image.shape) == 5:
        mean_pixel = mean_pixel[None, None, None, None, :]
    else:
        mean_pixel = mean_pixel[None, None, None, :]

    # 1. [-1,1] -> [0,1]
    image = (image + 1.0) * 0.5
    # 2. [0,1] -> [0,255] 再加回均值
    image = image * 255.0 + mean_pixel
    return tf.clip_by_value(image, 0., 255.)
"""""

# 训练/推理都用这个
def process_image(image, mean=None, norm=127.5):
    """
    图像归一化：将 [0, 255] 映射到 [-1, 1]

    Args:
        image: 输入图像，形状 [..., H, W, 3]，范围 [0, 255]
        mean: 每图像均值，形状 [..., 3] 或 None。若为 None 或 0，使用全局归一化
        norm: 归一化系数，默认 127.5

    Returns:
        归一化后的图像，范围 [-1, 1]，float32
    """
    image = tf.cast(image, tf.float32)

    # 判断 mean 是否为 None 或全 0
    if mean is None:
        use_mean = False
    else:
        mean = tf.cast(mean, tf.float32)
        # 检查是否全 0（兼容 [0,0,0] 或 [[0,0,0]] 等情况）
        use_mean = not tf.reduce_all(tf.equal(mean, 0.0))

    if use_mean:
        # 零中心化：减去每图像均值，去除亮度/色彩偏移
        # mean: [..., 3] -> [..., 1, 1, 3] 广播到图像
        ndim_img = len(image.shape)
        ndim_mean = len(mean.shape)

        # 动态 reshape：在 channel 前插入 (ndim_img - ndim_mean) 个 1
        target_shape = tf.shape(mean)[:-1]  # 保留 batch 维度
        for _ in range(ndim_img - ndim_mean):
            target_shape = tf.concat([target_shape, [1]], 0)
        target_shape = tf.concat([target_shape, [3]], 0)

        mean = tf.reshape(mean, target_shape)
        return (image - mean) / norm
    else:
        # 全局归一化：直接映射到 [-1, 1]
        # 0 -> -1, 255 -> 1
        return image / norm - 1.0

def unprocess_image1(image, mean_pixel, norm=127.5):
    image = tf.cast(image, tf.float32)
    mean_pixel = tf.cast(mean_pixel, tf.float32)

    # 只保留最后一维是 3
    tf.assert_equal(tf.shape(mean_pixel)[-1], 3)

    # 无论传进来是 [3]、[1,3] 还是 [B,3]，统一变成 [1,1,1,3]
    mean_pixel = tf.reshape(mean_pixel, [-1, 3])        # [..., 3]
    mean_pixel = tf.reduce_mean(mean_pixel, axis=0)     # [3]
    mean_pixel = tf.reshape(mean_pixel, [1, 1, 1, 3])   # [1,1,1,3]

    image = (image + 1.0) * 0.5 * norm + mean_pixel
    image = tf.clip_by_value(image, 0., 255.)
    image = tf.cast(image, tf.uint8)

    # 交换 R 与 B
    #image = tf.stack([image[..., 2], image[..., 1], image[..., 0]], axis=-1)
    return image


def unprocess_image(image, mean=None, norm=127.5):
    """
    图像反归一化：将 [-1, 1] 恢复为 [0, 255]

    Args:
        image: 网络输出，形状 [..., H, W, 3]，范围 [-1, 1]
        mean: 每图像均值，形状 [..., 3] 或 None。必须与 process_image 一致
        norm: 归一化系数，默认 127.5

    Returns:
        恢复后的图像，范围 [0, 255]，uint8
    """

    image = tf.cast(image, tf.float32)

    # 判断 mean 是否为 None 或全 0
    if mean is None:
        use_mean = False
    else:
        mean = tf.cast(mean, tf.float32)
        use_mean = not tf.reduce_all(tf.equal(mean, 0.0))

    if use_mean:
        # 零中心化的逆操作：(x + 1) * 0.5 * norm + mean
        # 先映射回 [0, norm]，再加 mean，最后到 [0, 255]

        # 动态 reshape mean 以匹配 image
        ndim_img = len(image.shape)
        ndim_mean = len(mean.shape)

        target_shape = tf.shape(mean)[:-1]
        for _ in range(ndim_img - ndim_mean):
            target_shape = tf.concat([target_shape, [1]], 0)
        target_shape = tf.concat([target_shape, [3]], 0)

        mean = tf.reshape(mean, target_shape)
        image = (image + 1.0) * 0.5 * norm + mean
    else:
        # 全局归一化的逆操作：(x + 1) * norm
        # -1 -> 0, 1 -> 255
        image = (image + 1.0) * norm

    # 截断到有效范围并转 uint8
    image = tf.clip_by_value(image, 0.0, 255.0)
    return tf.cast(image, tf.uint8)


# 工具函数
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

##########################


example_description = {

    'pz_condition': tf.io.FixedLenFeature([], tf.string),  # nome del pz condition
    'pz_target': tf.io.FixedLenFeature([], tf.string),  # nome del pz target

    'Ic_image_name': tf.io.FixedLenFeature([], tf.string),  # nome img condition
    'It_image_name': tf.io.FixedLenFeature([], tf.string),  # nome img target
    'Ic': tf.io.FixedLenFeature([], tf.string),  # Immagine di condizione Ic  存储条件图像的二进制数据
    'It': tf.io.FixedLenFeature([], tf.string),  # Immagine target It
    #'It_mask':    tf.io.FixedLenFeature([], tf.string),
    #'Ic_time':    tf.io.FixedLenFeature([], tf.int64),
    #'It_time':    tf.io.FixedLenFeature([], tf.int64),

    'image_height': tf.io.FixedLenFeature([], tf.int64, default_value=192),
    'image_width': tf.io.FixedLenFeature([], tf.int64, default_value=256),

    # valori delle coordinate originali della posa ridimensionati a 96x128
    'Ic_original_keypoints': tf.io.FixedLenFeature((), dtype=tf.string),
    'It_original_keypoints': tf.io.FixedLenFeature((), dtype=tf.string),
    'shape_len_Ic_original_keypoints': tf.io.FixedLenFeature([], tf.int64),
    'shape_len_It_original_keypoints': tf.io.FixedLenFeature([], tf.int64),

    # maschera binaria a radius (r_k) con shape [96, 128, 1]
    'Mc': tf.io.FixedLenFeature([192 * 256 * 1], tf.int64),
    'Mt': tf.io.FixedLenFeature([T * 192 * 256 * 1], tf.int64),

     # Sparse tensor per la posa. Gli indici e i valori considerano il riempimento (ingrandimento) del Keypoints di raggio r_k
    'Ic_indices': tf.io.FixedLenFeature((), dtype=tf.string),
    'Ic_values': tf.io.FixedLenFeature((), dtype=tf.string),
    'It_indices': tf.io.FixedLenFeature((), dtype=tf.string),
    'It_values': tf.io.FixedLenFeature((), dtype=tf.string),


    'shape_len_Ic_indices': tf.io.FixedLenFeature([], tf.int64),
    'shape_len_It_indices': tf.io.FixedLenFeature([], tf.int64),
    'radius_keypoints': tf.io.FixedLenFeature([], tf.int64),
}

# ritorna un TF.data  用于从 TFRecord 文件中读取和解码数据。它处理图像数据、稀疏张量、掩膜和一些附加信息，
# 最终返回一个包含所有解码信息的 TensorFlow 数据集。这可以用于后续的模型训练或评估。
def get_unprocess_dataset(name_tfrecord):
    """
    读取 TFRecord，返回 tf.data.Dataset
    """

    example_description = {
        'pz_condition': tf.io.FixedLenFeature([], tf.string),
        'pz_target': tf.io.FixedLenFeature([], tf.string),
        'Ic_image_name': tf.io.FixedLenFeature([], tf.string),
        'It_image_name': tf.io.FixedLenFeature([], tf.string),
        'Ic': tf.io.FixedLenFeature([], tf.string),
        'It': tf.io.FixedLenFeature([], tf.string),
        'image_format': tf.io.FixedLenFeature([], tf.string),
        'image_height': tf.io.FixedLenFeature([], tf.int64),
        'image_width': tf.io.FixedLenFeature([], tf.int64),
        'Ic_original_keypoints': tf.io.FixedLenFeature([], tf.string),
        'It_original_keypoints': tf.io.FixedLenFeature([], tf.string),
        'shape_len_Ic_original_keypoints': tf.io.FixedLenFeature([], tf.int64),
        'shape_len_It_original_keypoints': tf.io.FixedLenFeature([], tf.int64),
        'Mc': tf.io.FixedLenFeature([], tf.string),
        'Mt': tf.io.FixedLenFeature([], tf.string),
        'Ic_indices': tf.io.FixedLenFeature([], tf.string),
        'Ic_values': tf.io.FixedLenFeature([], tf.string),
        'It_indices': tf.io.FixedLenFeature([], tf.string),
        'It_values': tf.io.FixedLenFeature([], tf.string),
        'shape_len_Ic_indices': tf.io.FixedLenFeature([], tf.int64),
        'shape_len_Ic_values': tf.io.FixedLenFeature([], tf.int64),
        'shape_len_It_indices': tf.io.FixedLenFeature([], tf.int64),
        'shape_len_It_values': tf.io.FixedLenFeature([], tf.int64),
        'radius_keypoints': tf.io.FixedLenFeature([], tf.int64),

    }

    def _decode_function(example_proto):
        example = tf.io.parse_single_example(example_proto, example_description)

        T = 5 #序列长度

        # ---------- 基本字段 ----------
        name_condition = example['Ic_image_name']
        name_target = example['It_image_name']
        pz_condition = example['pz_condition']
        pz_target = example['pz_target']
        radius_keypoints = example['radius_keypoints']

        # ---------- 原始关键点 ----------
        Ic_original_kp = tf.reshape(
            tf.io.decode_raw(example['Ic_original_keypoints'], tf.int64),
            [1, 14, 2])

        It_original_kp = tf.reshape(
            tf.io.decode_raw(example['It_original_keypoints'], tf.int64),
            [T, 14, 2])

        # ---------- 稀疏坐标 ----------
        ic_idx_len = example['shape_len_Ic_indices']
        ic_val_len = example['shape_len_Ic_values']
        it_idx_len = example['shape_len_It_indices']
        it_val_len = example['shape_len_It_values']

        # 使用动态形状reshape
        Ic_indices = tf.reshape(
            tf.io.decode_raw(example['Ic_indices'], tf.int64),  ###'Ic_indices' 处理成了真实
            [ic_idx_len, 4])

        Ic_values = tf.reshape(
            tf.io.decode_raw(example['Ic_values'], tf.float32),
            [ic_val_len])

        It_indices = tf.reshape(
            tf.io.decode_raw(example['It_indices'], tf.int64),
            [it_idx_len, 4])

        It_values = tf.reshape(
            tf.io.decode_raw(example['It_values'], tf.float32),
            [it_val_len])

        # ---------- 图像 ----------
        Ic = tf.reshape(tf.io.decode_raw(example['Ic'], tf.uint8), [1, 192, 256, 3])
        It = tf.reshape(tf.io.decode_raw(example['It'], tf.uint8), [T, 192, 256, 3])

        # ---------- 稀疏姿态张量 ----------
        Pc = tf.SparseTensor(
            indices=Ic_indices,
            values=Ic_values,
            dense_shape=[1, 192, 256, 14])

        Pt = tf.SparseTensor(
            indices=It_indices,
            values=It_values,
            dense_shape=[T, 192, 256, 14])

        # ---------- 掩膜 ----------
        Mc = tf.io.decode_raw(example['Mc'], tf.uint8)
        Mc = tf.reshape(Mc, [1, 192, 256, 1])

        Mt = tf.io.decode_raw(example['Mt'], tf.uint8)
        Mt = tf.reshape(Mt, [T, 192, 256, 1])

        # 获取原始的长度信息
        shape_len_Ic = example['shape_len_Ic_indices']
        shape_len_It = example['shape_len_It_indices']

        # 返回与_preprocess函数匹配的参数列表（移除了It_mask）
        return (Ic, It, Pc, Pt, Mc, Mt,
                pz_condition, pz_target, name_condition, name_target,
                Ic_indices, It_indices,  # indices_0, indices_1
                Ic_values, It_values,     # values_0, values_1
                Ic_original_kp, It_original_kp,       # original_peaks_0, original_peaks_1
                radius_keypoints, shape_len_Ic, shape_len_It)

    dataset = tf.data.TFRecordDataset(name_tfrecord)
    dataset = dataset.map(_decode_function, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def preprocess_dataset(unprocess_dataset):
    def _preprocess(Ic, It, Pc, Pt, Mc, Mt, pz_condition, pz_target, name_condition, name_target,
                    indices_0, indices_1, values_0, values_1, original_peaks_0, original_peaks_1,
                    radius_keypoints, shape_len_Ic, shape_len_It):


        # 1. 掩膜（背景改成 #4D4A46）
        bg_color = tf.constant([77, 74, 70], dtype=tf.uint8)  # RGB(77,74,70)
        Ic = tf.where(Mc > 0, Ic, bg_color)
        It = tf.where(Mt > 0, It, bg_color)
        # 1. 掩膜
        #Ic = Ic * tf.cast(Mc > 0, Ic.dtype)
        #It = It * tf.cast(Mt > 0, It.dtype)


        # 不再计算均值，设为0
        mean_condition = tf.zeros([tf.shape(Ic)[0], 3], dtype=tf.float32)  # [B,3]
        mean_target = tf.zeros([tf.shape(It)[0], 3], dtype=tf.float32)  # [B,3]



        # 3. 归一化（广播自动兼容）
        Ic_proc = process_image(tf.cast(Ic, tf.float32), mean_condition,
                                norm=NORMALIZATION_VALUE_IMAGE)
        It_proc = process_image(tf.cast(It, tf.float32), mean_target,
                                norm=NORMALIZATION_VALUE_IMAGE)

        # 4. 姿态稠密化 & 缩放
        Pt = tf.cast(tf.sparse.to_dense(Pt, default_value=0, validate_indices=False), tf.float16)
        Pt = Pt * 2 - 1        # [0,1] -> [-1,1]

        # 5. 掩膜转半精度
        Mt = tf.cast(Mt, tf.float16)
        Mc = tf.cast(Mc, tf.float16)

        # 6. 返回（mean 也带出去，推理时原样传回 unprocess_image）
        return (Ic_proc, It_proc, Pt, Mt, Mc,
                pz_condition, pz_target, name_condition, name_target,
                mean_condition, mean_target)

    return unprocess_dataset.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)






"""""   测试颜色
# 1. 只抓 1 个样本
raw_ds = get_unprocess_dataset('/media/jy/36623f40-fafa-4a13-879d-8d47450b13cc/jy/gan/video generation/data/Syntetich_complete/tfrecord/testing_configuration/Syntetich_test.tfrecord').take(1)
proc_ds = preprocess_dataset(raw_ds)

for raw, proc in zip(raw_ds, proc_ds):
    (Ic_raw, It_raw, Pc_raw, Pt_raw, Mc_raw, Mt_raw,
     pz_condition, pz_target, name_c, name_t,
     Ic_idx, It_idx, Ic_val, It_val,
     Ic_kp, It_kp,
     radius_keypoints, shape_len_Ic, shape_len_It) = raw

    (Ic_proc, _, _, _, _, _, _, _, _, mean_c, _) = proc

    # 2. 先看原始图颜色对不对
    import matplotlib.pyplot as plt, cv2, numpy as np
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1)
    plt.title('RAW uint8 RGB')
    plt.imshow(Ic_raw[0].numpy()); plt.axis('off')

    # 3. 用你的 unprocess 还原
    Ic_bgr = unprocess_image(Ic_proc, mean_c)[0]   # 函数里已经 swap 成 BGR
    Ic_rgb = cv2.cvtColor(Ic_bgr.numpy(), cv2.COLOR_BGR2RGB)
    plt.subplot(1,3,2)
    plt.title('after unprocess (BGR→RGB)')
    plt.imshow(Ic_rgb); plt.axis('off')

    # 4. 把 mean 摘掉，只反归一化
    Ic_no_mean = (Ic_proc + 1.0) * 127.5
    Ic_no_mean = tf.clip_by_value(Ic_no_mean, 0, 255)
    Ic_no_mean = tf.cast(Ic_no_mean, tf.uint8)
    plt.subplot(1,3,3)
    plt.title('no mean, only (x+1)*127.5')
    plt.imshow(Ic_no_mean[0].numpy()); plt.axis('off')
    plt.show()

    # 5. 打印关键统计量
    print('Ic_proc  min/max:', float(tf.reduce_min(Ic_proc)), float(tf.reduce_max(Ic_proc)))
    print('mean_c (RGB)     :', mean_c[0].numpy())
    break
    
"""""
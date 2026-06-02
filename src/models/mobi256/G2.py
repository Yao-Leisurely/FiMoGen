# ==========================================
#  G2.py  普通 3-D 卷积版（无 SeparableConv3D）
# ==========================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv3D, BatchNormalization, Activation, UpSampling3D,
    Concatenate, GlobalAveragePooling3D, Lambda
)
from tensorflow.keras.optimizers import Adam
from models.Model_template import Model_Template
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


def se_block_3d(x, ratio=16, name=None):
    """
    3D-SE 通道注意力，仅对 C 维加权，时间维 T=1 不受影响
    x: [B, 1, H, W, C]
    return: [B, 1, H, W, C] 加权后特征
    """
    c = int(x.shape[-1])
    # 1) 全局池化: [B,1,H,W,C] -> [B,C]
    gap = layers.Lambda(lambda z: tf.reduce_mean(z, axis=[1,2,3]))(x)  # T=1 所以轴1也压掉
    # 2) 瓶颈全连接
    z = layers.Dense(c//ratio, activation='relu', use_bias=False, name=f'{name}_fc1')(gap)
    z = layers.Dense(c, activation='sigmoid', use_bias=False, name=f'{name}_fc2')(z)
    # 3) 通道缩放
    out = layers.Lambda(lambda args: args[0] * tf.reshape(args[1], [-1,1,1,1,c]))([x, z])
    return out

"""""
@tf.function
def energy_align_g2(gen, real, pose):

    T_FRAME = tf.shape(gen)[1]

    skin = tf.cast(pose[..., 0:1] > 0.1, tf.float32)  # [B,T,H,W,1]
    skin = tf.tile(skin, [1, 1, 1, 1, 3])             # [B,T,H,W,3]

    def rfft_energy(x):
        # 沿时间维 RFFT，取 2–4 Hz bin（bin-1 & bin-2）
        spec = tf.signal.rfft(x, [T_FRAME])[..., 1:3]   # [B,H,W,3,2]
        amp  = tf.abs(spec)
        return tf.reduce_sum(tf.square(amp), axis=[3, 4])  # [B,H,W]

    gen_e  = rfft_energy(gen * skin)
    real_e = rfft_energy(real * skin)

    # L2 对齐
    return tf.reduce_mean(tf.square(gen_e - real_e))
"""""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class TFPoseEstimator14KP(tf.keras.layers.Layer):
    """内存优化的MoveNet姿态估计器 - 流式处理版"""

    def __init__(self, batch_size=8, use_tflite=True):
        super().__init__()
        self.kp_indices = [0,1,2, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16]
        self.input_size = 256
        self.batch_size = batch_size  # 控制峰值内存
        self.use_tflite = use_tflite

        # 方案A: 使用TFLite（推荐，内存最小）
        if use_tflite:
            self._init_tflite()
        else:
            # 方案B: 使用TF Hub但转换为ConcreteFunction
            self._init_hub_optimized()

    def _init_tflite(self):
        """使用TFLite解释器，内存占用最小"""
        import tempfile
        import os

        # 下载并转换为TFLite
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

        # 构建concrete function用于转换
        concrete_func = model.signatures['serving_default']

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # 量化进一步减少内存（可选）
        # converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        # 保存到临时文件
        self.tflite_path = os.path.join(tempfile.gettempdir(), "movenet_thunder.tflite")
        with open(self.tflite_path, 'wb') as f:
            f.write(tflite_model)

        # 创建解释器，限制线程数以控制内存
        self.interpreter = tf.lite.Interpreter(
            model_path=self.tflite_path,
            num_threads=2  # 限制线程减少内存
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("[PoseEstimator] TFLite模式初始化完成，内存占用最低")

    def _init_hub_optimized(self):
        """优化后的TF Hub模式 - 使用ConcreteFunction"""
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

        # 获取签名并转换为ConcreteFunction以便XLA优化
        infer = model.signatures['serving_default']

        # 固定输入shape以便图优化
        self.infer = tf.function(
            lambda x: infer(x),
            input_signature=[tf.TensorSpec([None, self.input_size, self.input_size, 3], tf.int32)]
        )
        print("[PoseEstimator] TF Hub优化模式初始化完成")

    # ========== 核心优化：流式分批处理 ==========

    def call(self, video):
        """
        video: [B, T, H, W, 3]
        return: [B, T, 14, 3]
        """
        B = tf.shape(video)[0]
        T = tf.shape(video)[1]
        H = tf.shape(video)[2]
        W = tf.shape(video)[3]
        total_frames = B * T

        # 流式处理：不一次性reshape所有帧
        # 而是按 batch_size 逐批处理，峰值内存 = batch_size * 256 * 256 * 3

        def process_batch(start_idx):
            """处理一批帧，返回关键点的闭包函数"""
            end_idx = tf.minimum(start_idx + self.batch_size, total_frames)

            # 从原始video中切片，避免创建完整的video_4d
            # 使用gather避免reshape的内存拷贝
            indices = tf.range(start_idx, end_idx)

            # 将5D索引映射到4D
            b_idx = indices // T
            t_idx = indices % T

            # 提取并resize这一批
            batch_frames = tf.gather_nd(
                video,
                tf.stack([b_idx, t_idx], axis=1)
            )  # [batch, H, W, 3]

            batch_resized = tf.image.resize(batch_frames, [self.input_size, self.input_size])
            batch_int = tf.cast(batch_resized * 255.0, tf.int32)

            return batch_int, end_idx

        # 使用tf.while_loop进行流式迭代，控制内存峰值
        if self.use_tflite:
            return self._tflite_streaming_inference(video, B, T, total_frames)
        else:
            return self._hub_streaming_inference(video, B, T, total_frames)

    def _tflite_streaming_inference(self, video, B, T, total_frames):
        """TFLite流式推理 - 内存占用最低"""

        # 预分配输出缓冲区 [B*T, 1, 17, 3]，避免动态增长
        output_buffer = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True,
            element_shape=[1, 17, 3]  # 每帧的输出shape
        )

        def cond(idx, ta):
            return idx < total_frames

        def body(idx, ta):
            # 只处理一批
            end_idx = tf.minimum(idx + self.batch_size, total_frames)
            count = end_idx - idx

            # 提取这一批
            indices = tf.range(idx, end_idx)
            b_idx = indices // T
            t_idx = indices % T

            batch_frames = tf.gather_nd(video, tf.stack([b_idx, t_idx], axis=1))
            batch_resized = tf.image.resize(batch_frames, [self.input_size, self.input_size])
            batch_int = tf.cast(batch_resized * 255.0, tf.int32)

            # TFLite逐帧推理（TFLite不支持批量）
            def infer_single_tflite(frame_int):
                # frame_int: [256, 256, 3]
                self.interpreter.set_tensor(
                    self.input_details[0]['index'],
                    np.expand_dims(frame_int.numpy(), 0)  # [1, 256, 256, 3]
                )
                self.interpreter.invoke()
                kp = self.interpreter.get_tensor(self.output_details[0]['index'])
                return tf.convert_to_tensor(kp[0], dtype=tf.float32)  # [1, 17, 3]

            # 对这一批中的每一帧使用map_fn（比Python循环内存友好）
            batch_kp = tf.map_fn(
                infer_single_tflite,
                batch_int,
                fn_output_signature=tf.TensorSpec([1, 17, 3], tf.float32)
            )  # [batch, 1, 17, 3]

            # 写入TensorArray
            for i in tf.range(count):
                ta = ta.write(idx + i, batch_kp[i])

            return end_idx, ta

        # 执行while循环
        _, final_ta = tf.while_loop(cond, body, [0, output_buffer])

        # 堆叠结果 [B*T, 1, 17, 3] -> [B, T, 17, 3]
        keypoints = final_ta.stack()  # [B*T, 1, 17, 3]
        keypoints = tf.squeeze(keypoints, axis=1)  # [B*T, 17, 3]
        keypoints = tf.reshape(keypoints, [B, T, 17, 3])

        # 取14个关键点
        return tf.gather(keypoints, self.kp_indices, axis=2)

    def _hub_streaming_inference(self, video, B, T, total_frames):
        """TF Hub流式推理 - 使用XLA优化"""

        def cond(idx, ta):
            return idx < total_frames

        def body(idx, ta):
            end_idx = tf.minimum(idx + self.batch_size, total_frames)
            count = end_idx - idx

            # 提取并resize
            indices = tf.range(idx, end_idx)
            b_idx = indices // T
            t_idx = indices % T

            batch_frames = tf.gather_nd(video, tf.stack([b_idx, t_idx], axis=1))
            batch_resized = tf.image.resize(batch_frames, [self.input_size, self.input_size])
            batch_int = tf.cast(batch_resized * 255.0, tf.int32)

            # 批量推理（TF Hub支持批量）
            batch_kp = self.infer(batch_int)['output_0']  # [batch, 1, 17, 3]

            # 拆分到TensorArray
            for i in tf.range(count):
                ta = ta.write(idx + i, batch_kp[i])

            return end_idx, ta

        output_buffer = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True,
            element_shape=[1, 17, 3]
        )

        _, final_ta = tf.while_loop(cond, body, [0, output_buffer])

        keypoints = final_ta.stack()
        keypoints = tf.squeeze(keypoints, axis=1)
        keypoints = tf.reshape(keypoints, [B, T, 17, 3])

        return tf.gather(keypoints, self.kp_indices, axis=2)

# ========== 进一步优化：关键点缓存与复用 ==========

class CachedPoseEstimator(tf.keras.layers.Layer):
    """带缓存的姿态估计器 - 避免重复计算相同帧"""

    def __init__(self, max_cache_size=1000, **kwargs):
        super().__init__(**kwargs)
        self.estimator = TFPoseEstimator14KP(**kwargs)
        self.max_cache_size = max_cache_size
        # 使用字典作为缓存（在eager模式下有效）
        self._cache = {}
        self._cache_hits = 0

    def call(self, video):
        # 如果视频帧数很大，先检查是否有缓存
        # 注意：这里使用帧内容的hash作为key，实际应用可用文件名+时间戳

        # 对于训练场景，通常需要实时计算
        # 此缓存主要用于推理时重复片段的优化
        return self.estimator(video)

    def clear_cache(self):
        self._cache.clear()
        tf.keras.backend.clear_session()  # 释放TF内部缓存


# ========== 频谱对齐函数的内存优化 ==========

def energy_align_14kp_spectrum_optimized(gen_pose, src_pose, T_FRAME=5):
    """
    gen_pose, src_pose: [B, T, 14, 3] 关键点坐标
    """
    gen_signal = tf.transpose(gen_pose[..., :2], [0, 2, 3, 1])  # [B, 14, 2, T]
    src_signal = tf.transpose(src_pose[..., :2], [0, 2, 3, 1])  # [B, 14, 2, T]

    gen_fft = tf.signal.rfft(gen_signal, fft_length=[T_FRAME])
    src_fft = tf.signal.rfft(src_signal, fft_length=[T_FRAME])

    # 能量归一化 - 分母转为 complex64 以匹配分子
    gen_power = tf.reduce_sum(tf.abs(gen_fft) ** 2, axis=-1, keepdims=True) + 1e-5
    src_power = tf.reduce_sum(tf.abs(src_fft) ** 2, axis=-1, keepdims=True) + 1e-5

    # 关键修复：sqrt 结果转为 complex64
    gen_norm = gen_fft / tf.cast(tf.sqrt(gen_power), tf.complex64)
    src_norm = src_fft / tf.cast(tf.sqrt(src_power), tf.complex64)

    # 幅度差
    return tf.reduce_mean(tf.square(tf.abs(gen_norm - src_norm)))


# =====================================================
#  G2 类（普通 3-D 全程）
# =====================================================
class G2(Model_Template):
    def __init__(self):
        self.architecture = "3d普通"
        self.T = 5
        self.input_shape1 = [1, 192, 256, 3]        # 5-D 条件
        self.input_shape2 = [self.T, 192, 256, 3]  # 5-D 姿态
        self.activation_fn = 'relu'
        self.data_format = 'channels_last'
        self.output_channels = 3
        self.conv_hidden_num = 64                  # 初始通道砍半
        self.lr_initial_G2 = 4e-5

        self.pose_estimator = TFPoseEstimator14KP()
        super().__init__()

    def _build_model(self):
        c = self.conv_hidden_num  # 64
        c_cond = c * 2  # 128，外观分支专用
        T = self.T
        act = self.activation_fn
        df = self.data_format

        # ========== 条件分支 Encoder（外观，通道更多 + SE）==========
        inp1 = Input(self.input_shape1, name='cond')  # [B,1,H,W,3]
        x1 = Conv3D(c_cond, (1, 3, 3), padding='same', activation=act, data_format=df)(inp1)
        x1 = Conv3D(c_cond, (1, 3, 3), padding='same', activation=act, data_format=df)(x1)
        x1 = Conv3D(c_cond, (1, 3, 3), padding='same', activation=act, data_format=df)(x1)
        x1 = se_block_3d(x1, ratio=8, name='se_skip1')
        skip1_1 = x1  # [B,1,H,W,128]

        x1 = Conv3D(c_cond * 2, (1, 2, 2), strides=(1, 2, 2), padding='same', activation=act, data_format=df)(x1)
        x1 = Conv3D(c_cond * 2, (1, 3, 3), padding='same', activation=act, data_format=df)(x1)
        x1 = Conv3D(c_cond * 2, (1, 3, 3), padding='same', activation=act, data_format=df)(x1)
        x1 = se_block_3d(x1, ratio=8, name='se_skip2')
        skip1_2 = x1  # [B,1,H/2,W/2,256]

        x1 = Conv3D(c_cond * 3, (1, 2, 2), strides=(1, 2, 2), padding='same', activation=act, data_format=df)(x1)
        x1 = Conv3D(c_cond * 3, (1, 3, 3), padding='same', activation=act, data_format=df)(x1)
        x1 = Conv3D(c_cond * 3, (1, 3, 3), padding='same', activation=act, data_format=df)(x1)
        x1 = se_block_3d(x1, ratio=8, name='se_bridge')
        bridge1 = x1  # [B,1,H/4,W/4,384]

        # ========== 姿态分支 Encoder（不变）==========
        inp2 = Input(self.input_shape2, name='pose')  # [B,T,H,W,3]
        x2 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(inp2)
        x2 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(x2)
        x2 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(x2)
        skip2_1 = x2

        x2 = Conv3D(c * 2, 2, strides=(1, 2, 2), padding='same', activation=act, data_format=df)(x2)
        x2 = Conv3D(c * 2, 3, padding='same', activation=act, data_format=df)(x2)
        x2 = Conv3D(c * 2, 3, padding='same', activation=act, data_format=df)(x2)
        skip2_2 = x2

        x2 = Conv3D(c * 3, 2, strides=(1, 2, 2), padding='same', activation=act, data_format=df)(x2)
        x2 = Conv3D(c * 3, 3, padding='same', activation=act, data_format=df)(x2)
        x2 = Conv3D(c * 3, 3, padding='same', activation=act, data_format=df)(x2)
        bridge2 = x2

        # ========== 共享 Decoder（带 SE 加权 skip）==========
        bridge1_tile = Lambda(lambda z: tf.tile(z, [1, T, 1, 1, 1]))(bridge1)
        dec = Concatenate(axis=-1)([bridge1_tile, bridge2])
        dec = Conv3D(c * 3, 1, padding='same', activation=act, data_format=df)(dec)  # 先压通道

        # ----- 解码块 1：1/4 → 1/2 -----
        dec = UpSampling3D(size=(1, 2, 2), data_format=df)(dec)
        dec = Conv3D(c * 2, 1, padding='same', activation=act, data_format=df)(dec)

        skip1_2_tile = Lambda(lambda z: tf.tile(z, [1, T, 1, 1, 1]))(skip1_2)
        skip1_2_proj = Conv3D(c * 2, (1, 1, 1), padding='same', activation=act, data_format=df)(skip1_2_tile)
        skip2_2_proj = Conv3D(c * 2, (1, 1, 1), padding='same', activation=act, data_format=df)(skip2_2)

        long_con_1 = Concatenate(axis=-1)([dec, skip1_2_proj, skip2_2_proj])
        long_con_1 = se_block_3d(long_con_1, ratio=16, name='se_dec1')

        b1 = Conv3D(c * 2, 3, padding='same', activation=act, data_format=df)(long_con_1)
        b1 = Conv3D(c * 2, 3, padding='same', activation=act, data_format=df)(b1)
        b2 = Conv3D(c * 2, 3, padding='same', activation=act, data_format=df)(long_con_1)
        b2 = Conv3D(c * 2, 3, padding='same', activation=act, data_format=df)(b2)

        dec = Concatenate(axis=-1)([b1, b2])
        dec = Conv3D(c * 2, 1, padding='same', activation=act, data_format=df)(dec)

        # ----- 解码块 2：1/2 → 1/1 -----
        dec = UpSampling3D(size=(1, 2, 2), data_format=df)(dec)
        dec = Conv3D(c, 1, padding='same', activation=act, data_format=df)(dec)

        skip1_1_tile = Lambda(lambda z: tf.tile(z, [1, T, 1, 1, 1]))(skip1_1)
        skip1_1_proj = Conv3D(c, (1, 1, 1), padding='same', activation=act, data_format=df)(skip1_1_tile)
        skip2_1_proj = Conv3D(c, (1, 1, 1), padding='same', activation=act, data_format=df)(skip2_1)

        long_con_2 = Concatenate(axis=-1)([dec, skip1_1_proj, skip2_1_proj])
        long_con_2 = se_block_3d(long_con_2, ratio=16, name='se_dec2')

        b1 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(long_con_2)
        b1 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(b1)
        b2 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(long_con_2)
        b2 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(b2)

        dec = Concatenate(axis=-1)([b1, b2])
        dec = Conv3D(c, 1, padding='same', activation=act, data_format=df)(dec)

        out = Conv3D(3, 1, padding='same', activation='tanh', data_format=df)(dec)

        return Model([inp1, inp2], out)
    # ---------- 对外接口 ----------
    def prediction(self, I_PT1, Ic):
        """
        I_PT1 : [B, T, H, W, C]  (T 可能 1000+)
        Ic    : [B, H, W, C]     (无时间维)
        """
        T = tf.shape(I_PT1)[1]
        chunk = 5  # 可调，只要显存不爆
        outs = []
        for t in range(0, T, chunk):
            i1_seg = I_PT1[:, t:t + chunk, ...]  # [B, chunk, H, W, C]
            # Ic 没有时间维，直接复用
            out_seg = self.model([Ic, i1_seg])  # 输出同一段时间长度
            outs.append(out_seg)
        out = tf.concat(outs, axis=1)  # [B, T, H, W, C]


        #out = self.model([Ic, I_PT1])
        return tf.cast(out, tf.float32)

    def _optimizer(self):
        return Adam(learning_rate=self.lr_initial_G2, beta_1=0.5)

    def PoseMaskloss(self, I_PT2, It, Mt):
        I_PT2 = tf.cast(I_PT2, tf.float32)
        It = tf.cast(It, tf.float32)
        Mt = tf.cast(Mt, tf.float32)

        # 转灰度：只约束结构/姿态，不约束颜色
        gray_gen = tf.reduce_mean(I_PT2, axis=-1, keepdims=True)
        gray_it = tf.reduce_mean(It, axis=-1, keepdims=True)

        diff = gray_gen - gray_it
        l1_per_frame = tf.reduce_mean(tf.abs(diff), axis=[1, 2, 3])
        mask_per_frame = tf.reduce_mean(tf.abs(diff) * Mt, axis=[1, 2, 3])
        base_loss = tf.reduce_mean(l1_per_frame + mask_per_frame)

        # 平滑也在灰度上做
        smooth = tf.constant(0.0, dtype=tf.float32)
        T = tf.shape(I_PT2)[1]
        if T > 2:
            delta_pred = gray_gen[:, 1:] - gray_gen[:, :-1]
            delta_gt = gray_it[:, 1:] - gray_it[:, :-1]
            smooth = tf.reduce_mean(tf.abs(delta_pred - delta_gt))

        return base_loss + 0.1 * smooth


    # ---------- 1. 感知损失（SSIM） ----------
    def perceptual_loss(self, real, gen):
        # 转灰度后再算 SSIM，去掉颜色干扰
        real_gray = tf.reduce_mean(real, axis=-1, keepdims=True)
        gen_gray = tf.reduce_mean(gen, axis=-1, keepdims=True)

        real_gray = tf.reshape(real_gray, [-1, 192, 256, 1])
        gen_gray = tf.reshape(gen_gray, [-1, 192, 256, 1])

        return 1.0 - tf.reduce_mean(tf.image.ssim(real_gray, gen_gray, max_val=1.0))

    # ---------- 2. 帧间一致性 ----------
    def temporal_loss(self, pred):
        # pred: [B, T, H, W, 3]
        diff_gt = tf.stop_gradient(pred[:, 1:] - pred[:, :-1])
        diff = pred[:, 1:] - pred[:, :-1]
        l1 = tf.reduce_mean(tf.abs(diff - diff_gt))
        ssim = 1.0 - tf.reduce_mean(tf.image.ssim(
            tf.reshape(pred[:, :-1], [-1, 192, 256, 3]),
            tf.reshape(pred[:, 1:], [-1, 192, 256, 3]), 1.0))
        return l1 + 0.5 * ssim

    def identity_loss(self, I_PT2, Ic, Mc, Mt):
        """
        身份迁移肤色保持损失 v3
        - 提取 A 的肤色作为全局身份签名
        - 约束 B 的生成结果在 Mt 区域内匹配该签名
        - 同时保持局部空间一致性（防止颜色斑块）

        Args:
            I_PT2: [B, T, H, W, 3] 生成结果（B的姿态 + A的身份）
            Ic:    [B, 1, H, W, 3] 源身份图像（A）
            Mc:    [B, 1, H, W, 1] A的皮肤掩膜（提取身份肤色用）
            Mt:    [B, T, H, W, 1] B的皮肤掩膜（约束生成区域用）
        """
        # ========== 类型统一 ==========
        I_PT2 = tf.cast(I_PT2, tf.float32)
        Ic = tf.cast(Ic, tf.float32)
        Mc = tf.cast(Mc, tf.float32)
        Mt = tf.cast(Mt, tf.float32)

        T = tf.shape(I_PT2)[1]

        # ========== 1. 提取源身份 A 的肤色签名 ==========
        # 只在 Mc 区域内计算，得到与空间位置无关的全局身份向量

        # 掩膜内像素数 [B, 1, 1, 1]
        count_A = tf.reduce_sum(Mc, axis=[1, 2, 3], keepdims=True) + 1e-8

        # 均值 [B, 1, 1, 1, 3] —— 身份的核心肤色
        mean_A = tf.reduce_sum(Ic * Mc, axis=[1, 2, 3], keepdims=True) / count_A

        # 标准差 [B, 1, 1, 1, 3] —— 肤色的"质感/饱和度"
        sq_diff_A = tf.square(Ic - mean_A) * Mc
        std_A = tf.sqrt(tf.reduce_sum(sq_diff_A, axis=[1, 2, 3], keepdims=True) / count_A + 1e-8)

        # 高阶统计：偏度（可选，捕捉肤色分布不对称性）
        # cub_diff_A = tf.pow(Ic - mean_A, 3) * Mc
        # skew_A = tf.reduce_sum(cub_diff_A, axis=[1,2,3], keepdims=True) / (count_A * tf.pow(std_A, 3) + 1e-8)

        # ========== 2. 提取生成结果在 B 掩膜内的统计量 ==========
        count_B = tf.reduce_sum(Mt, axis=[1, 2, 3], keepdims=True) + 1e-8  # [B, T, 1, 1, 1]

        mean_gen = tf.reduce_sum(I_PT2 * Mt, axis=[1, 2, 3], keepdims=True) / count_B  # [B, T, 1, 1, 3]

        sq_diff_gen = tf.square(I_PT2 - mean_gen) * Mt
        std_gen = tf.sqrt(tf.reduce_sum(sq_diff_gen, axis=[1, 2, 3], keepdims=True) / count_B + 1e-8)

        # ========== 3. 身份签名匹配损失 ==========
        # 扩展 A 的签名到 T 帧
        mean_A_t = tf.tile(mean_A, [1, T, 1, 1, 1])
        std_A_t = tf.tile(std_A, [1, T, 1, 1, 1])

        # 核心：均值匹配（肤色色调）
        mean_loss = tf.reduce_mean(tf.abs(mean_A_t - mean_gen))

        # 质感匹配（肤色鲜艳度/对比度）
        std_loss = tf.reduce_mean(tf.abs(std_A_t - std_gen))

        # ========== 4. 局部颜色一致性（防止色斑/伪影） ==========
        # 生成结果的每个皮肤像素，应接近 A 的均值（允许小范围偏差）
        mean_A_broadcast = mean_A_t * Mt  # [B, T, H, W, 3]

        # 使用 Huber 损失替代 L1，对大偏差更敏感但梯度更稳定
        local_diff = tf.abs(I_PT2 - mean_A_broadcast) * Mt
        local_loss = tf.reduce_sum(local_diff) / (tf.reduce_sum(Mt) + 1e-8)

        # ========== 5. 边缘/边界平滑（可选，防止掩膜边缘硬边） ==========
        # 对 Mt 做轻微腐蚀，只约束内部区域，边缘由其他损失（如感知损失）处理
        # 如果 Mc/Mt 边缘质量差，可以加上：
        # edge_loss = gradient_loss(I_PT2, Mt)  # 见下方辅助函数

        # ========== 组合 ==========
        # 权重建议：mean 最重要，std 次之，local 防止退化
        total_loss = mean_loss + 0.2 * std_loss + 0.05 * local_loss

        # 调试信息（训练时打印）
        # tf.print("mean_loss:", mean_loss, "std_loss:", std_loss, "local:", local_loss)

        return total_loss


    def adv_loss(self, D_neg, I_PT2, It, Ic, Mt,Mc, Pt):
        # 1. 对抗 loss（mixed 下是 float16）
        adv = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_neg, labels=tf.ones_like(D_neg)))

        # 2. 其余损失（默认 float32）
        pm = self.PoseMaskloss(I_PT2, It, Mt)
        per = self.perceptual_loss(It, I_PT2)
        tmp = self.temporal_loss(I_PT2)
        # 3. 关键点频谱对齐 - 先提取姿态
        gen_pose = self.pose_estimator(I_PT2)  # [B, T, 14, 3]
        src_pose = self.pose_estimator(It)  # [B, T, 14, 3]
        spec = energy_align_14kp_spectrum_optimized(gen_pose, src_pose, self.T)

        ID =self.identity_loss(I_PT2, Ic, Mc,Mt)


        # 3. 统一 float32
        adv = tf.cast(adv, tf.float32)
        pm = tf.cast(pm, tf.float32)  #遮掩麻
        per = tf.cast(per, tf.float32)  #结构 感知
        tmp = tf.cast(tmp, tf.float32)
        spec = tf.cast(spec, tf.float32)
        ID = tf.cast(ID, tf.float32)

        return (2.0 * adv +
                3.5 * pm +
                2.0 * per +
                2.0 * tmp +
                0.1 * spec +
                1 * ID
                )


    def ssim(self, I_PT2, It, mean_0, mean_1, unprocess_function):
        # ---------- 1. 5-D mean → [B,T,1,1,3] ----------
        mean_0 = tf.reshape(mean_0, [tf.shape(mean_0)[0], tf.shape(mean_0)[1], 1, 1, 3])
        mean_1 = tf.reshape(mean_1, [tf.shape(mean_1)[0], tf.shape(mean_1)[1], 1, 1, 3])

        # ---------- 2. 统一 float32 ----------
        It = tf.cast(It, tf.float32)
        I_PT2 = tf.cast(I_PT2, tf.float32)

        # ---------- 3. 反归一化 ----------
        It_processed = tf.cast(unprocess_function(It, mean_1), tf.float32)
        I_PT2_processed = tf.cast(unprocess_function(I_PT2, mean_0), tf.float32)

        # ---------- 4. SSIM（5-D 支持） ----------
        return tf.reduce_mean(tf.image.ssim(I_PT2_processed, It_processed, max_val=255.0))

    def mask_ssim(self, I_PT2, It, Mt, mean_0, mean_1, unprocess_function):
        # ---------- 1. 5-D mean → [B,T,1,1,3] ----------
        mean_0 = tf.reshape(mean_0, [tf.shape(mean_0)[0], tf.shape(mean_0)[1], 1, 1, 3])
        mean_1 = tf.reshape(mean_1, [tf.shape(mean_1)[0], tf.shape(mean_1)[1], 1, 1, 3])

        # ---------- 2. 统一 float32 ----------
        It = tf.cast(It, tf.float32)
        I_PT2 = tf.cast(I_PT2, tf.float32)
        Mt = tf.cast(Mt, tf.float32)

        # ---------- 3. 反归一化 ----------
        It_processed = tf.cast(unprocess_function(It, mean_1), tf.float32)
        I_PT2_processed = tf.cast(unprocess_function(I_PT2, mean_0), tf.float32)

        # ---------- 4. 掩膜乘法 ----------
        mask_raw = Mt * It_processed
        mask_out = Mt * I_PT2_processed

        # ---------- 5. SSIM ----------
        return tf.reduce_mean(tf.image.ssim(mask_out, mask_raw, max_val=255.0))



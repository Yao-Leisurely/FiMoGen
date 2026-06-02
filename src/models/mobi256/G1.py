import numpy as np
import tensorflow as tf
from numba.core.typing.builtins import Print
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Activation, Concatenate
from models.Model_template import Model_Template
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose

from tensorflow.keras import layers
from tensorflow.keras.layers import (Input, Conv3D, Conv2D, UpSampling3D,
                                     Reshape, Add, Activation, Concatenate,
                                     Dense, Lambda)
from tensorflow.keras.optimizers import Adam
from models.Model_template import Model_Template  # 你的基类




# ===================== 3D 工具块 =====================
def conv3d_pad(x, filters, k=(3, 3, 3), s=(1, 1, 1), d=(1, 1, 1), act='relu'):
    """k=(kt,kh,kw) 只在空间下采样，时间 stride=1"""
    x = Conv3D(filters, k, strides=s, dilation_rate=d,
               padding='same', data_format='channels_last')(x)
    if act: x = Activation(act)(x)
    return x


def res3d_block(x, nf, k=(3, 3, 3), d=(1, 1, 1)):
    """3D 残差：主路两个 3D 卷积，跳连 1×1×1 投影（可选）"""
    res = x
    x = conv3d_pad(x, nf, k, d=d)
    x = conv3d_pad(x, nf, k, d=d, act=None)
    if res.shape[-1] != nf:  # 通道不一致时投影
        res = Conv3D(nf, 1, padding='same')(res)
    return Add()([res, x])

class G1(Model_Template):
    def __init__(self):
        self.architecture = "mobi"
        self.T = 5
        self.input_shape1 = [1, 192, 256, 3]  # 1 帧条件
        self.input_shape2 = [self.T, 192, 256, 28]  # T 帧姿态
        self.output_channels = 3
        self.conv_hidden_num = 128
        self.repeat_num = int(np.log2(96)) - 2  # 4 次下采样
        self.activation_fn = 'relu'
        self.data_format = 'channels_last'
        self.lr_initial_G1 = 2e-5
        super().__init__()

    # ---------- 模型 ----------
    def _build_model(self):
        # --------------- Encoder：条件/外观分支 ---------------
        inp_cond = Input(shape=self.input_shape1, name='cond')  # [B,1,H,W,3]
        # 先扩充时间维 → [B,1,H,W,C] → [B,1,H,W,C]  用 3D 卷积 (1,k,k)
        x_cd = Reshape([1, 192, 256, 3])(inp_cond)
        x_cd = conv3d_pad(x_cd, self.conv_hidden_num, k=(1, 3, 3))  # 时间核=1
        skips_cd = []
        for idx in range(self.repeat_num):
            c = self.conv_hidden_num * (idx + 1)
            # 3D 残差块（时间 stride=1）
            x_cd = res3d_block(x_cd, c, k=(1, 3, 3))
            skips_cd.append(x_cd)
            if idx < self.repeat_num - 1:
                # 空间下采样 2×，时间不变
                x_cd = conv3d_pad(x_cd, self.conv_hidden_num * (idx + 2),
                                  k=(1, 2, 2), s=(1, 2, 2))

        # 压到 64 维 latent
        B, _, Hf, Wf, Cf = x_cd.shape
        x_cd = Reshape([1, Hf * Wf * Cf])(x_cd)
        z_cd = Dense(64, activation=None)(x_cd)  # [B,1,64]

        # --------------- Encoder：姿态分支 ---------------
        inp_pose = Input(shape=self.input_shape2, name='pose')  # [B,T,H,W,28]
        x_ps = Reshape([self.T, 192, 256, 28])(inp_pose)
        x_ps = conv3d_pad(x_ps, self.conv_hidden_num, k=(1, 3, 3))
        skips_ps = []
        for idx in range(self.repeat_num):
            c = self.conv_hidden_num * (idx + 1)
            x_ps = res3d_block(x_ps, c, k=(1, 3, 3))
            skips_ps.append(x_ps)
            if idx < self.repeat_num - 1:
                x_ps = conv3d_pad(x_ps, self.conv_hidden_num * (idx + 2),
                                  k=(1, 2, 2), s=(1, 2, 2))

        x_ps = Reshape([self.T, Hf * Wf * Cf])(x_ps)
        z_ps = Dense(64, activation=None)(x_ps)  # [B,T,64]

        # --------------- Latent fusion ---------------
        z_cd_tile = Lambda(lambda z: tf.tile(z, [1, self.T, 1]))(z_cd)  # [B,T,64]
        z_fuse = Concatenate()([z_cd_tile, z_ps])  # [B,T,128]
        z_fuse = Dense(64, activation=self.activation_fn)(z_fuse)

        # 映射回 12×16×C
        decoder_init = Dense(Hf * Wf * self.conv_hidden_num, activation=None)(z_fuse)
        decoder_init = Reshape([self.T, Hf, Wf, self.conv_hidden_num])(decoder_init)

        # ======== 时序增强：真 3D 残差 ×2 （感受野 9 帧） ========
        x = decoder_init
        x = res3d_block(x, self.conv_hidden_num, k=(5, 3, 3))  # 时间核=5
        x = res3d_block(x, self.conv_hidden_num, k=(3, 3, 3))

        # --------------- Decoder ---------------
        # long skip：条件分支只有 1 帧 → tile
        def tile_time(x):
            return tf.tile(x[:, :1], [1, self.T, 1, 1, 1])

        long_skips = [
            Concatenate()([
                Lambda(tile_time)(sk_cd),
                sk_ps
            ]) for sk_cd, sk_ps in zip(skips_cd[::-1], skips_ps[::-1])
        ]

        for idx in range(self.repeat_num):
            skip = long_skips[idx]
            # 1. 降维：砍半通道
            skip = conv3d_pad(skip, skip.shape[-1] // 2, k=(1, 1, 1), act=None)

            # 2. concat
            x = Concatenate()([x, skip])

            # 3. 立刻升回所需通道（可逆）
            wanted = self.conv_hidden_num * (self.repeat_num - idx)  # 举例
            x = conv3d_pad(x, wanted, k=(1, 1, 1))  # 升回

            # 4. 继续残差
            x = res3d_block(x, wanted, k=(1, 3, 3))

            if idx < self.repeat_num - 1:
                x = UpSampling3D(size=(1, 2, 2))(x)
                # 下一级通道数
                next_c = self.conv_hidden_num * (self.repeat_num - 1 - idx)
                x = conv3d_pad(x, next_c, k=(1, 1, 1))

                # 输出层：1×1×1 卷积 → 3 通道
        out = Conv3D(self.output_channels, (1, 1, 1), padding='same', activation=None)(x)


        return Model(inputs=[inp_cond, inp_pose], outputs=out)


        # ---------- 优化器 ----------

    def _optimizer(self):
        optimizer = Adam(
            learning_rate=self.lr_initial_G1,
            beta_1=0.5,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        # print(f"[DEBUG] Optimizer created with lr: {self.lr_initial_G1}")
        return optimizer

    # ---------- 前向 ----------
    def prediction(self, Ic, Pt):
        # ----------  0. 基本类型保证 ----------
        Ic = tf.cast(Ic, dtype=tf.float32)
        Pt = tf.cast(Pt, dtype=tf.float32)

        # ----------  1. 先算差分+28 通道（原逻辑） ----------
        def compute_pose_with_delta(pose):
            delta = pose[:, 1:, ...] - pose[:, :-1, ...]
            batch_size = tf.shape(pose)[0]
            zero_frame = tf.zeros([batch_size, 1, 192, 256, 14], dtype=pose.dtype)
            delta_padded = tf.concat([zero_frame, delta], axis=1)
            return tf.concat([pose, delta_padded], axis=-1)  # [B, T, H, W, 28]

        Pt = compute_pose_with_delta(Pt)  # 现在 Pt 是 [B, T, H, W, 28]

        # ----------  2. 时间切片推理（新增） ----------

        T = tf.shape(Pt)[1]
        chunk = 5  # 可调，只要显存不爆
        outs = []
        for t in range(0, T, chunk):
            pt_seg = Pt[:, t:t + chunk, ...]
            out_seg = self.model([Ic, pt_seg])  # 输出同一段时间长度
            outs.append(out_seg)
        output = tf.concat(outs, axis=1)  # [B, T, H, W, C]


       # output = self.model([Ic, Pt])  ###用切片，这个就注释掉

        # ----------  3. NaN 修复（原逻辑） ----------
        if tf.reduce_any(tf.math.is_nan(output)):
            print("[WARNING G1] NaN detected in model output, replaced with zeros")
            output = tf.where(tf.math.is_nan(output), tf.zeros_like(output), output)

        return output

    def PoseMaskloss(self, I_PT1, It, Mt):
        # 统一数据类型
        I_PT1 = tf.cast(I_PT1, dtype=tf.float32)
        It = tf.cast(It, dtype=tf.float32)
        Mt = tf.cast(Mt, dtype=tf.float32)

        # 基础损失
        diff = I_PT1 - It
        l1_per_frame = tf.reduce_mean(tf.abs(diff), axis=[1, 2, 3])
        mask_per_frame = tf.reduce_mean(tf.abs(diff) * Mt, axis=[1, 2, 3])
        base_loss = tf.reduce_mean(l1_per_frame + mask_per_frame)

        # 内存友好的平滑损失计算
        smooth = tf.constant(0.0, dtype=tf.float32)
        if I_PT1.shape[0] > 1:
            # 方法1: 使用更小的窗口计算平滑损失
            frame_interval = 2  # 每隔一帧计算
            total_smooth = 0.0
            count = 0

            for i in range(0, I_PT1.shape[0] - frame_interval, frame_interval):
                delta_pred = I_PT1[i + frame_interval] - I_PT1[i]
                delta_gt = It[i + frame_interval] - It[i]
                smooth_diff = delta_pred - delta_gt
                total_smooth += tf.reduce_mean(tf.abs(smooth_diff))
                count += 1

            if count > 0:
                smooth = total_smooth / tf.cast(count, tf.float32)

        total_loss = base_loss + 0.1 * smooth
        return total_loss

    def ssim(self, I_PT1, It, mean_0, mean_1, unprocess_function):
        # 确保输入是float32
        I_PT1 = tf.cast(I_PT1, dtype=tf.float32)
        It = tf.cast(It, dtype=tf.float32)

        # 确保mean参数也是float32
        mean_0 = tf.cast(mean_0, dtype=tf.float32)
        mean_1 = tf.cast(mean_1, dtype=tf.float32)

        It = unprocess_function(It, mean_1)
        I_PT1 = unprocess_function(I_PT1, mean_0)

        mean = tf.reduce_mean(tf.image.ssim(I_PT1, It, max_val=255))
        return mean

    def mask_ssim(self, I_PT1, It, Mt, mean_0, mean_1, unprocess_function):
        # 1. 5-D mean → [B,T,1,1,3]
        mean_0 = tf.reshape(mean_0, [tf.shape(mean_0)[0], tf.shape(mean_0)[1], 1, 1, 3])
        mean_1 = tf.reshape(mean_1, [tf.shape(mean_1)[0], tf.shape(mean_1)[1], 1, 1, 3])

        # 2. 统一 float32
        It = tf.cast(It, tf.float32)
        I_PT1 = tf.cast(I_PT1, tf.float32)
        Mt = tf.cast(Mt, tf.float32)

        # 3. 反归一化（强制 float）
        It_processed = tf.cast(unprocess_function(It, mean_1), tf.float32)
        I_PT1_processed = tf.cast(unprocess_function(I_PT1, mean_0), tf.float32)

        # 4. 掩膜乘法（float 域）
        mask_raw = Mt * It_processed
        mask_out = Mt * I_PT1_processed

        # 5. SSIM（需要 float）
        return tf.reduce_mean(tf.image.ssim(mask_raw, mask_out, max_val=255.0))






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


T_FRAME = 20
LOW_BIN = 3            # 0, 1.5, 3 Hz
CORE_BIN = [1, 2]      # 2–4 Hz 核心带

@tf.function
def energy_align(img, pose):
    """三通道 2–4 Hz 能量谱 L2 对齐（无标签）"""
    img = tf.cast(img, tf.float32)
    pose = tf.cast(pose, tf.float32)
    # 分别对 R/G/B 做 RFFT
    spec_img  = tf.signal.rfft(img, [T_FRAME])[..., :LOW_BIN]          # [B,H,W,3,3]
    spec_pose = tf.signal.rfft(tf.reduce_mean(pose[:,:,:,:,:14], axis=-1, keepdims=True),
                               [T_FRAME])[..., :LOW_BIN]                # [B,H,W,1,3]
    img_amp  = K.abs(spec_img)      # [B,H,W,3,3]
    pose_amp = K.abs(spec_pose)     # [B,H,W,1,3]

    # 三通道能量和 → 单通道比例
    img_energy  = tf.reduce_sum(tf.square(img_amp),   [1,2,3,4], keepdims=True) + 1e-5  # [B,1,1,1,1]
    pose_energy = tf.reduce_sum(tf.square(pose_amp), [1,2,3,4], keepdims=True) + 1e-5  # [B,1,1,1,1]
    img_amp_norm  = img_amp  / tf.sqrt(img_energy)     # 三通道同时缩放
    pose_amp_norm = pose_amp / tf.sqrt(pose_energy)

    # 只盯 2–4 Hz（bin-1 & bin-2）
    core_img  = img_amp_norm[..., 1:3, :]   # [B,H,W,3,2]
    core_pose = pose_amp_norm[..., 1:3, :]  # [B,H,W,1,2]
    return K.mean(K.square(core_img - core_pose))



# =====================================================
#  G2 类（普通 3-D 全程）
# =====================================================
class G2(Model_Template):
    def __init__(self):
        self.architecture = "3d普通"
        self.T = 20
        self.input_shape1 = [1, 96, 128, 3]        # 5-D 条件
        self.input_shape2 = [self.T, 96, 128, 3]  # 5-D 姿态
        self.activation_fn = 'relu'
        self.data_format = 'channels_last'
        self.output_channels = 3
        self.conv_hidden_num = 64                  # 初始通道砍半
        self.lr_initial_G2 = 4e-5
        super().__init__()

    def _build_model(self):
        c = self.conv_hidden_num
        T = self.T
        act = self.activation_fn
        df = self.data_format  # 'channels_last' 即可

        # ---------- 条件分支 Encoder ----------
        inp1 = Input(self.input_shape1, name='cond')  # [B,1,H,W,C]
        x1 = Conv3D(c, (1,3,3), padding='same', activation=act, data_format=df)(inp1)
        x1 = Conv3D(c, (1,3,3), padding='same', activation=act, data_format=df)(x1)
        x1 = Conv3D(c, (1,3,3), padding='same', activation=act, data_format=df)(x1)
        #x1 = se_block_3d(x1, ratio=8, name='se_skip1')
        skip1_1 = x1  # 1/1

        x1 = Conv3D(c * 2, (1,2,2), strides=(1, 2, 2), padding='same', activation=act, data_format=df)(x1)
        x1 = Conv3D(c * 2, (1,3,3), padding='same', activation=act, data_format=df)(x1)
        x1 = Conv3D(c * 2, (1,3,3), padding='same', activation=act, data_format=df)(x1)
        #x1 = se_block_3d(x1, ratio=8, name='se_skip2')
        skip1_2 = x1  # 1/2

        x1 = Conv3D(c * 3, (1,2,2), strides=(1, 2, 2), padding='same', activation=act, data_format=df)(x1)
        x1 = Conv3D(c * 3, (1,3,3), padding='same', activation=act, data_format=df)(x1)
        x1 = Conv3D(c * 3, (1,3,3), padding='same', activation=act, data_format=df)(x1)
       # x1 = se_block_3d(x1, ratio=8, name='se_bridge')
        bridge1 = x1  # 1/4

        # ---------- 姿态分支 Encoder ----------
        inp2 = Input(self.input_shape2, name='pose')  # [B,T,H,W,C]
        x2 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(inp2)
        x2 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(x2)
        x2 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(x2)
        skip2_1 = x2  # 1/1

        x2 = Conv3D(c * 2, 2, strides=(1, 2, 2), padding='same', activation=act, data_format=df)(x2)
        x2 = Conv3D(c * 2, 3, padding='same', activation=act, data_format=df)(x2)
        x2 = Conv3D(c * 2, 3, padding='same', activation=act, data_format=df)(x2)
        skip2_2 = x2  # 1/2

        x2 = Conv3D(c * 3, 2, strides=(1, 2, 2), padding='same', activation=act, data_format=df)(x2)
        x2 = Conv3D(c * 3, 3, padding='same', activation=act, data_format=df)(x2)
        x2 = Conv3D(c * 3, 3, padding='same', activation=act, data_format=df)(x2)
        bridge2 = x2  # 1/4

        # ---------- 共享 Decoder ----------
        # 先把条件分支的 bridge 复制 T 份
        bridge1_tile = Lambda(lambda z: tf.tile(z, [1, T, 1, 1, 1]))(bridge1)  # [B,T,H/4,W/4,c*3]
        dec = Concatenate(axis=-1)([bridge1_tile, bridge2])  # [B,T,H/4,W/4,2*c*3]

        # 解码块 1 ：1/4 → 1/2
        # --------------------------------------------------
        dec = UpSampling3D(size=(1, 2, 2), data_format=df)(dec)  # [B,T,H/2,W/2,2*c*3]
        dec = Conv3D(c * 2, 1, padding='same', activation=act, data_format=df)(dec)  # 先压通道

        # 把条件分支的 skip 也 tile 成 [B,T,...]
        skip1_2_tile = Lambda(lambda z: tf.tile(z, [1, T, 1, 1, 1]))(skip1_2)  # [B,T,H/2,W/2,c*2]
        long_con_1 = Concatenate(axis=-1)([dec, skip1_2_tile, skip2_2])  # [B,T,H/2,W/2,c*2*3]

        # 并行支路 1
        b1 = Conv3D(c * 2, 3, padding='same', activation=act, data_format=df)(long_con_1)
        b1 = Conv3D(c * 2, 3, padding='same', activation=act, data_format=df)(b1)

        # 并行支路 2
        b2 = Conv3D(c * 2, 3, padding='same', activation=act, data_format=df)(long_con_1)
        b2 = Conv3D(c * 2, 3, padding='same', activation=act, data_format=df)(b2)

        # 合并 + 通道压缩
        dec = Concatenate(axis=-1)([b1, b2])  # [B,T,H/2,W/2,c*4]
        dec = Conv3D(c * 2, 1, padding='same', activation=act, data_format=df)(dec)  # → c*2

        # --------------------------------------------------
        # 解码块 2 ：1/2 → 1/1
        # --------------------------------------------------
        dec = UpSampling3D(size=(1, 2, 2), data_format=df)(dec)  # [B,T,H,W,c*2]
        dec = Conv3D(c, 1, padding='same', activation=act, data_format=df)(dec)  # 压到 c

        skip1_1_tile = Lambda(lambda z: tf.tile(z, [1, T, 1, 1, 1]))(skip1_1)  # [B,T,H,W,c]
        long_con_2 = Concatenate(axis=-1)([dec, skip1_1_tile, skip2_1])  # [B,T,H,W,c*3]

        # 并行支路 1
        b1 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(long_con_2)
        b1 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(b1)

        # 并行支路 2
        b2 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(long_con_2)
        b2 = Conv3D(c, 3, padding='same', activation=act, data_format=df)(b2)

        # 合并 + 通道压缩
        dec = Concatenate(axis=-1)([b1, b2])  # [B,T,H,W,c*2]
        dec = Conv3D(c, 1, padding='same', activation=act, data_format=df)(dec)  # → c

        # --------------------------------------------------
        out = Conv3D(3, 1, padding='same', activation='tanh', data_format=df)(dec)  # [B,T,H,W,3]


        return Model([inp1, inp2], out)

    # ---------- 对外接口 ----------
    def prediction(self, I_PT1, Ic):
        """
        I_PT1 : [B, T, H, W, C]  (T 可能 1000+)
        Ic    : [B, H, W, C]     (无时间维)
        """
        T = tf.shape(I_PT1)[1]
        chunk = 20  # 可调，只要显存不爆
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
        # 统一数据类型
        I_PT2 = tf.cast(I_PT2, dtype=tf.float32)
        It = tf.cast(It, dtype=tf.float32)
        Mt = tf.cast(Mt, dtype=tf.float32)

        # 基础损失
        diff = I_PT2 - It
        l1_per_frame = tf.reduce_mean(tf.abs(diff), axis=[1, 2, 3])
        mask_per_frame = tf.reduce_mean(tf.abs(diff) * Mt, axis=[1, 2, 3])
        base_loss = tf.reduce_mean(l1_per_frame + mask_per_frame)

        # 内存友好的平滑损失计算
        smooth = tf.constant(0.0, dtype=tf.float32)
        if I_PT2.shape[0] > 1:
            frame_interval = 2  # 每隔一帧计算
            total_smooth = 0.0
            count = 0

            for i in range(0, I_PT2.shape[0] - frame_interval, frame_interval):
                delta_pred = I_PT2[i + frame_interval] - I_PT2[i]
                delta_gt = It[i + frame_interval] - It[i]
                smooth_diff = delta_pred - delta_gt
                total_smooth += tf.reduce_mean(tf.abs(smooth_diff))
                count += 1

            if count > 0:
                smooth = total_smooth / tf.cast(count, tf.float32)

        total_loss = base_loss + 0.1 * smooth
        return total_loss

    # ---------- 1. 感知损失（SSIM） ----------
    def perceptual_loss(self, real, gen):
        real = tf.reshape(real, [-1, 96, 128, 3])
        gen = tf.reshape(gen, [-1, 96, 128, 3])
        return 1.0 - tf.reduce_mean(tf.image.ssim(real, gen, max_val=1.0))

    # ---------- 2. 帧间一致性 ----------
    def temporal_loss(self, pred):
        # pred: [B, T, H, W, 3]
        diff_gt = tf.stop_gradient(pred[:, 1:] - pred[:, :-1])
        diff = pred[:, 1:] - pred[:, :-1]
        l1 = tf.reduce_mean(tf.abs(diff - diff_gt))
        ssim = 1.0 - tf.reduce_mean(tf.image.ssim(
            tf.reshape(pred[:, :-1], [-1, 96, 128, 3]),
            tf.reshape(pred[:, 1:], [-1, 96, 128, 3]), 1.0))
        return l1 + 0.5 * ssim

    def adv_loss(self, D_neg, I_PT2, It, Mt, Pt):
        0
        # 1. 对抗 loss（mixed 下是 float16）
        adv = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_neg, labels=tf.ones_like(D_neg)))

        # 2. 其余损失（默认 float32）
        pm = self.PoseMaskloss(I_PT2, It, Mt)
        per = self.perceptual_loss(It, I_PT2)
        tmp = self.temporal_loss(I_PT2)
        spec = energy_align(I_PT2,Pt)

        # 3. 统一 float32
        adv = tf.cast(adv, tf.float32)
        pm = tf.cast(pm, tf.float32)
        per = tf.cast(per, tf.float32)
        tmp = tf.cast(tmp, tf.float32)
        #spec = tf.cast(spec, tf.float32)  0.0001 * spec

        return (2.0 * adv +
                3.5 * pm +
                2.0 * per +
                2.0 * tmp
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



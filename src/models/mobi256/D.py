# ==========================================
#  D.py  序列感知 3-D 判别器  【训练脚本零改动】
# ==========================================
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv3D, BatchNormalization, LeakyReLU, Dense, Lambda
)
from tensorflow.keras.optimizers import Adam
from models.Model_template import Model_Template


class D(Model_Template):
    def __init__(self):
        self.architecture = "3d"
        self.T = 5
        self.input_shape = [self.T, 192, 256, 3]
        self.activation_fn = 'lrelu'
        self.lr_initial_D = 2.5e-6
        super().__init__()

    # ---------- 网络：双头，但对外隐藏 ----------
    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        c, strides, channels = 32, \
            [(1,2,2),(1,2,2),(1,2,2),(1,2,1)], \
            [32,64,128,256]
        x = inputs
        for i, (st, ch) in enumerate(zip(strides, channels)):
            x = Conv3D(ch, 3, strides=st, padding='same',
                      use_bias=False, name=f'conv3d_{i+1}')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)

        # 1. 真/假头：3-D PatchGAN
        patch = Conv3D(1, 1, padding='same', name='patch')(x)
        # 2. 时序头：帧间差异
        diff = Lambda(lambda t: t[:, 1:] - t[:, :-1])(x)
        temp = Conv3D(1, 1, padding='same', name='temp')(diff)
        temp = Lambda(lambda t: tf.reduce_mean(t, [2, 3, 4]))(temp)  # [B,T-1]

        # 对外只暴露一个 logits，内部把两个信息拼在一起
        # patch: [B,T',H',W',1]  -> 全局平均
        patch_gap = Lambda(lambda t: tf.reduce_mean(t, [1, 2, 3, 4]))(patch)
        # temp:  [B,T-1]        -> 平均
        temp_gap = Lambda(lambda t: tf.reduce_mean(t, 1))(temp)
        # 加权合并（可调权重）
        logits = patch_gap + 0.01 * temp_gap
        return Model(inputs, logits, name='D_hidden_sequence')

    # ---------- 优化器 ----------
    def _optimizer(self):
        return Adam(learning_rate=self.lr_initial_D, beta_1=0.5, epsilon=1e-8)

    # ---------- 前向：接口完全不变 ----------
    def prediction(self, It, I_PT2, Ic, training=False):
        It = tf.cast(It, tf.float32)
        I_PT2 = tf.cast(I_PT2, tf.float32)
        Ic = tf.cast(Ic, tf.float32)

        # ---- 输入噪声（仅训练阶段） ----
        if training:
            noise_std = 0.01
            It += tf.random.normal(tf.shape(It), stddev=noise_std)
            I_PT2 += tf.random.normal(tf.shape(I_PT2), stddev=noise_std)
            Ic += tf.random.normal(tf.shape(Ic), stddev=noise_std)

        Ic = tf.repeat(Ic, repeats=self.T, axis=1)
        input_D = tf.concat([It, I_PT2, Ic], axis=0)
        logits = self.model(input_D, training=training)  # 也传给底层 BN/Drop（如果有）
        return tf.reshape(logits, [-1])

    # ---------- 损失：接口完全不变 ----------
    def adv_loss(self, D_pos, D_neg_ref, D_neg_raw0):
        real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_pos, labels=tf.ones_like(D_pos)))
        fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_neg_ref, labels=tf.zeros_like(D_neg_ref)))
        fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_neg_raw0, labels=tf.zeros_like(D_neg_raw0)))
        loss = 0.5 * real + 0.25 * (fake1 + fake2)

        # 时序信息已经通过 0.1*temp_gap 融进 logits，这里不再额外返回
        # 如需监控，可打开下方 tf.print（训练日志会多出两行）
        # tf.print("temp_real", tf.reduce_mean(tf.abs(temp_pos)),
        #          "temp_fake", tf.reduce_mean(tf.abs(temp_ref))+tf.reduce_mean(tf.abs(temp_raw0)))
        return [loss, fake1 + fake2, real]   # 与原接口一致
# -*- coding: utf-8 -*-
"""
ghost_custom_layers.py — Custom layers for Ghost-CAS-UNet v2
=============================================================
This file must be kept alongside ghost_cas_unet_v2_full.h5 on the RPi.
It is auto-imported by batch_benchmark.py and rpi_benchmark.py.
"""
import tensorflow as tf
from tensorflow.keras import layers

class GhostModule(layers.Layer):
    def __init__(self, filters, kernel_size=1, ratio=2, dw_kernel=3, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        self.primary_filters = max(1, filters // ratio)
        self.ghost_filters = filters - self.primary_filters

        self.primary_conv = layers.Conv2D(
            self.primary_filters, kernel_size, padding='same', use_bias=False
        )
        self.primary_gn = layers.GroupNormalization(
            groups=min(8, self.primary_filters), axis=-1
        )
        self.primary_act = layers.Activation(activation)

        self.ghost_dw = layers.DepthwiseConv2D(
            dw_kernel, padding='same', use_bias=False
        )
        self.ghost_gn = layers.GroupNormalization(
            groups=min(8, self.primary_filters), axis=-1
        )
        self.ghost_act = layers.Activation(activation)

    def call(self, x, training=None):
        primary = self.primary_conv(x)
        primary = self.primary_gn(primary, training=training)
        primary = self.primary_act(primary)

        ghost = self.ghost_dw(primary)
        ghost = self.ghost_gn(ghost, training=training)
        ghost = self.ghost_act(ghost)

        out = tf.concat([primary, ghost], axis=-1)
        return out[:, :, :, :self.filters]

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio,
        })
        return config


class CoordinateAttention(layers.Layer):
    def __init__(self, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        mid_channels = max(8, channels // self.reduction)

        self.shared_conv = layers.Conv2D(mid_channels, 1, use_bias=False)
        self.shared_gn = layers.GroupNormalization(groups=min(8, mid_channels), axis=-1)
        self.shared_act = layers.Activation('relu')

        self.conv_h = layers.Conv2D(channels, 1, use_bias=False)
        self.conv_w = layers.Conv2D(channels, 1, use_bias=False)

        super().build(input_shape)

    def call(self, x, training=None):
        x_shape = tf.shape(x)
        h = x_shape[1]
        w = x_shape[2]

        pool_h = tf.reduce_mean(x, axis=2, keepdims=True)
        pool_w = tf.reduce_mean(x, axis=1, keepdims=True)
        pool_w_t = tf.transpose(pool_w, perm=[0, 2, 1, 3])

        combined = tf.concat([pool_h, pool_w_t], axis=1)

        combined = self.shared_conv(combined)
        combined = self.shared_gn(combined, training=training)
        combined = self.shared_act(combined)

        split_h, split_w = tf.split(combined, [h, w], axis=1)

        attn_h = tf.sigmoid(self.conv_h(split_h))
        split_w_back = tf.transpose(split_w, perm=[0, 2, 1, 3])
        attn_w = tf.sigmoid(self.conv_w(split_w_back))

        out = x * attn_h * attn_w
        return out

    def get_config(self):
        config = super().get_config()
        config.update({'reduction': self.reduction})
        return config


class AttentionGate(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.wl = layers.Conv2D(self.filters, 1, strides=1, padding='same', use_bias=True)
        self.wg = layers.Conv2D(self.filters, 1, strides=1, padding='same', use_bias=True)
        self.psi = layers.Conv2D(1, 1, strides=1, padding='same', use_bias=True)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, g = inputs
        xl = self.wl(x)
        gg = self.wg(g)
        gg = tf.image.resize(gg, (tf.shape(x)[1], tf.shape(x)[2]))

        joined = tf.add(xl, gg)
        act = tf.nn.relu(joined)
        
        psi = self.psi(act)
        coef = tf.nn.sigmoid(psi)
        
        return tf.multiply(x, coef)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config


class GhostBottleneck(layers.Layer):
    def __init__(self, filters, ratio=2, use_attention=False, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.target_filters = filters
        self.use_attention = use_attention
        self.dilation_rate = dilation_rate
        
        self.ghost1 = GhostModule(filters, kernel_size=1, ratio=ratio)
        self.ghost2 = GhostModule(filters, kernel_size=3, ratio=ratio)
        
        if use_attention:
            self.attention = CoordinateAttention()
        self.residual_conv = None

    def build(self, input_shape):
        if input_shape[-1] != self.target_filters:
            self.residual_conv = layers.Conv2D(
                self.target_filters, 1, padding='same', use_bias=False
            )
        super().build(input_shape)

    def call(self, x, training=None):
        residual = x
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
            
        out = self.ghost1(x, training=training)
        out = self.ghost2(out, training=training)
        
        if self.use_attention:
            out = self.attention(out, training=training)
            
        return out + residual

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.target_filters,
            'ratio': self.ratio,
            'use_attention': self.use_attention,
            'dilation_rate': self.dilation_rate,
        })
        return config


class DW_ASPP(layers.Layer):
    def __init__(self, out_channels, rates=(2, 4, 6), **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.rates = rates

    def build(self, input_shape):
        ch = input_shape[-1]
        branch_ch = max(ch // 4, 16)

        self.b1_conv = layers.Conv2D(branch_ch, 1, padding='same', use_bias=False)
        self.b1_bn = layers.BatchNormalization()

        self.dw_branches = []
        for rate in self.rates:
            dw = layers.DepthwiseConv2D(
                3, padding='same', dilation_rate=rate, use_bias=False
            )
            pw = layers.Conv2D(branch_ch, 1, padding='same', use_bias=False)
            bn = layers.BatchNormalization()
            self.dw_branches.append((dw, pw, bn))

        self.gap_conv = layers.Conv2D(branch_ch, 1, use_bias=False)
        self.gap_bn = layers.BatchNormalization()

        self.proj_conv = layers.Conv2D(self.out_channels, 1, padding='same', use_bias=False)
        self.proj_bn = layers.BatchNormalization()

        super().build(input_shape)

    def call(self, x, training=None):
        branches = []

        b1 = tf.nn.relu(self.b1_bn(self.b1_conv(x), training=training))
        branches.append(b1)

        for dw, pw, bn in self.dw_branches:
            b = dw(x)
            b = pw(b)
            b = tf.nn.relu(bn(b, training=training))
            branches.append(b)

        gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        gap = tf.nn.relu(self.gap_bn(self.gap_conv(gap), training=training))
        gap = tf.image.resize(gap, (tf.shape(x)[1], tf.shape(x)[2]))
        branches.append(gap)

        out = tf.concat(branches, axis=-1)
        out = tf.nn.relu(self.proj_bn(self.proj_conv(out), training=training))
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'rates': self.rates,
        })
        return config

GHOST_CUSTOM_OBJECTS = {
    'GhostModule': GhostModule,
    'CoordinateAttention': CoordinateAttention,
    'AttentionGate': AttentionGate,
    'GhostBottleneck': GhostBottleneck,
    'DW_ASPP': DW_ASPP,
}

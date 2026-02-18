# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Segmentation Models
===================

Includes:
1. Standard U-Net (baseline)
2. Ghost-CA-UNet (Novel: Ghost Modules + Coordinate Attention)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from typing import Tuple, List


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class GhostModule(layers.Layer):
    """
    Ghost Module (Han et al., CVPR 2020).

    Generates feature maps using cheap linear operations instead of
    full convolutions. Reduces FLOPs by ~50% while maintaining accuracy.

    Instead of producing `filters` feature maps via Conv2D, we:
    1. Produce `filters // ratio` "primary" features via Conv2D.
    2. Generate the remaining via cheap depthwise convolution (the "ghosts").
    3. Concatenate both to get the full `filters` output.
    """

    def __init__(self, filters, kernel_size=1, ratio=2, dw_kernel=3, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        # Number of primary (intrinsic) features
        self.primary_filters = max(1, filters // ratio)
        # Ghost features = total - primary
        self.ghost_filters = filters - self.primary_filters

        self.primary_conv = layers.Conv2D(
            self.primary_filters, kernel_size, padding='same', use_bias=False
        )
        self.primary_bn = layers.BatchNormalization()
        self.primary_act = layers.Activation(activation)

        # Cheap operation: depthwise conv to generate ghosts
        self.ghost_dw = layers.DepthwiseConv2D(
            dw_kernel, padding='same', use_bias=False
        )
        self.ghost_bn = layers.BatchNormalization()
        self.ghost_act = layers.Activation(activation)

    def call(self, x, training=None):
        # Primary features
        primary = self.primary_conv(x)
        primary = self.primary_bn(primary, training=training)
        primary = self.primary_act(primary)

        # Ghost features (cheap linear transform of primary)
        ghost = self.ghost_dw(primary)
        ghost = self.ghost_bn(ghost, training=training)
        ghost = self.ghost_act(ghost)

        # Concatenate and slice to exact filter count
        out = tf.concat([primary, ghost], axis=-1)
        # Use explicit 4D indexing for graph-mode compatibility
        return out[:, :, :, :self.filters]

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio,
        })
        return config


class CoordinateAttention(layers.Layer):
    """
    Coordinate Attention (Hou et al., CVPR 2021).

    Unlike SE/CBAM which lose positional info, Coordinate Attention
    encodes precise spatial positions via separate H and W pooling.
    This is critical for detecting tiny lesions (microaneurysms)
    where "WHERE" matters as much as "WHAT".
    """

    def __init__(self, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        mid_channels = max(8, channels // self.reduction)

        self.shared_conv = layers.Conv2D(mid_channels, 1, use_bias=False)
        self.shared_bn = layers.BatchNormalization()
        self.shared_act = layers.Activation('relu')

        self.conv_h = layers.Conv2D(channels, 1, use_bias=False)
        self.conv_w = layers.Conv2D(channels, 1, use_bias=False)

        super().build(input_shape)

    def call(self, x, training=None):
        # Use tf.shape() for dynamic shapes (graph-mode compatible)
        x_shape = tf.shape(x)
        h = x_shape[1]
        w = x_shape[2]

        # Pool along Width -> (B, H, 1, C)
        pool_h = tf.reduce_mean(x, axis=2, keepdims=True)
        # Pool along Height -> (B, 1, W, C)
        pool_w = tf.reduce_mean(x, axis=1, keepdims=True)
        # Transpose pool_w to (B, W, 1, C) for concatenation
        pool_w_t = tf.transpose(pool_w, perm=[0, 2, 1, 3])

        # Concatenate along spatial dim -> (B, H+W, 1, C)
        combined = tf.concat([pool_h, pool_w_t], axis=1)

        # Shared transform
        combined = self.shared_conv(combined)
        combined = self.shared_bn(combined, training=training)
        combined = self.shared_act(combined)

        # Split back
        split_h, split_w = tf.split(combined, [h, w], axis=1)

        # Generate attention maps
        attn_h = tf.sigmoid(self.conv_h(split_h))  # (B, H, 1, C)
        split_w_back = tf.transpose(split_w, perm=[0, 2, 1, 3])  # (B, 1, W, C)
        attn_w = tf.sigmoid(self.conv_w(split_w_back))  # (B, 1, W, C)

        # Apply attention
        out = x * attn_h * attn_w
        return out

    def get_config(self):
        config = super().get_config()
        config.update({'reduction': self.reduction})
        return config


class AttentionGate(layers.Layer):
    """
    Attention Gate (Oktay et al., 2018).

    Filters features propagated through skip connections.
    Signal from lower layer (gating) controls which parts of the
    skip connection (x) are relevant.
    """

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        # input_shape is a list: [x_shape, g_shape]
        self.wl = layers.Conv2D(self.filters, 1, strides=1, padding='same', use_bias=True)
        self.wg = layers.Conv2D(self.filters, 1, strides=1, padding='same', use_bias=True)
        self.psi = layers.Conv2D(1, 1, strides=1, padding='same', use_bias=True)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, g = inputs  # x=skip_connection, g=gating_signal

        # Align channels
        xl = self.wl(x)
        gg = self.wg(g)

        # SAFEGUARD: Always resize g to match x using dynamic shapes.
        # Static shapes are None in graph mode, so we always resize.
        # tf.image.resize is a no-op if shapes already match at runtime.
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
    """
    Ghost Bottleneck Block for U-Net encoder/decoder.

    Structure: GhostModule -> BN -> GhostModule -> Residual Connection
    """

    def __init__(self, filters, ratio=2, use_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.target_filters = filters
        self.use_attention = use_attention
        
        # 1. First Ghost Module (Expansion or feature extraction?)
        # Standard implementation often expands then shrinks, or effectively acts as ResBlock.
        # Here we keep it simple: Two Ghost Modules preserving channel count (or matching target).
        self.ghost1 = GhostModule(filters, kernel_size=1, ratio=ratio)
        self.ghost2 = GhostModule(filters, kernel_size=3, ratio=ratio)
        
        if use_attention:
            self.attention = CoordinateAttention()
        self.residual_conv = None

    def build(self, input_shape):
        if input_shape[-1] != self.target_filters:
            # 1x1 Conv to match channels for residual
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
        })
        return config


# =============================================================================
# STANDARD U-NET (Baseline)
# =============================================================================

def create_unet_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 3,
    encoder_filters: List[int] = [16, 32, 64, 128],
    dropout_rate: float = 0.2
) -> Model:
    """
    Standard U-Net for Semantic Segmentation (Baseline).
    """
    inputs = Input(shape=input_shape)

    # Encoder
    skip_connections = []
    x = inputs

    for filters in encoder_filters:
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        skip_connections.append(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout_rate)(x)

    # Bottleneck
    x = layers.Conv2D(encoder_filters[-1] * 2, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Decoder
    for i, filters in enumerate(reversed(encoder_filters)):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip_connections[-(i + 1)]])
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)

    return Model(inputs, outputs, name="UNet")


# =============================================================================
# GHOST-CA-UNET (Novel Architecture)
# =============================================================================

def create_ghost_unet_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 3,
    encoder_filters: List[int] = [16, 32, 64, 128],
    dropout_rate: float = 0.15,
    ghost_ratio: int = 2,
    use_skip_attention: bool = True
) -> Model:
    """
    Ghost-CA-UNet / Ghost-CAS-UNet (With Skip Attention).

    Novelty:
    - Encoder/Decoder: Ghost Module based.
    - Bottleneck: Coordinate Attention.
    - Skips: Optional Attention Gates (Ghost-CAS-UNet).
    """
    inputs = Input(shape=input_shape)

    # ---- ENCODER ----
    skip_connections = []
    x = inputs

    for i, filters in enumerate(encoder_filters):
        x = GhostBottleneck(filters, ratio=ghost_ratio, use_attention=False)(x)
        skip_connections.append(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout_rate)(x)

    # ---- BOTTLENECK ----
    bottleneck_filters = encoder_filters[-1] * 2
    x = GhostModule(bottleneck_filters, kernel_size=3, ratio=ghost_ratio)(x)
    x = CoordinateAttention()(x)

    # ---- DECODER ----
    for i, filters in enumerate(reversed(encoder_filters)):
        # 1. Upsample
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        
        # 2. Skip Connection
        skip = skip_connections[-(i + 1)]
        
        # [NOVELTY] Attention Gate on Skip
        if use_skip_attention:
            skip = AttentionGate(filters)([skip, x])
            
        x = layers.Concatenate()([x, skip])
        
        # 3. Block
        x = GhostBottleneck(filters, ratio=ghost_ratio, use_attention=False)(x)

    # ---- OUTPUT ----
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)

    name = "Ghost_CAS_UNet" if use_skip_attention else "Ghost_CA_UNet"
    return Model(inputs, outputs, name=name)


# =============================================================================
# MOBILE U-NET (Competitor: Depthwise Separable Convs)
# =============================================================================

def create_mobile_unet_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 3,
    encoder_filters: List[int] = [16, 32, 64, 128],
    dropout_rate: float = 0.1
) -> Model:
    """
    Mobile-UNet (Standard Efficient Baseline).
    Uses SeparableConv2D instead of Conv2D or GhostModule.
    """
    inputs = Input(shape=input_shape)
    skip_connections = []
    x = inputs

    # Encoder
    for filters in encoder_filters:
        # SepConv Block 1
        x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # SepConv Block 2
        x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        skip_connections.append(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout_rate)(x)

    # Bottleneck
    x = layers.SeparableConv2D(encoder_filters[-1] * 2, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Decoder
    for i, filters in enumerate(reversed(encoder_filters)):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip_connections[-(i + 1)]])
        
        x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)
    return Model(inputs, outputs, name="Mobile_UNet")


# =============================================================================
# MODEL REGISTRY
# =============================================================================

SEGMENTATION_MODELS = {
    'unet': create_unet_model,
    'mobile_unet': create_mobile_unet_model,
    'ghost_ca_unet': create_ghost_unet_model,
}
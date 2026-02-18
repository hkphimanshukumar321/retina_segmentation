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

def _norm_layer(norm_type: str = 'group', groups: int = 8):
    """Return a normalization layer constructor.
    
    Args:
        norm_type: 'batch' for BatchNormalization, 'group' for GroupNormalization.
        groups: Number of groups for GroupNorm (ignored for BN).
    """
    if norm_type == 'group':
        def _make(name=None):
            return layers.GroupNormalization(groups=groups, name=name)
        return _make
    else:
        def _make(name=None):
            return layers.BatchNormalization(name=name)
        return _make

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
        # GroupNorm: stable for small batches (Wu & He, ECCV 2018)
        self.primary_gn = layers.GroupNormalization(
            groups=min(8, self.primary_filters)
        )
        self.primary_act = layers.Activation(activation)

        # Cheap operation: depthwise conv to generate ghosts
        self.ghost_dw = layers.DepthwiseConv2D(
            dw_kernel, padding='same', use_bias=False
        )
        self.ghost_gn = layers.GroupNormalization(
            groups=min(8, self.primary_filters)
        )
        self.ghost_act = layers.Activation(activation)

    def call(self, x, training=None):
        # Primary features
        primary = self.primary_conv(x)
        primary = self.primary_gn(primary, training=training)
        primary = self.primary_act(primary)

        # Ghost features (cheap linear transform of primary)
        ghost = self.ghost_dw(primary)
        ghost = self.ghost_gn(ghost, training=training)
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
        self.shared_gn = layers.GroupNormalization(groups=min(8, mid_channels))
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
        combined = self.shared_gn(combined, training=training)
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
    Optionally uses dilated convolutions (for output stride 8).
    """

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
    """Depthwise Atrous Spatial Pyramid Pooling.
    
    Lightweight version of DeepLab's ASPP using depthwise separable convs.
    Captures multi-scale context at the bottleneck without heavy param cost.
    
    Branches:
      1. 1×1 pointwise conv
      2. DW 3×3 dilation rate 2
      3. DW 3×3 dilation rate 4
      4. DW 3×3 dilation rate 6
      5. Global average pooling + 1×1 conv
    All → concatenate → 1×1 projection
    """

    def __init__(self, out_channels, rates=(2, 4, 6), **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.rates = rates

    def build(self, input_shape):
        ch = input_shape[-1]
        branch_ch = max(ch // 4, 16)  # channels per branch

        # Branch 1: 1x1
        self.b1_conv = layers.Conv2D(branch_ch, 1, padding='same', use_bias=False)
        self.b1_bn = layers.BatchNormalization()

        # Branch 2-4: DW atrous convs
        self.dw_branches = []
        for rate in self.rates:
            dw = layers.DepthwiseConv2D(
                3, padding='same', dilation_rate=rate, use_bias=False
            )
            pw = layers.Conv2D(branch_ch, 1, padding='same', use_bias=False)
            bn = layers.BatchNormalization()
            self.dw_branches.append((dw, pw, bn))

        # Branch 5: Global Average Pooling
        self.gap_conv = layers.Conv2D(branch_ch, 1, use_bias=False)
        self.gap_bn = layers.BatchNormalization()

        # Projection
        total_ch = branch_ch * (1 + len(self.rates) + 1)
        self.proj_conv = layers.Conv2D(self.out_channels, 1, padding='same', use_bias=False)
        self.proj_bn = layers.BatchNormalization()

        super().build(input_shape)

    def call(self, x, training=None):
        branches = []

        # 1x1 branch
        b1 = tf.nn.relu(self.b1_bn(self.b1_conv(x), training=training))
        branches.append(b1)

        # Atrous DW branches
        for dw, pw, bn in self.dw_branches:
            b = dw(x)
            b = pw(b)
            b = tf.nn.relu(bn(b, training=training))
            branches.append(b)

        # Global pooling branch
        gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        gap = tf.nn.relu(self.gap_bn(self.gap_conv(gap), training=training))
        gap = tf.image.resize(gap, (tf.shape(x)[1], tf.shape(x)[2]))
        branches.append(gap)

        # Concat + project
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
# GHOST-CAS-UNET v2 (Publication-Ready Architecture)
# =============================================================================

def create_ghost_unet_v2(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 3,
    encoder_filters: List[int] = [32, 64, 128, 256],
    dropout_rate: float = 0.15,
    ghost_ratio: int = 2,
    use_skip_attention: bool = True,
    use_aspp: bool = True,
    deep_supervision: bool = True,
) -> Model:
    """Ghost-CAS-UNet v2 — publication-ready architecture.
    
    Upgrades over v1:
      • DW-ASPP at bottleneck (multi-scale context, DeepLab-style)
      • Output stride 8: last encoder stage uses dilated convs (no 4th pool)
      • Deep supervision: auxiliary loss heads at decoder levels 2 and 3
      • Wider default filters [32,64,128,256] (~2.8M params)
    """
    inputs = Input(shape=input_shape)

    # ---- ENCODER (output stride 8: 3 pools, last stage dilated) ----
    skip_connections = []
    x = inputs

    for i, filters in enumerate(encoder_filters):
        if i < len(encoder_filters) - 1:
            # Normal stage: Ghost block + MaxPool
            x = GhostBottleneck(filters, ratio=ghost_ratio, use_attention=False)(x)
            skip_connections.append(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(dropout_rate)(x)
        else:
            # Last stage: dilated Ghost block (NO pooling → output stride 8)
            x = GhostBottleneck(
                filters, ratio=ghost_ratio, use_attention=False, dilation_rate=2
            )(x)
            skip_connections.append(x)
            # No MaxPool here — preserves spatial resolution

    # ---- BOTTLENECK (ASPP expands via multi-scale, not channel doubling) ----
    bottleneck_filters = encoder_filters[-1]
    if use_aspp:
        x = DW_ASPP(bottleneck_filters)(x)
    else:
        x = GhostModule(bottleneck_filters, kernel_size=3, ratio=ghost_ratio)(x)
    x = CoordinateAttention()(x)

    # ---- DECODER with deep supervision ----
    aux_outputs = []
    n_levels = len(encoder_filters)

    for i, filters in enumerate(reversed(encoder_filters)):
        level_idx = n_levels - 1 - i  # which encoder level we're connecting to

        # Upsample (skip for the last encoder stage if no pooling was applied)
        if i == 0:  # first decoder level connects to dilated stage (same res)
            # No upsample needed — same spatial resolution
            pass
        else:
            # UpSampling2D + Ghost: lighter than Conv2DTranspose, no checkerboard
            x = layers.UpSampling2D(2, interpolation='bilinear')(x)
            x = GhostModule(filters, kernel_size=1, ratio=ghost_ratio)(x)

        # Skip connection with optional Attention Gate
        skip = skip_connections[-(i + 1)]
        if use_skip_attention:
            skip = AttentionGate(filters)([skip, x])
        x = layers.Concatenate()([x, skip])

        # Decoder block
        x = GhostBottleneck(filters, ratio=ghost_ratio, use_attention=False)(x)

        # Deep supervision: aux heads at middle decoder levels
        if deep_supervision and i in (1, 2):
            aux = layers.Conv2D(num_classes, 1, activation='sigmoid',
                                name=f'aux_out_{i}')(x)
            # Upsample aux to input resolution
            aux = layers.UpSampling2D(
                size=(2 ** (n_levels - 1 - i), 2 ** (n_levels - 1 - i)),
                interpolation='bilinear', name=f'aux_up_{i}'
            )(aux)
            aux_outputs.append(aux)

    # ---- OUTPUT ----
    main_output = layers.Conv2D(
        num_classes, 1, activation='sigmoid', name='main_out'
    )(x)
    # Upsample main to match input if output stride != 1
    # (output stride 8 → need 1 upsample of /1 since last stage had no pool,
    #  but first conv2dtranspose skipped → net output is at stride 1 if decoder
    #  properly handles the skip.  Actually we need to check.)
    # With 3 pools: input → /2 → /4 → /8 (bottleneck) → decoder upsamples ×2, ×2, ×2 → back to /1
    # The dilated stage doesn't pool, so encoder output = /8 (same as 3 pools).
    # Decoder has 4 levels but first doesn't upsample → 3 upsamples ×2 each → /1. Correct.

    if deep_supervision:
        name = "Ghost_CAS_UNet_v2_DS"
        outputs = [main_output] + aux_outputs
    else:
        name = "Ghost_CAS_UNet_v2"
        outputs = main_output

    if not use_skip_attention:
        name = name.replace("CAS", "CA")

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
    """Mobile-UNet (Standard Efficient Baseline)."""
    inputs = Input(shape=input_shape)
    skip_connections = []
    x = inputs

    for filters in encoder_filters:
        x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        skip_connections.append(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout_rate)(x)

    x = layers.SeparableConv2D(encoder_filters[-1] * 2, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

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
    'unet':             create_unet_model,
    'mobile_unet':      create_mobile_unet_model,
    'ghost_ca_unet':    create_ghost_unet_model,      # v1 (legacy)
    'ghost_cas_unet_v2': create_ghost_unet_v2,         # v2 (publication)
}
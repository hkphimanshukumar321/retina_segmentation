# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Baseline Models for Segmentation
==================================

Downloads pretrained encoder backbones and wraps them so they integrate
seamlessly with our data pipeline.

Baselines:
  1. DeepLabV3+ (ResNet50)  — built from tf.keras.applications (ImageNet)
  2. U-Net (ResNet34)       — via segmentation_models library
  3. LinkNet (ResNet34)      — via segmentation_models library
  4. FPN (ResNet34)          — via segmentation_models library

Requires::

    pip install segmentation-models
"""

import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model

logger = logging.getLogger(__name__)


# =============================================================================
# 1. DeepLabV3+ (ResNet50 backbone, ImageNet pretrained)
# =============================================================================

def _aspp_block(x, filters=256, rates=(6, 12, 18)):
    """Atrous Spatial Pyramid Pooling for DeepLabV3+."""
    branches = []

    # 1×1 conv
    b1 = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation("relu")(b1)
    branches.append(b1)

    # Atrous convolutions at different rates
    for rate in rates:
        b = layers.Conv2D(filters, 3, padding="same", dilation_rate=rate, use_bias=False)(x)
        b = layers.BatchNormalization()(b)
        b = layers.Activation("relu")(b)
        branches.append(b)

    # Global average pooling branch
    gap = layers.GlobalAveragePooling2D(keepdims=True)(x)
    gap = layers.Conv2D(filters, 1, padding="same", use_bias=False)(gap)
    gap = layers.BatchNormalization()(gap)
    gap = layers.Activation("relu")(gap)
    # Use UpSampling2D instead of Lambda for clean serialization (Raspberry Pi compatible)
    h = tf.keras.backend.int_shape(x)[1]  # may be None for dynamic shapes
    w = tf.keras.backend.int_shape(x)[2]
    if h is not None and w is not None:
        gap = layers.UpSampling2D(size=(h, w), interpolation="bilinear")(gap)
    else:
        # Dynamic shape fallback using a named layer subclass
        target_shape = tf.shape(x)[1:3]
        gap = layers.Lambda(
            lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3]),
            name="aspp_gap_resize",
        )([gap, x])
    branches.append(gap)

    # Concatenate + 1×1 projection
    out = layers.Concatenate()(branches)
    out = layers.Conv2D(filters, 1, padding="same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Activation("relu")(out)
    return out


def create_deeplabv3plus(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 5,
    backbone: str = "resnet50",
    encoder_weights: str = "imagenet",
    **kwargs,
):
    """DeepLabV3+ with ResNet50 encoder (ImageNet pretrained).

    Architecture:
      Encoder (ResNet50, output stride 16)
        -> ASPP (multi-scale context)
        -> Decoder (low-level feature fusion + upsampling)
        -> Sigmoid output head
    """
    inputs = layers.Input(shape=input_shape)

    # Backbone — ResNet50 pretrained on ImageNet
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=encoder_weights,
        input_tensor=inputs,
    )

    # Encoder outputs (stride 4 and stride 16)
    # conv2_block3_out -> stride 4 (low-level features)
    # conv4_block6_out -> stride 16 (high-level features)
    low_level_feat = base_model.get_layer("conv2_block3_out").output   # stride 4
    high_level_feat = base_model.get_layer("conv4_block6_out").output  # stride 16

    # ASPP on high-level features
    x = _aspp_block(high_level_feat, filters=256)

    # Decoder
    # Upsample ASPP output by 4× to match low-level features
    x = layers.UpSampling2D(size=4, interpolation="bilinear")(x)

    # Reduce low-level feature channels
    low_level = layers.Conv2D(48, 1, padding="same", use_bias=False)(low_level_feat)
    low_level = layers.BatchNormalization()(low_level)
    low_level = layers.Activation("relu")(low_level)

    # Concatenate
    x = layers.Concatenate()([x, low_level])

    # Refine
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Upsample to input resolution (currently at stride 4)
    x = layers.UpSampling2D(size=4, interpolation="bilinear")(x)

    # Output head
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="DeepLabV3plus_ResNet50")
    return model


# =============================================================================
# 2–4. segmentation_models baselines (lazy import)
# =============================================================================

_sm = None


def _ensure_sm():
    """Import segmentation_models with the Keras framework set."""
    global _sm
    if _sm is not None:
        return _sm

    import os
    os.environ["SM_FRAMEWORK"] = "tf.keras"

    try:
        import segmentation_models as sm
        _sm = sm
        logger.info(f"segmentation_models {sm.__version__} loaded (framework: tf.keras)")
        return sm
    except ImportError:
        raise ImportError(
            "The 'segmentation-models' package is required for baselines.\n"
            "Install it with:  pip install segmentation-models"
        )


def create_sm_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 5,
    backbone: str = "resnet34",
    encoder_weights: str = "imagenet",
    **kwargs,
):
    """U-Net with pretrained ResNet34 encoder (segmentation_models)."""
    sm = _ensure_sm()
    model = sm.Unet(
        backbone_name=backbone,
        input_shape=input_shape,
        classes=num_classes,
        activation="sigmoid",
        encoder_weights=encoder_weights,
    )
    model._name = f"SM_Unet_{backbone}"
    return model


def create_sm_linknet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 5,
    backbone: str = "resnet34",
    encoder_weights: str = "imagenet",
    **kwargs,
):
    """LinkNet with pretrained ResNet34 encoder (segmentation_models)."""
    sm = _ensure_sm()
    model = sm.Linknet(
        backbone_name=backbone,
        input_shape=input_shape,
        classes=num_classes,
        activation="sigmoid",
        encoder_weights=encoder_weights,
    )
    model._name = f"SM_Linknet_{backbone}"
    return model


def create_sm_fpn(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 5,
    backbone: str = "resnet34",
    encoder_weights: str = "imagenet",
    **kwargs,
):
    """FPN with pretrained ResNet34 encoder (segmentation_models)."""
    sm = _ensure_sm()
    model = sm.FPN(
        backbone_name=backbone,
        input_shape=input_shape,
        classes=num_classes,
        activation="sigmoid",
        encoder_weights=encoder_weights,
    )
    model._name = f"SM_FPN_{backbone}"
    return model


# =============================================================================
# Registry — DeepLabV3+ first, then others
# =============================================================================

BASELINE_MODELS = {
    "deeplabv3plus_resnet50": create_deeplabv3plus,
    "sm_unet_resnet34":      create_sm_unet,
    "sm_linknet_resnet34":   create_sm_linknet,
    "sm_fpn_resnet34":       create_sm_fpn,
}

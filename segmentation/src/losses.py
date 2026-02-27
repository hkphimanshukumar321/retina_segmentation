# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Custom Loss Functions — v2
===========================

Includes:
  • Focal Tversky Loss (Abraham et al., 2018) — targets sparse small objects
  • Lovász-Softmax Loss (Berman et al., CVPR 2018) — directly optimizes IoU
  • Combined v2 Loss — Lovász + Focal Tversky + BCE (publication-ready)
"""

import tensorflow as tf
from tensorflow.keras import backend as K


# =============================================================================
# FOCAL TVERSKY LOSS  (unchanged from v1)
# =============================================================================

def focal_tversky_loss(alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
    """
    Focal Tversky Loss (Abraham et al., 2018).
    
    Tversky Index (TI) = TP / (TP + alpha*FP + beta*FN)
    Loss = (1 - TI)^gamma
    
    alpha < beta → penalizes False Negatives more (good for tiny lesions like MA).
    """
    def loss(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        tp = K.sum(y_true * y_pred)
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        
        tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return K.pow((1 - tversky_index), gamma)
        
    return loss


# =============================================================================
# LOVÁSZ-SOFTMAX LOSS  (differentiable IoU surrogate)
# =============================================================================

def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t. sorted errors."""
    p = tf.cast(tf.shape(gt_sorted)[0], dtype=tf.float32)
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1.0 - gt_sorted)
    jaccard = 1.0 - intersection / union
    jaccard = tf.concat([jaccard[:1], jaccard[1:] - jaccard[:-1]], axis=0)
    return jaccard


def _lovasz_softmax_flat(probas, labels):
    """Multi-label Lovász loss on flattened tensors.
    
    Args:
        probas: (P,) float tensor of predicted probabilities for one class
        labels: (P,) float tensor of ground truth (0 or 1)
    """
    # Sort by descending prediction error
    errors = tf.abs(labels - probas)
    ordering = tf.argsort(errors, direction='DESCENDING')
    errors_sorted = tf.gather(errors, ordering)
    labels_sorted = tf.gather(labels, ordering)
    
    grad = _lovasz_grad(labels_sorted)
    loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1)
    return loss


def lovasz_softmax_loss():
    """Lovász-Softmax loss for multi-label sigmoid segmentation.
    
    Directly optimizes the Lovász extension of the Jaccard index (IoU).
    Berman et al., "The Lovász-Softmax Loss", CVPR 2018.
    
    Returns:
        Loss function compatible with Keras model.compile().
    """
    def loss(y_true, y_pred):
        # y_true, y_pred: (B, H, W, C) — multi-label sigmoid outputs
        # Use static shape for channel count (available at graph-build time)
        C = y_pred.shape[-1] or 3  # fallback to 3 if None
        total_loss = 0.0
        
        for c in range(C):
            y_true_c = tf.reshape(y_true[..., c], [-1])
            y_pred_c = tf.reshape(y_pred[..., c], [-1])
            y_true_c = tf.cast(y_true_c, tf.float32)
            total_loss += _lovasz_softmax_flat(y_pred_c, y_true_c)
        
        return total_loss / tf.cast(C, tf.float32)
    
    return loss


# =============================================================================
# COMBINED LOSSES
# =============================================================================

def combined_loss(alpha=0.3, beta=0.7, gamma=0.75):
    """
    Combined Loss v1: Binary Cross Entropy + Focal Tversky.
    Retained for backward compatibility.
    """
    ft_loss = focal_tversky_loss(alpha, beta, gamma)

    def loss(y_true, y_pred):
        bce_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        return bce_loss + ft_loss(y_true, y_pred)

    return loss


# =============================================================================
# CLASS-WEIGHTED FOCAL TVERSKY  (IDRiD literature — sparse class up-weighting)
# =============================================================================

def class_weighted_focal_tversky_loss(
    class_weights=None,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    smooth: float = 1e-6,
):
    """Per-class weighted Focal Tversky Loss.

    Each class contributes its Tversky loss multiplied by a class weight.
    Sparse/hard classes (MA, SE) get higher weights; easy classes (OD) get lower.

    Default weights:
        [3.0, 1.5, 1.0, 2.5, 0.5]  = [MA, HE, EX, SE, OD]

    Up-weights MA×3 and SE×2.5 — from IDRiD challenge top-team strategy.
    Normalised so total weight = num_classes (no overall scale shift).

    References:
        Abraham & Khan, "A Novel Focal Tversky Loss", ISBI 2019.
        Top IDRiD challenge submissions (Porwal et al. MedIA 2020).
    """
    DEFAULT_WEIGHTS = [3.0, 1.5, 1.0, 2.5, 0.5]  # MA, HE, EX, SE, OD

    def loss(y_true, y_pred):
        n_classes = tf.shape(y_pred)[-1]
        weights = class_weights if class_weights is not None else DEFAULT_WEIGHTS

        total = 0.0
        total_w = sum(weights[:len(weights)])

        for c in range(len(weights)):
            w  = weights[c]
            yt = tf.reshape(y_true[..., c], [-1])
            yp = tf.reshape(y_pred[..., c], [-1])

            tp = K.sum(yt * yp)
            fp = K.sum((1.0 - yt) * yp)
            fn = K.sum(yt * (1.0 - yp))

            ti = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
            total += w * K.pow(1.0 - ti, gamma)

        # Normalise by total weight so magnitude is comparable to base FT loss
        return total / (total_w + smooth)

    return loss


def combined_loss_v2(
    w_lovasz: float = 0.5,
    w_focal_tversky: float = 0.3,
    w_bce: float = 0.2,
    ft_alpha: float = 0.3,
    ft_beta: float = 0.7,
    ft_gamma: float = 0.75,
    class_weights=None,          # NEW: pass list to enable per-class weighting
):
    """Combined Loss v2: Lovász-Softmax + (Class-Weighted) Focal Tversky + BCE.

    When class_weights is provided (default: [3.0,1.5,1.0,2.5,0.5] for
    IDRID 5-class), uses class_weighted_focal_tversky_loss instead of the
    uniform version. This specifically helps MA and SE which are starved.

    • Lovász directly optimizes IoU (biggest impact)
    • Class-Weighted Focal Tversky up-weights MA/SE false negatives
    • BCE provides stable gradients in early epochs
    """
    lov = lovasz_softmax_loss()
    if class_weights is not None:
        ft = class_weighted_focal_tversky_loss(class_weights, ft_alpha, ft_beta, ft_gamma)
    else:
        ft = focal_tversky_loss(ft_alpha, ft_beta, ft_gamma)

    def loss(y_true, y_pred):
        bce_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        return (
            w_lovasz         * lov(y_true, y_pred)
            + w_focal_tversky * ft(y_true, y_pred)
            + w_bce           * bce_loss
        )

    return loss

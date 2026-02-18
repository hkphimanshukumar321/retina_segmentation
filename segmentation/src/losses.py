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
    bce = tf.keras.losses.BinaryCrossentropy()
    
    def loss(y_true, y_pred):
        return bce(y_true, y_pred) + ft_loss(y_true, y_pred)
        
    return loss


def combined_loss_v2(
    w_lovasz=0.5,
    w_focal_tversky=0.3,
    w_bce=0.2,
    ft_alpha=0.3,
    ft_beta=0.7,
    ft_gamma=0.75,
):
    """Combined Loss v2: Lovász-Softmax + Focal Tversky + BCE.
    
    • Lovász directly optimizes IoU (biggest impact)
    • Focal Tversky targets sparse FN (tiny lesions like MA)
    • BCE provides smooth gradients for early training stability
    
    Args:
        w_lovasz: Weight for Lovász-Softmax component
        w_focal_tversky: Weight for Focal Tversky component
        w_bce: Weight for BCE component
    """
    lov = lovasz_softmax_loss()
    ft = focal_tversky_loss(ft_alpha, ft_beta, ft_gamma)
    bce = tf.keras.losses.BinaryCrossentropy()
    
    def loss(y_true, y_pred):
        return (
            w_lovasz * lov(y_true, y_pred)
            + w_focal_tversky * ft(y_true, y_pred)
            + w_bce * bce(y_true, y_pred)
        )
    
    return loss

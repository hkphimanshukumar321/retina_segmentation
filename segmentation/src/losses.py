# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Custom Loss Functions
=====================

Specialized losses for handling extreme class imbalance in medical segmentation.
Includes Focal Tversky Loss for Microaneurysm (small object) detection.
"""

import tensorflow as tf
from tensorflow.keras import backend as K

def focal_tversky_loss(alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
    """
    Focal Tversky Loss (Abraham et al., 2018).
    
    Tversky Index (TI) = TP / (TP + alpha*FP + beta*FN)
    Loss = (1 - TI)^gamma
    
    Args:
        alpha: Weight for False Positives (0.3 -> penalize FN more).
        beta: Weight for False Negatives (0.7 -> penalize FN more).
        gamma: Focusing parameter (0.75 -> focus on hard examples).
        smooth: Smoothing factor to avoid division by zero.
        
    Best for: Tiny lesions like Microaneurysms (MAs) where FN is costly.
    """
    def loss(y_true, y_pred):
        # Flatten
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        # True Positives, False Positives, False Negatives
        tp = K.sum(y_true * y_pred)
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        
        # Tversky Index
        tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        
        # Focal Tversky Loss
        return K.pow((1 - tversky_index), gamma)
        
    return loss

def combined_loss(alpha=0.3, beta=0.7, gamma=0.75):
    """
    Combined Loss: Binary Cross Entropy + Focal Tversky.
    Stabilizes training while focusing on hard examples.
    """
    ft_loss = focal_tversky_loss(alpha, beta, gamma)
    bce = tf.keras.losses.BinaryCrossentropy()
    
    def loss(y_true, y_pred):
        return bce(y_true, y_pred) + ft_loss(y_true, y_pred)
        
    return loss

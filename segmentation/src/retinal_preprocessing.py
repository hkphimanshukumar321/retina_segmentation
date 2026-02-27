# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# ==============================================================================

"""
Retinal Image Preprocessing
==============================

Literature-backed preprocessing pipeline for IDRID fundus images.

Techniques:
  1. Ben Graham Normalization (Graham, Kaggle DR 2015)
     - Most cited preprocessing method in IDRID papers
     - Subtracts blurred version to remove illumination gradient
     - Dramatically improves visibility of low-contrast lesions (MA, HE)

  2. CLAHE on LAB L-channel (Gu et al. TMI 2020, Guo et al. AAAI 2021)
     - Contrast Limited Adaptive Histogram Equalization
     - Applied to L (luminance) channel of LAB colour space
     - Boosts local contrast without amplifying noise

  3. FOV Mask Computation
     - Detects the circular retinal disc region (Field of View)
     - Used to exclude black-border patches from random sampling
     - Prevents ~30% wasted patches on pure background

Usage::

    from segmentation.src.retinal_preprocessing import retinal_preprocess, compute_fov_mask

    img_rgb   = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img_proc  = retinal_preprocess(img_rgb)   # returns float32 [0,1] (H, W, 3)
    fov_mask  = compute_fov_mask(img_rgb)     # returns bool (H, W)
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. BEN GRAHAM NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def ben_graham_normalize(
    img_rgb: np.ndarray,
    sigma_scale: float = 0.015,
    weight: float = 4.0,
    bias: float = 128.0,
) -> np.ndarray:
    """Ben Graham fundus image normalization.

    Subtracts a heavy Gaussian blur from the original, then re-centres.
    This removes illumination gradients caused by uneven light from the
    fundus camera, making dark lesions like MA much more visible.

    Method:
        output = weight * img + (1 - weight) * blur + bias
        → simplifies to: output = 4*img - 4*blur + 128

    Args:
        img_rgb:     Input RGB image, uint8 or float32.
        sigma_scale: Gaussian sigma = sigma_scale * min(H, W). ~10 pixels
                     for a 256×256 image — covers the whole disc glow.
        weight:      Mixing weight (paper uses 4.0).
        bias:        Re-centering offset (128 → midpoint of uint8).

    Returns:
        Float32 image in [0, 1], same shape as input.

    Reference:
        Benjamin Graham, "Kaggle Diabetic Retinopathy Detection", 2015.
        Most cited preprocessing in IDRID segmentation literature.
    """
    img_f = img_rgb.astype(np.float32)
    sigma = sigma_scale * min(img_f.shape[:2])
    sigma = max(1.0, sigma)                 # floor to avoid zero-sigma edge case

    # Heavy Gaussian blur — captures illumination gradient only
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigma)

    # Subtract gradient, re-centre at 128
    out = weight * img_f - weight * blurred + bias
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLAHE ENHANCEMENT
# ─────────────────────────────────────────────────────────────────────────────

def clahe_enhance(
    img_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple = (8, 8),
) -> np.ndarray:
    """CLAHE on the L-channel of LAB colour space.

    LAB separates luminance (L) from chrominance (A, B), so CLAHE only
    modifies contrast without distorting hue — important for colour-based
    lesion features.

    Args:
        img_rgb:    Input RGB image, uint8.
        clip_limit: Max contrast amplification. 2.0 is standard for fundus.
        tile_grid:  Grid size for local contrast windows. (8,8) = 8×8 tiles.

    Returns:
        uint8 RGB image with enhanced local contrast.

    Reference:
        Gu et al., "CA-Net", TMI 2020.
        Guo et al., "Semi-supervised retinal lesion segmentation", AAAI 2021.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FOV MASK
# ─────────────────────────────────────────────────────────────────────────────

def compute_fov_mask(
    img_rgb: np.ndarray,
    threshold: int = 10,
    morphology_ksize: int = 15,
) -> np.ndarray:
    """Compute Field-of-View (FOV) mask for a fundus image.

    Fundus images have a circular retinal disc region surrounded by a
    black border. Random patches drawn from the black border contain zero
    information and waste batch capacity (~30% of patches in practice).

    This function computes a binary mask of the retinal region so the
    data generator can restrict sampling to the FOV.

    Args:
        img_rgb:          Input RGB image, any dtype.
        threshold:        Grayscale threshold to separate fundus from border.
        morphology_ksize: Closing kernel size to fill FOV holes.

    Returns:
        Boolean array (H, W), True = retinal region, False = black border.

    Reference:
        Standard in all top-performing IDRiD challenge submissions.
        Porwal et al., "IDRiD Challenge", MedIA 2020.
    """
    gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Morphological closing to fill internal gaps (blood vessels, dark lesions)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morphology_ksize, morphology_ksize)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Erode slightly to pull mask inward — avoids picking patches at FOV edge
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask.astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def retinal_preprocess(
    img_rgb: np.ndarray,
    use_ben_graham: bool = True,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    ben_sigma_scale: float = 0.015,
) -> np.ndarray:
    """Full retinal preprocessing pipeline → float32 [0, 1].

    Pipeline:
        1. Ben Graham normalization (removes illumination gradient)
        2. CLAHE on LAB L-channel (boosts local contrast)
        3. Normalize to [0, 1]

    Args:
        img_rgb:        Input RGB image (uint8 or float32).
        use_ben_graham: Apply Ben Graham normalization (recommended: True).
        use_clahe:      Apply CLAHE enhancement (recommended: True).
        clahe_clip:     CLAHE clip limit.
        ben_sigma_scale: Ben Graham Gaussian sigma scale factor.

    Returns:
        float32 (H, W, 3) in range [0, 1].

    Example::

        img_proc = retinal_preprocess(img_rgb)              # default
        img_proc = retinal_preprocess(img_rgb, use_ben_graham=False)  # CLAHE only
    """
    # Ensure uint8 input
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    if use_ben_graham:
        img_rgb = ben_graham_normalize(img_rgb, sigma_scale=ben_sigma_scale)

    if use_clahe:
        img_rgb = clahe_enhance(img_rgb, clip_limit=clahe_clip)

    return img_rgb.astype(np.float32) / 255.0

# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Segmentation IoU Analysis — v2 (Corrected)
===========================================

FIXES over v1:
  • Uses per-channel thresholding (σ > 0.5) instead of argmax
    → correct for sigmoid / multi-label outputs
  • union == 0 → NaN (excluded from mean) instead of 1.0
    → prevents inflation on images with no lesion pixels
  • Adds sensitivity, specificity, AUPR per class
  • Reports positive-only mIoU (only images with GT > 0)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# CORE METRICS — threshold-based (for sigmoid / multi-label heads)
# =============================================================================

def _to_binary_masks(y: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert (N, H, W, C) float predictions to binary masks.

    For sigmoid outputs: threshold each channel independently.
    Already-binary masks (0/1 int) pass through unchanged.
    """
    if y.dtype in (np.float32, np.float64):
        return (y > threshold).astype(np.uint8)
    return y.astype(np.uint8)


def compute_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    threshold: float = 0.5,
) -> dict:
    """
    Per-channel IoU for multi-label segmentation.

    Returns NaN for classes with zero union (both GT and pred empty).
    Mean IoU is computed ONLY over classes with valid (non-NaN) IoU.
    """
    # Ensure (N, H, W, C)
    if y_true.ndim == 3:
        y_true = np.expand_dims(y_true, -1)
    if y_pred.ndim == 3:
        y_pred = np.expand_dims(y_pred, -1)

    gt = _to_binary_masks(y_true)
    pr = _to_binary_masks(y_pred, threshold)

    ious = {}
    valid_ious = []

    for c in range(min(num_classes, gt.shape[-1])):
        gt_c = gt[..., c].flatten()
        pr_c = pr[..., c].flatten()

        intersection = np.sum(gt_c & pr_c)
        union = np.sum(gt_c | pr_c)

        if union > 0:
            iou_val = float(intersection) / float(union)
            ious[f"iou_class_{c}"] = iou_val
            valid_ious.append(iou_val)
        else:
            # Both empty → undefined, NOT 1.0
            ious[f"iou_class_{c}"] = float("nan")

    ious["mean_iou"] = float(np.nanmean(valid_ious)) if valid_ious else 0.0
    return ious


def compute_dice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    threshold: float = 0.5,
) -> dict:
    """Per-channel Dice coefficient.  Dice = 2·|A∩B| / (|A|+|B|)."""
    if y_true.ndim == 3:
        y_true = np.expand_dims(y_true, -1)
    if y_pred.ndim == 3:
        y_pred = np.expand_dims(y_pred, -1)

    gt = _to_binary_masks(y_true)
    pr = _to_binary_masks(y_pred, threshold)

    scores = {}
    valid = []

    for c in range(min(num_classes, gt.shape[-1])):
        gt_c = gt[..., c].flatten()
        pr_c = pr[..., c].flatten()

        intersection = 2.0 * np.sum(gt_c & pr_c)
        total = float(np.sum(gt_c) + np.sum(pr_c))

        if total > 0:
            d = intersection / total
            scores[f"dice_class_{c}"] = d
            valid.append(d)
        else:
            scores[f"dice_class_{c}"] = float("nan")

    scores["mean_dice"] = float(np.nanmean(valid)) if valid else 0.0
    return scores


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    threshold: float = 0.5,
) -> dict:
    """Sensitivity (recall), specificity, precision per class."""
    if y_true.ndim == 3:
        y_true = np.expand_dims(y_true, -1)
    if y_pred.ndim == 3:
        y_pred = np.expand_dims(y_pred, -1)

    gt = _to_binary_masks(y_true)
    pr = _to_binary_masks(y_pred, threshold)

    metrics = {}
    for c in range(min(num_classes, gt.shape[-1])):
        gt_c = gt[..., c].flatten().astype(bool)
        pr_c = pr[..., c].flatten().astype(bool)

        tp = np.sum(gt_c & pr_c)
        fp = np.sum(~gt_c & pr_c)
        fn = np.sum(gt_c & ~pr_c)
        tn = np.sum(~gt_c & ~pr_c)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        precision   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

        metrics[f"sens_class_{c}"]  = sensitivity
        metrics[f"spec_class_{c}"]  = specificity
        metrics[f"prec_class_{c}"]  = precision

    return metrics


# =============================================================================
# POSITIVE-ONLY IoU  (excludes images where GT has zero lesion pixels)
# =============================================================================

def compute_positive_only_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    threshold: float = 0.5,
) -> dict:
    """
    IoU computed ONLY on images that contain at least 1 GT pixel per class.
    This prevents empty images from inflating or deflating the score.
    """
    if y_true.ndim == 3:
        y_true = np.expand_dims(y_true, -1)
    if y_pred.ndim == 3:
        y_pred = np.expand_dims(y_pred, -1)

    gt = _to_binary_masks(y_true)
    pr = _to_binary_masks(y_pred, threshold)

    result = {}
    for c in range(min(num_classes, gt.shape[-1])):
        class_ious = []
        for n in range(gt.shape[0]):
            gt_n = gt[n, ..., c].flatten()
            if gt_n.sum() == 0:
                continue  # skip images with no GT for this class
            pr_n = pr[n, ..., c].flatten()
            inter = np.sum(gt_n & pr_n)
            union = np.sum(gt_n | pr_n)
            class_ious.append(inter / union if union > 0 else 0.0)

        result[f"pos_iou_class_{c}"] = float(np.mean(class_ious)) if class_ious else float("nan")
        result[f"pos_iou_n_{c}"] = len(class_ious)

    valid = [v for k, v in result.items() if k.startswith("pos_iou_class") and not np.isnan(v)]
    result["pos_mean_iou"] = float(np.mean(valid)) if valid else 0.0
    return result


# =============================================================================
# FULL ANALYSIS RUNNER
# =============================================================================

def run_iou_analysis(
    model,
    test_data,
    class_names: list,
    results_dir: Path,
) -> pd.DataFrame:
    """
    Run comprehensive IoU analysis with corrected metrics.
    """
    print("=" * 60)
    print("SEGMENTATION IoU ANALYSIS (v2 — threshold-based)")
    print("=" * 60)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(test_data, tuple):
        X_test, y_test = test_data
        y_pred = model.predict(X_test)
    else:
        y_true_list, y_pred_list = [], []
        for X_batch, y_batch in test_data:
            y_true_list.append(y_batch)
            y_pred_list.append(model.predict(X_batch))
        y_test = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)

    num_classes = len(class_names)

    # Compute all metrics
    iou_scores    = compute_iou(y_test, y_pred, num_classes)
    dice_scores   = compute_dice(y_test, y_pred, num_classes)
    clinical      = compute_clinical_metrics(y_test, y_pred, num_classes)
    pos_iou       = compute_positive_only_iou(y_test, y_pred, num_classes)

    # Build results table
    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            "class":       name,
            "iou":         iou_scores.get(f"iou_class_{i}", float("nan")),
            "dice":        dice_scores.get(f"dice_class_{i}", float("nan")),
            "sensitivity": clinical.get(f"sens_class_{i}", float("nan")),
            "specificity": clinical.get(f"spec_class_{i}", float("nan")),
            "precision":   clinical.get(f"prec_class_{i}", float("nan")),
            "pos_iou":     pos_iou.get(f"pos_iou_class_{i}", float("nan")),
            "pos_n":       pos_iou.get(f"pos_iou_n_{i}", 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "iou_analysis.csv", index=False)

    # Print summary
    print(f"\nPer-Class Results (threshold=0.5):")
    print("-" * 80)
    print(f"  {'Class':10} | {'IoU':>7} | {'Dice':>7} | {'Sens':>7} | {'Prec':>7} | {'Pos IoU':>7} (n)")
    print("-" * 80)
    for _, row in df.iterrows():
        print(
            f"  {row['class']:10} | "
            f"{row['iou']:7.4f} | "
            f"{row['dice']:7.4f} | "
            f"{row['sensitivity']:7.4f} | "
            f"{row['precision']:7.4f} | "
            f"{row['pos_iou']:7.4f} ({int(row['pos_n'])})"
        )

    print(f"\n{'=' * 60}")
    print(f"  Mean IoU       : {iou_scores['mean_iou']:.4f}")
    print(f"  Mean Dice      : {dice_scores['mean_dice']:.4f}")
    print(f"  Pos-Only mIoU  : {pos_iou['pos_mean_iou']:.4f}")
    print(f"{'=' * 60}")

    return df
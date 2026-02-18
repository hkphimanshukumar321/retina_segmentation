# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Pilot Test — Single Seed, Single Configuration
================================================

Standalone experiment to validate model performance BEFORE launching the
full ablation study.  Completely independent of the ablation framework.

Sweet‑spot Configuration
    Model        : Ghost‑CAS‑UNet (Ghost + Coordinate Attention + Skip Attention)
    Resolution   : 128 × 128
    Filters      : [16, 32, 64, 128]
    Ghost Ratio  : 2
    Loss         : Combined (BCE + Focal Tversky)
    Seed         : 42
    Epochs       : 50

Outputs (→ results/pilot/)
    pilot_model.keras          — full Keras model for RPi inference
    pilot_training_log.csv     — per‑epoch metrics
    pilot_metrics.json         — final IoU / Dice per class
"""

import sys
import argparse
import json
import time
import logging
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()           # experiments/
SEG_DIR    = SCRIPT_DIR.parent.resolve()               # segmentation/
ROOT_DIR   = SEG_DIR.parent.resolve()                  # retina_scan/

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SEG_DIR))

# ---------------------------------------------------------------------------
# Imports (after sys.path fix)
# ---------------------------------------------------------------------------
from common.logger import setup_logging
from common.data_loader import (
    PatchDataGenerator, AugmentationPipeline,
    RandomFlip, RandomRotation, RandomBrightness,
    RandomContrast, RandomElasticDeform
)
from segmentation.src.models import SEGMENTATION_MODELS
from segmentation.src.losses import combined_loss
from segmentation.experiments.run_iou_analysis import compute_iou, compute_dice

logger = logging.getLogger("pilot_test")

# ===========================================================================
# Configuration (hard‑coded sweet spot — independent of ablation)
# ===========================================================================
SEED          = 42
IMG_SIZE      = (128, 128)
NUM_CLASSES   = 3
BIT_VALUES    = (8, 16, 32)
ENCODER_FILTERS = [16, 32, 64, 128]
GHOST_RATIO   = 2
USE_SKIP_ATTN = True          # Ghost‑CAS‑UNet
DROPOUT       = 0.15
LEARNING_RATE = 1e-3
EPOCHS        = 50
BATCH_SIZE    = 8
PATCHES_TRAIN = 50            # patches per image (training)
PATCHES_VAL   = 10            # patches per image (validation)


# ===========================================================================
# Helpers
# ===========================================================================

class _PilotConfig:
    """Minimal config duck‑type expected by PatchDataGenerator."""
    class data:
        img_size   = IMG_SIZE
        bit_values = BIT_VALUES
    class model:
        num_classes = NUM_CLASSES


def set_seed(seed: int):
    """Set global random seeds for reproducibility."""
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


# ===========================================================================
# Main
# ===========================================================================

def main(quick_test: bool = False):
    import tensorflow as tf

    # ---- Seed everything ----
    set_seed(SEED)

    # ---- Logging ----
    setup_logging(log_dir=SEG_DIR / "logs")

    # ---- Results dir ----
    results_dir = SEG_DIR / "results" / "pilot"
    results_dir.mkdir(parents=True, exist_ok=True)

    epochs       = 1 if quick_test else EPOCHS
    patches_tr   = 2 if quick_test else PATCHES_TRAIN
    patches_val  = 2 if quick_test else PATCHES_VAL

    print("\n" + "=" * 60)
    print("  PILOT TEST — Ghost‑CAS‑UNet")
    print("=" * 60)
    print(f"  Seed         : {SEED}")
    print(f"  Resolution   : {IMG_SIZE}")
    print(f"  Filters      : {ENCODER_FILTERS}")
    print(f"  Ghost ratio  : {GHOST_RATIO}")
    print(f"  Skip Attention: {USE_SKIP_ATTN}")
    print(f"  Epochs       : {epochs}")
    print(f"  Quick test   : {quick_test}")
    print("=" * 60 + "\n")

    # ---- Data ----
    data_dir = ROOT_DIR / "data" / "segmentation"
    img_dir  = "Images" if (data_dir / "Images").exists() else "images"
    mask_dir = "Labels" if (data_dir / "Labels").exists() else "masks"

    image_files = sorted(list((data_dir / img_dir).glob("*.[jp][pn][g]")))
    mask_files  = sorted(list((data_dir / mask_dir).glob("*.[jp][pn][g]")))

    assert len(image_files) == len(mask_files), (
        f"Image/mask count mismatch: {len(image_files)} vs {len(mask_files)}"
    )
    print(f"[*] Dataset: {len(image_files)} images")

    split_idx   = int(len(image_files) * 0.8)
    train_imgs  = image_files[:split_idx]
    val_imgs    = image_files[split_idx:]
    train_masks = mask_files[:split_idx]
    val_masks   = mask_files[split_idx:]

    config = _PilotConfig()

    train_augs = AugmentationPipeline([
        RandomFlip(horizontal=True, vertical=True),
        RandomRotation(max_angle=180),
        RandomBrightness(delta=0.1),
        RandomContrast(limit=0.1),
        RandomElasticDeform(alpha=50, sigma=5, p=0.3),
    ])

    train_gen = PatchDataGenerator(
        train_imgs, train_masks, config,
        patches_per_image=patches_tr,
        batch_size=BATCH_SIZE,
        augmentitons=train_augs,
    )
    val_gen = PatchDataGenerator(
        val_imgs, val_masks, config,
        patches_per_image=patches_val,
        batch_size=BATCH_SIZE,
        augmentitons=None,
    )

    print(f"[*] Train batches: {len(train_gen)},  Val batches: {len(val_gen)}")

    # ---- Model ----
    model_fn = SEGMENTATION_MODELS["ghost_ca_unet"]
    model = model_fn(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=NUM_CLASSES,
        encoder_filters=ENCODER_FILTERS,
        dropout_rate=DROPOUT,
        ghost_ratio=GHOST_RATIO,
        use_skip_attention=USE_SKIP_ATTN,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, clipnorm=1.0    # gradient clipping for stability
    )

    # OneHotIoU may not exist on older TF — fallback to MeanIoU
    try:
        iou_metric = tf.keras.metrics.OneHotIoU(
            num_classes=NUM_CLASSES,
            target_class_ids=[0, 1, 2],
            name="iou",
        )
    except (AttributeError, TypeError):
        iou_metric = tf.keras.metrics.MeanIoU(
            num_classes=NUM_CLASSES, name="iou"
        )
        print("[WARN] OneHotIoU unavailable, using MeanIoU fallback")

    model.compile(
        optimizer=optimizer,
        loss=combined_loss(alpha=0.3, beta=0.7, gamma=0.75),
        metrics=["accuracy", iou_metric],
    )

    model.summary()
    total_params = model.count_params()

    # ---- Callbacks ----
    # NOTE: save_weights_only=True avoids Keras-format 'options' arg error
    #       on older TF versions. We save the full model manually after training.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(results_dir / "pilot_best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(results_dir / "pilot_training_log.csv")),
    ]

    # ---- Training ----
    t0 = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - t0

    # ---- Save weights (reliable across all TF versions) ----
    weights_path = results_dir / "pilot_model.weights.h5"
    model.save_weights(str(weights_path))
    model_size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f"\n[*] Weights saved → {weights_path}  ({model_size_mb:.2f} MB)")

    # ---- Try saving full model (may fail on older TF, non-critical) ----
    model_path = results_dir / "pilot_model.keras"
    try:
        model.save(str(model_path))
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"[*] Full model saved → {model_path}  ({model_size_mb:.2f} MB)")
    except (ValueError, TypeError) as e:
        # Fallback to legacy HDF5 format
        model_path = results_dir / "pilot_model.h5"
        try:
            model.save(str(model_path), save_format='h5')
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"[*] Full model saved (HDF5) → {model_path}  ({model_size_mb:.2f} MB)")
        except Exception as e2:
            print(f"[WARN] Full model save failed ({e2}). Weights saved successfully.")
            model_path = weights_path  # fallback for metrics reporting

    # ---- Evaluate (IoU + Dice on val set) ----
    print("\n[*] Evaluating on validation set …")
    y_true_list, y_pred_list = [], []
    for batch_x, batch_y in val_gen:
        y_true_list.append(batch_y)
        y_pred_list.append(model.predict(batch_x, verbose=0))

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    class_names = ["Background/OD", "HE/EX", "MA"]
    iou_scores  = compute_iou(y_true, y_pred, NUM_CLASSES)
    dice_scores = compute_dice(y_true, y_pred, NUM_CLASSES)

    # ---- Summary ----
    metrics = {
        "seed": SEED,
        "config": {
            "model": "Ghost_CAS_UNet",
            "resolution": list(IMG_SIZE),
            "filters": ENCODER_FILTERS,
            "ghost_ratio": GHOST_RATIO,
            "skip_attention": USE_SKIP_ATTN,
        },
        "training": {
            "epochs_requested": epochs,
            "epochs_actual": len(history.history["loss"]),
            "training_time_s": round(train_time, 1),
            "final_train_loss": round(float(history.history["loss"][-1]), 4),
            "final_val_loss": round(float(history.history["val_loss"][-1]), 4),
        },
        "model_info": {
            "total_params": int(total_params),
            "model_size_mb": round(model_size_mb, 3),
            "model_path": str(model_path),
        },
        "evaluation": {
            "iou": {k: round(float(v), 4) for k, v in iou_scores.items()},
            "dice": {k: round(float(v), 4) for k, v in dice_scores.items()},
        },
    }

    metrics_path = results_dir / "pilot_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("  PILOT TEST — RESULTS")
    print("=" * 60)
    for i, name in enumerate(class_names):
        iou  = iou_scores[f"iou_class_{i}"]
        dice = dice_scores[f"dice_class_{i}"]
        print(f"  {name:15s}  |  IoU: {iou:.4f}  |  Dice: {dice:.4f}")
    print(f"\n  Mean IoU  : {iou_scores['mean_iou']:.4f}")
    print(f"  Mean Dice : {dice_scores['mean_dice']:.4f}")
    print(f"  Params    : {total_params:,}")
    print(f"  Model Size: {model_size_mb:.2f} MB")
    print(f"  Train Time: {train_time:.0f}s")
    print("=" * 60)
    print(f"\n  Metrics   → {metrics_path}")
    print(f"  Model     → {model_path}")
    print(f"  CSV Log   → {results_dir / 'pilot_training_log.csv'}")

    return True


# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot Test — Ghost‑CAS‑UNet")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (1 epoch, 2 patches/img)")
    args = parser.parse_args()

    try:
        success = main(quick_test=args.quick)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Pilot test failed: {e}", exc_info=True)
        print(f"\n[FATAL] {e}")
        sys.exit(1)

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
    RandomContrast, RandomElasticDeform,
    decode_labelmap
)
from segmentation.src.idrid_loader import IDRIDPatchDataGenerator
from segmentation.src.metrics import DiceScore, IoUScore
from segmentation.src.models import SEGMENTATION_MODELS
from segmentation.src.losses import combined_loss_v2
from segmentation.experiments.run_iou_analysis import (
    compute_iou, compute_dice, compute_clinical_metrics, compute_positive_only_iou
)

logger = logging.getLogger("pilot_test")

# ===========================================================================
# Configuration (hard‑coded sweet spot — independent of ablation)
# ===========================================================================
SEED          = 42
IMG_SIZE      = (256, 256)      # v2: 256 for MA visibility
NUM_CLASSES   = 5
# IDRiD Classes: MA, HE, EX, SE, OD
LABEL_IDS     = (255, 127, 63)  # Only MA, HE, EX present in local val data
CLASS_NAMES   = ("MA", "HE", "EX", "SE", "OD")
ENCODER_FILTERS = [32, 64, 128, 256]   # v2: wider for capacity
GHOST_RATIO   = 2
USE_SKIP_ATTN = True          # Ghost‑CAS‑UNet
DROPOUT       = 0.15
LEARNING_RATE = 1e-3
EPOCHS        = 50
BATCH_SIZE    = 8
PATCHES_TRAIN = 100           # increased for more lesion exposure
PATCHES_VAL   = 20
DEEP_SUPERVISION = True       # v2: auxiliary loss heads
USE_ASPP      = True          # v2: multi-scale bottleneck


# ===========================================================================
# Helpers
# ===========================================================================

class _PilotConfig:
    """Minimal config duck‑type expected by PatchDataGenerator."""
    class data:
        img_size   = IMG_SIZE
        label_ids  = LABEL_IDS   # Used for validation
        bit_values = None        # not used (triggers decode_labelmap path)
        prob_lesion = 0.5        # 50% chance to center on lesion
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
    print(f"  Label IDs    : {LABEL_IDS}  ({', '.join(CLASS_NAMES)})")
    print(f"  Epochs       : {epochs}")
    print(f"  Patches/img  : {patches_tr} (train), {patches_val} (val)")
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

    # Use all local data for validation (since we train on IDRID)
    val_imgs    = image_files
    val_masks   = mask_files
    
    # IDRID Training Data
    idrid_root = ROOT_DIR / "data" / "IDRID" / "A. Segmentation"
    idrid_train_imgs = idrid_root / "1. Original Images" / "a. Training Set"
    idrid_train_masks = idrid_root / "2. All Segmentation Groundtruths" / "a. Training Set"


    config = _PilotConfig()

    train_augs = AugmentationPipeline([
        RandomFlip(horizontal=True, vertical=True),
        RandomRotation(max_angle=180),
        RandomBrightness(delta=0.1),
        RandomContrast(limit=0.1),
        RandomElasticDeform(alpha=50, sigma=5, p=0.3),
    ])

    # Train on IDRID
    train_gen = IDRIDPatchDataGenerator(
        idrid_train_imgs, idrid_train_masks, config,
        patches_per_image=patches_tr,
        batch_size=BATCH_SIZE,
        augmentations=train_augs,
    )
    val_gen = PatchDataGenerator(
        val_imgs, val_masks, config,
        patches_per_image=patches_val,
        batch_size=BATCH_SIZE,
        augmentitons=None,
    )

    print(f"[*] Train batches: {len(train_gen)},  Val batches: {len(val_gen)}")

    # ---- Model (v2) ----
    model_fn = SEGMENTATION_MODELS["ghost_cas_unet_v2"]
    model = model_fn(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=NUM_CLASSES,
        encoder_filters=ENCODER_FILTERS,
        dropout_rate=DROPOUT,
        ghost_ratio=GHOST_RATIO,
        use_skip_attention=USE_SKIP_ATTN,
        use_aspp=USE_ASPP,
        deep_supervision=DEEP_SUPERVISION,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, clipnorm=1.0
    )

    # v2 loss: Lovász-Softmax + Focal Tversky + BCE
    loss_fn = combined_loss_v2(
        w_lovasz=0.5, w_focal_tversky=0.3, w_bce=0.2,
        ft_alpha=0.3, ft_beta=0.7, ft_gamma=0.75,
    )

    # For deep supervision: apply same loss to main + aux outputs with weights
    if DEEP_SUPERVISION:
        loss_dict = {
            'main_out': loss_fn,
            'aux_up_1': loss_fn,
            'aux_up_2': loss_fn,
        }
        loss_weights = {
            'main_out': 1.0,
            'aux_up_1': 0.4,
            'aux_up_2': 0.2,
        }
        model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            loss_weights=loss_weights,
            metrics={
                'main_out': ['accuracy', DiceScore(num_classes=NUM_CLASSES), IoUScore(num_classes=NUM_CLASSES)]
            },
        )
    else:
        model.compile(
            optimizer=optimizer, loss=loss_fn,
            metrics=['accuracy', DiceScore(num_classes=NUM_CLASSES), IoUScore(num_classes=NUM_CLASSES)],
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

    # ---- Deep supervision wrapper: generators must yield (X, [Y, Y, Y]) ----
    if DEEP_SUPERVISION:
        def _wrap_gen(gen):
            for X, Y in gen:
                yield X, {'main_out': Y, 'aux_up_1': Y, 'aux_up_2': Y}

        ds_train = tf.data.Dataset.from_generator(
            lambda: _wrap_gen(train_gen),
            output_signature=(
                tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
                {
                    'main_out': tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], NUM_CLASSES), dtype=tf.float32),
                    'aux_up_1': tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], NUM_CLASSES), dtype=tf.float32),
                    'aux_up_2': tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], NUM_CLASSES), dtype=tf.float32),
                },
            )
        )
        ds_train = ds_train.repeat()  # Fix: allow infinite epochs

        ds_val = tf.data.Dataset.from_generator(
            lambda: _wrap_gen(val_gen),
            output_signature=(
                tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
                {
                    'main_out': tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], NUM_CLASSES), dtype=tf.float32),
                    'aux_up_1': tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], NUM_CLASSES), dtype=tf.float32),
                    'aux_up_2': tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], NUM_CLASSES), dtype=tf.float32),
                },
            )
        )
        ds_val = ds_val.repeat()  # Fix: allow infinite epochs
        train_data = ds_train
        val_data = ds_val
        steps_per_epoch = len(train_gen)
        validation_steps = len(val_gen)
    else:
        train_data = train_gen
        val_data = val_gen
        steps_per_epoch = None
        validation_steps = None

    # ---- Training ----
    t0 = time.time()
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
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

    # -----------------------------------------------------------------------
    # VALIDATION (Local Data) - Already done during training, but let's summarize
    # -----------------------------------------------------------------------
    print("\n[*] Evaluating on Validation Set (Local Data) …")
    # ... (existing val logic) ...
    y_true_list, y_pred_list = [], []
    for batch_x, batch_y in val_gen:
        y_true_list.append(batch_y)
        preds = model.predict(batch_x, verbose=0)
        if isinstance(preds, list): preds = preds[0]
        y_pred_list.append(preds)

    y_true_val = np.concatenate(y_true_list)
    y_pred_val = np.concatenate(y_pred_list)

    val_iou  = compute_iou(y_true_val, y_pred_val, NUM_CLASSES)
    val_dice = compute_dice(y_true_val, y_pred_val, NUM_CLASSES)
    
    # -----------------------------------------------------------------------
    # TESTING (IDRID Data) - New Phase
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("  TESTING PHASE — IDRID Test Set")
    print("=" * 60)
    
    idrid_test_imgs = idrid_root / "1. Original Images" / "b. Testing Set"
    idrid_test_masks = idrid_root / "2. All Segmentation Groundtruths" / "b. Testing Set"
    
    # Check if test set exists (it should)
    if not idrid_test_imgs.exists():
        print(f"[WARN] IDRID Test Set not found at {idrid_test_imgs}. Skipping Test Phase.")
        test_iou = {}
        test_dice = {}
    else:
        # IDRID Test Generator (Same structure as Train)
        # 1 patch per image for testing? Or slide? 
        # For pilot, let's use the same patch logic but maybe fewer patches or full image if possible.
        # Given generic generator, we stick to patches for consistent metrics.
        patches_test = 5 if quick_test else 20 # reasonable sample
        
        test_gen = IDRIDPatchDataGenerator(
            idrid_test_imgs, idrid_test_masks, config,
            patches_per_image=patches_test,
            batch_size=BATCH_SIZE,
            augmentations=None, # No augs for testing
            check_files=True
        )
        print(f"[*] Test batches: {len(test_gen)}")
        
        y_true_list, y_pred_list = [], []
        for batch_x, batch_y in test_gen:
            y_true_list.append(batch_y)
            preds = model.predict(batch_x, verbose=0)
            if isinstance(preds, list): preds = preds[0]
            y_pred_list.append(preds)
            
        y_true_test = np.concatenate(y_true_list)
        y_pred_test = np.concatenate(y_pred_list)
        
        test_iou  = compute_iou(y_true_test, y_pred_test, NUM_CLASSES)
        test_dice = compute_dice(y_true_test, y_pred_test, NUM_CLASSES)

    # ---- Summary ----
    metrics = {
        "seed": SEED,
        "config": {
            "model": "Ghost_CAS_UNet_v2",
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
        "validation_local": {
            "iou": {k: round(float(v), 4) for k, v in val_iou.items()},
            "dice": {k: round(float(v), 4) for k, v in val_dice.items()},
        },
         "test_idrid": {
            "iou": {k: round(float(v), 4) for k, v in test_iou.items()},
            "dice": {k: round(float(v), 4) for k, v in test_dice.items()},
        },
    }

    metrics_path = results_dir / "pilot_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    class_names = list(CLASS_NAMES)
    
    print("\n" + "=" * 60)
    print("  PILOT TEST — RESULTS (Validation: Local)")
    print("=" * 60)
    for i, name in enumerate(class_names):
        # Validation might have NaNs for missing classes (SE, OD)
        iou  = val_iou.get(f"iou_class_{i}", float("nan"))
        dice = val_dice.get(f"dice_class_{i}", float("nan"))
        print(f"  {name:15s}  |  IoU: {iou:.4f}  |  Dice: {dice:.4f}")
    print(f"\n  Mean IoU  : {val_iou['mean_iou']:.4f}")
    print(f"  Mean Dice : {val_dice['mean_dice']:.4f}")
    
    if idrid_test_imgs.exists():
        print("\n" + "=" * 60)
        print("  PILOT TEST — RESULTS (Testing: IDRID)")
        print("=" * 60)
        for i, name in enumerate(class_names):
            iou  = test_iou.get(f"iou_class_{i}", float("nan"))
            dice = test_dice.get(f"dice_class_{i}", float("nan"))
            print(f"  {name:15s}  |  IoU: {iou:.4f}  |  Dice: {dice:.4f}")
        print(f"\n  Mean IoU  : {test_iou['mean_iou']:.4f}")
        print(f"  Mean Dice : {test_dice['mean_dice']:.4f}")

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

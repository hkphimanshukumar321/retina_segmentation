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
from common.hardware import setup_gpu_memory_growth, get_gpu_info
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
BATCH_SIZE    = 16
PATCHES_TRAIN = 100           # patches per image for IDRID training (43 imgs)
PATCHES_VAL   = 30            # patches per image for IDRID val split (11 imgs)
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

    # ---- Hardware configuration ----
    setup_gpu_memory_growth()
    gpu_info = get_gpu_info()
    print(f"\n[*] Hardware Detection:")
    print(f"    - GPUs Found: {gpu_info['num_gpus']}")
    print(f"    - Names: {', '.join(gpu_info['gpu_names'])}")

    # ---- Logging ----
    setup_logging(log_dir=SEG_DIR / "logs")

    # ---- Results dir ----
    results_dir = SEG_DIR / "results" / "pilot"
    results_dir.mkdir(parents=True, exist_ok=True)

    epochs       = 1 if quick_test else EPOCHS
    patches_tr   = 2 if quick_test else PATCHES_TRAIN
    patches_val  = 2 if quick_test else PATCHES_VAL   # IDRID val split

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

    # ---- IDRID Dataset Paths ----
    idrid_root = ROOT_DIR / "data" / "IDRID" / "A. Segmentation"

    # Training split:  IDRiD_01 – IDRiD_43 (43 imgs, 5 classes)
    idrid_train_imgs  = idrid_root / "1. Original Images"               / "a. Training Set"
    idrid_train_masks = idrid_root / "2. All Segmentation Groundtruths" / "a. Training Set"

    # Validation split: IDRiD_44 – IDRiD_54 (11 imgs, 5 classes)
    # Created by: python segmentation/scripts/prepare_val_split.py
    idrid_val_imgs    = idrid_root / "1. Original Images"               / "a. Val Set"
    idrid_val_masks   = idrid_root / "2. All Segmentation Groundtruths" / "a. Val Set"

    # Local data (for post-training generalization test only — 3 classes: MA, HE, EX)
    local_data_dir = ROOT_DIR / "data" / "segmentation"
    local_img_dir  = "Images" if (local_data_dir / "Images").exists() else "images"
    local_mask_dir = "Labels" if (local_data_dir / "Labels").exists() else "masks"
    local_image_files = sorted(list((local_data_dir / local_img_dir).glob("*.[jp][pn][g]")))
    local_mask_files  = sorted(list((local_data_dir / local_mask_dir).glob("*.[jp][pn][g]")))

    # Sanity checks
    if not idrid_val_imgs.exists():
        raise RuntimeError(
            f"IDRID val split not found: {idrid_val_imgs}\n"
            f"Run first: python segmentation/scripts/prepare_val_split.py"
        )

    print(f"[*] IDRID train dir : {idrid_train_imgs}")
    print(f"[*] IDRID val   dir : {idrid_val_imgs}")
    print(f"[*] Local data      : {len(local_image_files)} images (used for generalization test only)")

    config = _PilotConfig()

    train_augs = AugmentationPipeline([
        RandomFlip(horizontal=True, vertical=True),
        RandomRotation(max_angle=180),
        RandomBrightness(delta=0.1),
        RandomContrast(limit=0.1),
        RandomElasticDeform(alpha=50, sigma=5, p=0.3),
    ])

    # ---- Distributed Strategy ----
    strategy = tf.distribute.MirroredStrategy()
    print(f"\n[*] Strategy: {strategy.num_replicas_in_sync} replicas in sync.")
    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

    # Train generator: IDRID training split (43 images, 5 classes, augmented)
    train_gen = IDRIDPatchDataGenerator(
        idrid_train_imgs, idrid_train_masks, config,
        patches_per_image=patches_tr,
        batch_size=global_batch_size,
        augmentations=train_augs,
    )
    # Val generator: IDRID val split (11 images, same 5 classes, no augmentation)
    # This ensures EarlyStopping monitors a loss from the SAME class distribution as training.
    val_gen = IDRIDPatchDataGenerator(
        idrid_val_imgs, idrid_val_masks, config,
        patches_per_image=patches_val,
        batch_size=global_batch_size,
        augmentations=None,
    )

    print(f"[*] Train batches: {len(train_gen)},  Val batches (IDRID split): {len(val_gen)}")

    # ---- Data Sanity Check (Professional Verification) ----
    print("\n" + "-" * 40)
    print("  DATA SANITY CHECK")
    print("-" * 40)
    try:
        # Fetch one batch
        X_sample, Y_sample = train_gen[0] # (B, H, W, 3), (B, H, W, 5)
        print(f"[*] Sample Batch Shape: Input {X_sample.shape}, Target {Y_sample.shape}")
        
        # Check Value Ranges
        print(f"[*] Input Range       : [{X_sample.min():.2f}, {X_sample.max():.2f}] (Expected 0.0-1.0)")
        
        # Check Class Distribution in this batch
        # Y_sample is (B, H, W, 5)
        total_pixels = Y_sample.size
        print("[*] Target Class Distribution (Pixels):")
        for i, name in enumerate(CLASS_NAMES):
            count = Y_sample[..., i].sum()
            ratio = (count / (X_sample.shape[0]*X_sample.shape[1]*X_sample.shape[2])) * 100
            print(f"    - {name}: {int(count)} pixels ({ratio:.2f}%)")
            
        print("-" * 40 + "\n")
    except Exception as e:
        print(f"[WARN] Sanity check failed: {e}")

    # ---- Model (v2) ----
    with strategy.scope():
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

    # ---- Custom Callback for Clean Logging ----
    class SimpleLogger(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.start_time = None
            self.epoch_start_time = None
            
        def on_train_begin(self, logs=None):
            self.start_time = time.time()
            
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
            if self.start_time is None:  # Fallback if on_train_begin wasn't called properly
                self.start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            
            # Timing and ETA
            epoch_time = time.time() - self.epoch_start_time
            total_elapsed = time.time() - self.start_time
            epochs_done = epoch + 1
            total_epochs = self.params['epochs']
            
            avg_time_per_epoch = total_elapsed / epochs_done
            epochs_remaining = total_epochs - epochs_done
            eta_seconds = epochs_remaining * avg_time_per_epoch
            
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            
            # Filter and rename keys for display
            display_logs = {}
            for k, v in logs.items():
                if "val_" in k:
                    prefix = "val_"
                    base = k[4:]
                else:
                    prefix = ""
                    base = k
                
                # Simplify names: main_out_iou_score -> iou
                if "main_out_" in base:
                    clean_name = base.replace("main_out_", "")
                elif "aux" in base:
                    continue # Skip aux metrics
                else:
                    clean_name = base
                
                display_logs[prefix + clean_name] = v
            
            # Format output
            msg = f"Epoch {epochs_done}/{total_epochs}"
            msg += f" [{epoch_time:.0f}s, ETA: {eta_str}]"
            msg += f" - loss: {display_logs.get('loss', 0):.4f}"
            msg += f" - acc: {display_logs.get('accuracy', 0):.4f}"
            msg += f" - iou: {display_logs.get('iou_score', 0):.4f}"
            msg += f" - dice: {display_logs.get('dice_score', 0):.4f}"
            
            if 'val_loss' in display_logs:
                msg += f" | val_loss: {display_logs['val_loss']:.4f}"
                msg += f" - val_acc: {display_logs.get('val_accuracy', 0):.4f}"
                msg += f" - val_iou: {display_logs.get('val_iou_score', 0):.4f}"
                msg += f" - val_dice: {display_logs.get('val_dice_score', 0):.4f}"
            
            print(msg)

    CLASS_NAMES_CB = ("MA", "HE", "EX", "SE", "OD")

    class PerClassMetricsCallback(tf.keras.callbacks.Callback):
        """Print per-class IoU & Dice on val_gen after every epoch.
        Classes with IoU < 0.05 are flagged STARVING for immediate visibility.
        """
        def __init__(self, generator, num_classes, class_names,
                     threshold=0.5, every_n=1):
            super().__init__()
            self.generator   = generator
            self.num_classes = num_classes
            self.class_names = class_names
            self.threshold   = threshold
            self.every_n     = every_n

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.every_n != 0:
                return
            y_true_list, y_pred_list = [], []
            for i in range(len(self.generator)):
                bx, by = self.generator[i]
                preds = self.model.predict(bx, verbose=0)
                if isinstance(preds, list):
                    preds = preds[0]  # main head
                gt = by if not isinstance(by, dict) else by["main_out"]
                y_true_list.append(gt)
                y_pred_list.append(preds)
            y_true = np.concatenate(y_true_list, axis=0)
            y_pred = (np.concatenate(y_pred_list, axis=0) > self.threshold).astype(np.float32)

            print(f"\n  +-- Per-Class Val IoU/Dice  (Epoch {epoch+1}) ------+")
            print(f"  | {'Class':<5}  {'IoU':>6}  {'Dice':>6}  {'GT px':>8}  Status")
            print(f"  | {'─'*48}")
            for c in range(self.num_classes):
                name  = self.class_names[c] if c < len(self.class_names) else f"C{c}"
                gt_c  = y_true[..., c]
                pr_c  = y_pred[..., c]
                gt_px = int(gt_c.sum())
                tp = float(np.sum(gt_c * pr_c))
                fp = float(np.sum((1-gt_c) * pr_c))
                fn = float(np.sum(gt_c * (1-pr_c)))
                iou  = tp/(tp+fp+fn)   if (tp+fp+fn)  >0 else float("nan")
                dice = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else float("nan")
                iou_s  = f"{iou:.4f}"  if not np.isnan(iou)  else "  nan "
                dice_s = f"{dice:.4f}" if not np.isnan(dice) else "  nan "
                flag   = "  << STARVING" if (np.isnan(iou) or iou < 0.05) else ""
                print(f"  | {name:<5}  {iou_s:>6}  {dice_s:>6}  {gt_px:>8,}{flag}")
            print(f"  +{'─'*52}+\n")

    # ---- Callbacks ----
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(results_dir / "pilot_best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(results_dir / "pilot_training_log.csv")),
        SimpleLogger(),
        PerClassMetricsCallback(
            generator=val_gen,
            num_classes=NUM_CLASSES,
            class_names=CLASS_NAMES_CB,
            threshold=0.5,
            every_n=1,   # every epoch; set to 2-5 to reduce overhead
        ),
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
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
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
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
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
    
    # We remove SimpleLogger from callbacks and use verbose=1 
    # to render the default ETA + Batch progress bar accurately over TQDM outputs on CLI
    # Reinitialize standard logs while keeping CSV backup
    active_callbacks = [c for c in callbacks if not isinstance(c, SimpleLogger)]
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=active_callbacks,
        verbose=1, # Enabled standard keras progress bar (fix #4)
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
    # EVALUATION 1: IDRID Val Split (same data used for EarlyStopping monitoring)
    # -----------------------------------------------------------------------
    print("\n[*] Evaluating on IDRID Validation Split (11 images, full 5 classes) …")
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
    # EVALUATION 2: Local Data Generalization Test (3 classes: MA, HE, EX)
    # Note: SE and OD will be NaN here as local data has no such annotations.
    # -----------------------------------------------------------------------
    local_iou, local_dice = {}, {}
    if local_image_files:
        print("\n[*] Evaluating generalization on Local Data (3-class vessel labels) …")
        local_gen = PatchDataGenerator(
            local_image_files, local_mask_files, config,
            patches_per_image=patches_val,
            batch_size=global_batch_size,
            augmentitons=None,
        )
        ly_true_list, ly_pred_list = [], []
        for batch_x, batch_y in local_gen:
            ly_true_list.append(batch_y)
            preds = model.predict(batch_x, verbose=0)
            if isinstance(preds, list): preds = preds[0]
            ly_pred_list.append(preds)
        ly_true = np.concatenate(ly_true_list)
        ly_pred = np.concatenate(ly_pred_list)
        local_iou  = compute_iou(ly_true, ly_pred, NUM_CLASSES)
        local_dice = compute_dice(ly_true, ly_pred, NUM_CLASSES)
    
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
            batch_size=global_batch_size,
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
    def _safe_round(d: dict) -> dict:
        """Round floats; keep NaN as None for JSON serialization."""
        out = {}
        for k, v in d.items():
            try:
                import math
                out[k] = None if math.isnan(float(v)) else round(float(v), 4)
            except Exception:
                out[k] = v
        return out

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
        "validation_idrid_split": {  # IDRID val split — same 5 classes as training
            "note": "IDRiD_44-54 (11 images, all 5 classes)",
            "iou": _safe_round(val_iou),
            "dice": _safe_round(val_dice),
        },
        "generalization_local": {   # Local data — only MA/HE/EX (SE & OD are NaN)
            "note": "Local vessel labels (3 classes). SE & OD are NaN (no GT).",
            "iou": _safe_round(local_iou) if local_iou else {},
            "dice": _safe_round(local_dice) if local_dice else {},
        },
        "test_idrid": {
            "note": "Official IDRID test set (IDRiD_55-81, all 5 classes)",
            "iou": _safe_round(test_iou),
            "dice": _safe_round(test_dice),
        },
    }

    metrics_path = results_dir / "pilot_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    class_names = list(CLASS_NAMES)

    def _print_results(title: str, iou_d: dict, dice_d: dict, note: str = ""):
        import math
        print("\n" + "=" * 60)
        print(f"  {title}")
        if note:
            print(f"  [{note}]")
        print("=" * 60)
        for i, name in enumerate(class_names):
            iou  = iou_d.get(f"iou_class_{i}",  float("nan"))
            dice = dice_d.get(f"dice_class_{i}", float("nan"))
            iou_str  = f"{iou:.4f}"  if not (isinstance(iou,  float) and math.isnan(iou))  else "  nan "
            dice_str = f"{dice:.4f}" if not (isinstance(dice, float) and math.isnan(dice)) else "  nan "
            print(f"  {name:15s}  |  IoU: {iou_str}  |  Dice: {dice_str}")
        mean_iou  = iou_d.get('mean_iou',   float('nan'))
        mean_dice = dice_d.get('mean_dice',  float('nan'))
        print(f"\n  Mean IoU  : {mean_iou:.4f}")
        print(f"  Mean Dice : {mean_dice:.4f}")

    _print_results(
        "RESULTS — IDRID Val Split (5 classes, monitoring set)",
        val_iou, val_dice,
        note="IDRiD_44 – IDRiD_54"
    )

    if local_iou:
        _print_results(
            "RESULTS — Local Data Generalization (3 classes: MA/HE/EX)",
            local_iou, local_dice,
            note="SE & OD are NaN — no GT annotations in local data"
        )

    if idrid_test_imgs.exists():
        _print_results(
            "RESULTS — IDRID Official Test Set (5 classes)",
            test_iou, test_dice,
            note="IDRiD_55 – IDRiD_81"
        )

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

# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Segmentation Master Runner
==========================

Main entry point for segmentation experiments.

Usage::

    # Main model (Ghost-CAS-UNet v2)
    python run.py
    python run.py --quick          # smoke test (1 epoch)

    # Ablation study
    python run.py --ablation
    python run.py --ablation --quick

    # Baseline comparison
    python run.py --baselines
    python run.py --baselines --quick
"""

import sys
import os

# Server/A100 Optimization & Fixes
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Disable problematic Grappler layout optimizer that often hangs on A100 MIG
os.environ.setdefault("TF_GRAF_OPTIMIZER_JIT_COMPILE", "0")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import random
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — must come BEFORE any local imports
# ---------------------------------------------------------------------------
current_dir = Path(__file__).parent.resolve()   # segmentation/
root_dir = current_dir.parent                   # retina_scan/
sys.path.insert(0, str(root_dir))               # for common.*
sys.path.insert(0, str(current_dir))            # for config, baselines, src.*

# ---------------------------------------------------------------------------
# Local imports (paths are now correct)
# ---------------------------------------------------------------------------
from common.logger import setup_logging
from common.hardware import get_gpu_info
from common.ablation import BaseAblationStudy, AblationParameter
from config import SegmentationConfig

from src.models import SEGMENTATION_MODELS
from src.losses import combined_loss, combined_loss_v2
from src.metrics import DiceScore, IoUScore

logger = logging.getLogger("segmentation_runner")

# Class names for reporting
CLASS_NAMES = ("MA", "HE", "EX", "SE", "OD")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = 42):
    """Set global random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    logger.info(f"Random seeds set to {seed}")


# =============================================================================
# DATA HELPERS — shared by main, ablation, and baselines
# =============================================================================

def _resolve_data(config: SegmentationConfig):
    """Locate image/mask files for training and validation.

    Priority:
      1. IDRID detected + val split exists → train on IDRID train split,
         validate on IDRID val split (same 5-class distribution).  ← CORRECT
      2. IDRID detected but no val split → raise RuntimeError with instructions.
      3. No IDRID → 80/20 split on local data.

    Returns:
        (train_imgs, train_masks, val_imgs, val_masks, data_dir) or None
    """
    data_dir = Path(config.data.data_dir or (root_dir / "data" / "segmentation"))
    if not data_dir.exists():
        from common.interactive_setup import setup_dataset_interactive
        if setup_dataset_interactive("segmentation", data_dir):
            print("\n[*] Data setup complete. Resuming experiment...")
        else:
            return None

    img_subdir = "Images" if (data_dir / "Images").exists() else "images"
    mask_subdir = "Labels" if (data_dir / "Labels").exists() else "masks"

    image_files = sorted(list((data_dir / img_subdir).glob("*.[jp][pn][g]")))
    mask_files = sorted(list((data_dir / mask_subdir).glob("*.[jp][pn][g]")))

    if len(image_files) != len(mask_files):
        logger.error(f"Mismatch: {len(image_files)} images vs {len(mask_files)} masks!")
        return None

    # Check if IDRID training data is available
    idrid_root = root_dir / "data" / "IDRID" / "A. Segmentation"
    idrid_train_imgs  = idrid_root / "1. Original Images"               / "a. Training Set"
    idrid_val_imgs    = idrid_root / "1. Original Images"               / "a. Val Set"
    idrid_train_masks = idrid_root / "2. All Segmentation Groundtruths" / "a. Training Set"
    idrid_val_masks   = idrid_root / "2. All Segmentation Groundtruths" / "a. Val Set"
    use_idrid = idrid_train_imgs.exists() and idrid_train_masks.exists()

    if use_idrid:
        # Require the val split — must run prepare_val_split.py first
        if not idrid_val_imgs.exists():
            raise RuntimeError(
                f"IDRID val split not found: {idrid_val_imgs}\n"
                f"Run first:  python segmentation/scripts/prepare_val_split.py"
            )
        # train/val both come from IDRID (same 5-class distribution)
        # local data is NOT used for val — it would corrupt val_loss (only 3 classes)
        train_imgs, train_masks = [], []  # placeholder; _build_generators uses IDRID paths
        val_imgs   = []                   # placeholder; _build_generators uses IDRID val path
        val_masks  = []
        logger.info(
            f"IDRID detected → training on IDRID train split, "
            f"validating on IDRID val split (a. Val Set)."
        )
        logger.info(f"Local data ({len(image_files)} imgs) kept for post-training generalization only.")
    else:
        # No IDRID → 80/20 split on local data
        split_idx = int(len(image_files) * 0.8)
        train_imgs, val_imgs = image_files[:split_idx], image_files[split_idx:]
        train_masks, val_masks = mask_files[:split_idx], mask_files[split_idx:]
        logger.info(f"Local data split: {len(train_imgs)} train, {len(val_imgs)} val")

    return train_imgs, train_masks, val_imgs, val_masks, data_dir


def _build_generators(train_imgs, train_masks, val_imgs, val_masks, config,
                      quick_test: bool = False):
    """Build train/val patch generators."""
    from common.data_loader import (
        PatchDataGenerator, AugmentationPipeline,
        RandomFlip, RandomRotation, RandomBrightness,
        RandomContrast, RandomElasticDeform,
    )

    # Check for IDRID dataset (preferred for training)
    idrid_root = root_dir / "data" / "IDRID" / "A. Segmentation"
    idrid_train_imgs  = idrid_root / "1. Original Images"               / "a. Training Set"
    idrid_train_masks = idrid_root / "2. All Segmentation Groundtruths" / "a. Training Set"
    idrid_val_imgs    = idrid_root / "1. Original Images"               / "a. Val Set"
    idrid_val_masks   = idrid_root / "2. All Segmentation Groundtruths" / "a. Val Set"
    use_idrid = idrid_train_imgs.exists() and idrid_train_masks.exists()

    train_augs = AugmentationPipeline([
        RandomFlip(horizontal=True, vertical=True),
        RandomRotation(max_angle=180),
        RandomBrightness(delta=0.1),
        RandomContrast(limit=0.1),
        RandomElasticDeform(alpha=50, sigma=5, p=0.3),
    ])

    patches_tr  = 2 if quick_test else config.training.patches_per_image
    patches_val = 2 if quick_test else config.training.patches_per_image_val

    if use_idrid:
        from src.idrid_loader import IDRIDPatchDataGenerator
        logger.info("Using IDRID dataset for training.")
        train_gen = IDRIDPatchDataGenerator(
            idrid_train_imgs, idrid_train_masks, config,
            patches_per_image=patches_tr,
            batch_size=config.training.batch_size,
            augmentations=train_augs,
        )
        # Val also on IDRID val split — same 5-class distribution as training
        # This is critical for EarlyStopping to work correctly.
        logger.info("Using IDRID val split for validation (same 5-class distribution).")
        val_gen = IDRIDPatchDataGenerator(
            idrid_val_imgs, idrid_val_masks, config,
            patches_per_image=patches_val,
            batch_size=config.training.batch_size,
            augmentations=None,
        )
    else:
        logger.info("Using local dataset for training.")
        train_gen = PatchDataGenerator(
            train_imgs, train_masks, config,
            patches_per_image=patches_tr,
            batch_size=config.training.batch_size,
            augmentitons=train_augs,
        )
        val_gen = PatchDataGenerator(
            val_imgs, val_masks, config,
            patches_per_image=patches_val,
            batch_size=config.training.batch_size,
            augmentitons=None,
        )

    return train_gen, val_gen


def _build_test_generator(config, quick_test: bool = False):
    """Build IDRID test set generator. Returns None if test set not found."""
    idrid_root = root_dir / "data" / "IDRID" / "A. Segmentation"
    idrid_test_imgs = idrid_root / "1. Original Images" / "b. Testing Set"
    idrid_test_masks = idrid_root / "2. All Segmentation Groundtruths" / "b. Testing Set"

    if not idrid_test_imgs.exists():
        logger.warning(f"IDRID test set not found at {idrid_test_imgs}")
        return None

    from src.idrid_loader import IDRIDPatchDataGenerator
    patches_test = 5 if quick_test else 20

    test_gen = IDRIDPatchDataGenerator(
        idrid_test_imgs, idrid_test_masks, config,
        patches_per_image=patches_test,
        batch_size=config.training.batch_size,
        augmentations=None,
        check_files=True,
        enable_copy_paste=False,   # No copy-paste at test/val time
    )
    return test_gen


def _train_model(model, train_gen, val_gen, config, results_dir: Path,
                 quick_test: bool = False, model_prefix: str = "model"):
    """Compile, train, and save a model. Returns (history, metrics_dict)."""
    import tensorflow as tf

    results_dir.mkdir(parents=True, exist_ok=True)
    epochs = 1 if quick_test else config.training.epochs

    # --- Compile ---
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.training.learning_rate,
        clipnorm=config.training.clip_norm,
    )

    # Choose loss based on model name
    model_name_lower = model.name.lower() if hasattr(model, "name") else ""
    if "ghost" in model_name_lower:
        # Class-weighted Tversky: MA x3.0, HE x1.5, EX x1.0, SE x2.5, OD x0.5
        # Matches pilot_test.py — Porwal et al. MedIA 2020 top-team strategy
        IDRID_CLASS_WEIGHTS = [3.0, 1.5, 1.0, 2.5, 0.5]
        loss_fn = combined_loss_v2(
            w_lovasz=0.5, w_focal_tversky=0.3, w_bce=0.2,
            ft_alpha=0.3, ft_beta=0.7, ft_gamma=0.75,
            class_weights=IDRID_CLASS_WEIGHTS,
        )
    else:
        loss_fn = combined_loss(alpha=0.3, beta=0.7, gamma=0.75)

    metrics_list = [
        "accuracy",
        DiceScore(num_classes=config.model.num_classes),
        IoUScore(num_classes=config.model.num_classes),
    ]

    # Deep supervision handling
    is_ds = len(model.outputs) > 1
    if is_ds:
        # Use model.output_names (e.g. ['main_out', 'aux_up_1', 'aux_up_2'])
        # NOT out.name.split('/') which gives keras_tensor_* tensor names
        out_names = list(model.output_names)
        loss_dict = {n: loss_fn for n in out_names}
        weight_dict = {}
        for i, n in enumerate(out_names):
            weight_dict[n] = 1.0 if i == 0 else max(0.2, 0.6 - 0.2 * i)
        model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            loss_weights=weight_dict,
            metrics={out_names[0]: metrics_list},  # metrics only on main head
        )
    else:
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_list)

    model.summary(print_fn=logger.info)

    # --- Custom Logger ---
    class SimpleLogger(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.start_time = None
            self.epoch_start_time = None
            
        def on_train_begin(self, logs=None):
            self.start_time = time.time()
            
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
            if self.start_time is None:
                self.start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            epoch_time = time.time() - self.epoch_start_time
            total_elapsed = time.time() - self.start_time
            epochs_done = epoch + 1
            total_epochs = self.params['epochs']
            
            avg_time_per_epoch = total_elapsed / epochs_done
            eta_seconds = (total_epochs - epochs_done) * avg_time_per_epoch
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            
            display_logs = {}
            for k, v in logs.items():
                prefix = "val_" if "val_" in k else ""
                base = k[4:] if "val_" in k else k
                clean_name = base.replace("main_out_", "") if "main_out_" in base else base
                if "aux" not in base:
                    display_logs[prefix + clean_name] = v
            
            msg = f"Epoch {epochs_done}/{total_epochs} [{epoch_time:.0f}s, ETA: {eta_str}]"
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

    CLASS_NAMES_RUN = ("MA", "HE", "EX", "SE", "OD")

    class PerClassMetricsCallback(tf.keras.callbacks.Callback):
        """Print per-class IoU & Dice on val_gen every epoch (mirrors pilot_test.py)."""
        def __init__(self, generator, num_classes, class_names, threshold=0.5, every_n=1):
            super().__init__()
            self.generator = generator; self.num_classes = num_classes
            self.class_names = class_names; self.threshold = threshold
            self.every_n = every_n

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.every_n != 0: return
            y_true_list, y_pred_list = [], []
            for i in range(len(self.generator)):
                bx, by = self.generator[i]
                preds = self.model.predict(bx, verbose=0)
                if isinstance(preds, list): preds = preds[0]
                gt = by if not isinstance(by, dict) else by["main_out"]
                y_true_list.append(gt); y_pred_list.append(preds)
            y_true = np.concatenate(y_true_list, axis=0)
            y_pred = (np.concatenate(y_pred_list, axis=0) > self.threshold).astype(np.float32)
            print(f"\n  +-- Per-Class Val IoU/Dice (Epoch {epoch+1}) ------+")
            print(f"  | {'Class':<5}  {'IoU':>6}  {'Dice':>6}  {'GT px':>8}  Status")
            print(f"  | {'-'*48}")
            for c in range(self.num_classes):
                name = self.class_names[c] if c < len(self.class_names) else f"C{c}"
                gt_c = y_true[..., c]; pr_c = y_pred[..., c]; gt_px = int(gt_c.sum())
                tp = float(np.sum(gt_c * pr_c)); fp = float(np.sum((1-gt_c)*pr_c))
                fn = float(np.sum(gt_c*(1-pr_c)))
                iou = tp/(tp+fp+fn) if (tp+fp+fn)>0 else float("nan")
                dice = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else float("nan")
                iou_s = f"{iou:.4f}" if not np.isnan(iou) else "  nan "
                dice_s = f"{dice:.4f}" if not np.isnan(dice) else "  nan "
                flag = "  << STARVING" if (np.isnan(iou) or iou < 0.05) else ""
                print(f"  | {name:<5}  {iou_s:>6}  {dice_s:>6}  {gt_px:>8,}{flag}")
            print(f"  +{'-'*52}+\n")

    # --- Callbacks ---
    callbacks = [
        SimpleLogger(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(results_dir / f"{model_prefix}_best.weights.h5"),
            monitor="val_loss", save_best_only=True,
            save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(results_dir / f"{model_prefix}_training_log.csv")
        ),
        PerClassMetricsCallback(
            generator=val_gen,
            num_classes=config.model.num_classes,
            class_names=CLASS_NAMES_RUN,
            threshold=0.5,
            every_n=1,
        ),
    ]

    # Deep supervision: wrap generators to duplicate Y for each output head
    if is_ds:
        out_names = list(model.output_names)

        class DSWrapper(tf.keras.utils.Sequence):
            """Wraps a Sequence to return dict targets for deep supervision."""
            def __init__(self, gen, names):
                self.gen = gen
                self.names = names
            def __len__(self):
                return len(self.gen)
            def __getitem__(self, idx):
                X, Y = self.gen[idx]
                return X, {n: Y for n in self.names}

        train_data = DSWrapper(train_gen, out_names)
        val_data = DSWrapper(val_gen, out_names)
        steps_per_epoch, validation_steps = None, None
    else:
        train_data, val_data = train_gen, val_gen
        steps_per_epoch, validation_steps = None, None

    # --- Train ---
    t0 = time.time()
    print(f"[*] Starting model.fit() at {time.ctime()}...", flush=True)
    try:
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return None, {}
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return None, {}

    train_time = time.time() - t0

    # --- Save ---
    model.save_weights(str(results_dir / f"{model_prefix}_final.weights.h5"))
    try:
        model.save(str(results_dir / f"{model_prefix}_final.keras"))
    except (ValueError, TypeError):
        try:
            model.save(str(results_dir / f"{model_prefix}_final.h5"), save_format="h5")
        except Exception:
            pass  # weights already saved

    logger.info(f"Training complete in {train_time/60:.1f} min.")
    return history, {"train_time_s": round(train_time, 1)}


# =============================================================================
# TEST PHASE — evaluate on IDRID test set
# =============================================================================

def _evaluate_test(model, config, results_dir: Path, model_name: str,
                   quick_test: bool = False) -> Optional[Dict]:
    """Run test evaluation on IDRID test set. Returns metrics dict or None."""
    import math
    from segmentation.experiments.run_iou_analysis import (
        compute_iou, compute_dice, compute_clinical_metrics,
    )

    test_gen = _build_test_generator(config, quick_test)
    if test_gen is None:
        print("[!] IDRID test set not found. Skipping test phase.")
        return None

    print(f"\n{'='*60}")
    print(f"  TEST PHASE — {model_name} on IDRID Test Set")
    print(f"{'='*60}")
    print(f"[*] Test batches: {len(test_gen)}")

    y_true_list, y_pred_list = [], []
    for batch_x, batch_y in test_gen:
        y_true_list.append(batch_y)
        preds = model.predict(batch_x, verbose=0)
        if isinstance(preds, list):
            preds = preds[0]  # main output for deep supervision
        y_pred_list.append(preds)

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    num_classes = config.model.num_classes
    test_iou      = compute_iou(y_true, y_pred, num_classes)
    test_dice     = compute_dice(y_true, y_pred, num_classes)
    test_clinical = compute_clinical_metrics(y_true, y_pred, num_classes)
    # compute_clinical_metrics returns keys: sens_class_N, spec_class_N, prec_class_N

    # --- Confusion Matrix ---
    _save_confusion_matrix(y_true, y_pred, num_classes, results_dir, model_name)

    # --- Prediction Visualisation ---
    _save_prediction_samples(test_gen, model, results_dir, model_name, num_samples=4)

    # --- NaN-safe helpers ---
    def _fmt(v):
        """Format float for printing; display 'nan' instead of crashing."""
        try:
            return f"{float(v):8.4f}" if not math.isnan(float(v)) else "     nan"
        except Exception:
            return "     err"

    def _safe_round(d: dict) -> dict:
        """Round floats for JSON; replace NaN with null (valid JSON)."""
        out = {}
        for k, v in d.items():
            try:
                out[k] = None if math.isnan(float(v)) else round(float(v), 4)
            except Exception:
                out[k] = v
        return out

    # Print results
    print(f"\n  {'Class':15s}  |  {'IoU':>8s}  |  {'Dice':>8s}  |  {'Sens':>8s}  |  {'Prec':>8s}")
    print(f"  {'-'*60}")
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        iou  = test_iou.get(f"iou_class_{i}",    float("nan"))
        dice = test_dice.get(f"dice_class_{i}",   float("nan"))
        # FIX: correct key names from compute_clinical_metrics are sens_class_N / prec_class_N
        sens = test_clinical.get(f"sens_class_{i}", float("nan"))
        prec = test_clinical.get(f"prec_class_{i}", float("nan"))
        print(f"  {name:15s}  |  {_fmt(iou)}  |  {_fmt(dice)}  |  {_fmt(sens)}  |  {_fmt(prec)}")
    print(f"\n  Mean IoU  : {_fmt(test_iou.get('mean_iou', float('nan'))).strip()}")
    print(f"  Mean Dice : {_fmt(test_dice.get('mean_dice', float('nan'))).strip()}")

    # Save — NaN values become JSON null (Python None)
    metrics = {
        "model": model_name,
        "iou":      _safe_round(test_iou),
        "dice":     _safe_round(test_dice),
        "clinical": _safe_round(test_clinical),
    }
    with open(results_dir / f"{model_name}_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _save_confusion_matrix(y_true, y_pred, num_classes, results_dir, model_name):
    """Generate and save per-class confusion matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        results_dir.mkdir(parents=True, exist_ok=True)

        y_pred_bin = (y_pred > 0.5).astype(np.float32)

        fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))
        if num_classes == 1:
            axes = [axes]

        for c in range(num_classes):
            gt = y_true[..., c].flatten().astype(int)
            pred = y_pred_bin[..., c].flatten().astype(int)

            # 2x2 confusion matrix
            from sklearn.metrics import confusion_matrix as sk_cm
            cm = sk_cm(gt, pred, labels=[0, 1])

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[c],
                        xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class {c}"
            axes[c].set_title(name)
            axes[c].set_xlabel("Predicted")
            axes[c].set_ylabel("Actual")

        plt.suptitle(f"Confusion Matrix — {model_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(results_dir / f"{model_name}_confusion_matrix.png", dpi=150)
        plt.close()
        logger.info(f"Confusion matrix saved -> {model_name}_confusion_matrix.png")
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix: {e}")


def _save_prediction_samples(test_gen, model, results_dir, model_name, num_samples=4):
    """Save side-by-side prediction visualizations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results_dir.mkdir(parents=True, exist_ok=True)

        X_sample, Y_sample = test_gen[0]
        preds = model.predict(X_sample, verbose=0)
        if isinstance(preds, list):
            preds = preds[0]

        n = min(num_samples, X_sample.shape[0])
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for i in range(n):
            # Original
            axes[i, 0].imshow(X_sample[i])
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")

            # Ground truth (overlay channels as RGB)
            gt_rgb = np.zeros((*Y_sample.shape[1:3], 3), dtype=np.float32)
            for c in range(min(3, Y_sample.shape[-1])):
                gt_rgb[..., c] = Y_sample[i, ..., c]
            axes[i, 1].imshow(gt_rgb)
            axes[i, 1].set_title("Ground Truth (R=MA, G=HE, B=EX)")
            axes[i, 1].axis("off")

            # Prediction
            pred_rgb = np.zeros((*preds.shape[1:3], 3), dtype=np.float32)
            for c in range(min(3, preds.shape[-1])):
                pred_rgb[..., c] = (preds[i, ..., c] > 0.5).astype(np.float32)
            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")

        plt.suptitle(f"Predictions — {model_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(results_dir / f"{model_name}_predictions.png", dpi=150)
        plt.close()
        logger.info(f"Predictions saved -> {model_name}_predictions.png")
    except Exception as e:
        logger.warning(f"Could not generate prediction samples: {e}")


# =============================================================================
# MAIN MODEL RUN
# =============================================================================

def run_main(quick_test: bool = False) -> bool:
    """Run the main Ghost-CAS-UNet v2 training + test."""
    print("\n" + "=" * 60)
    print("  SEGMENTATION — MAIN MODEL")
    print("=" * 60)

    setup_logging(log_dir=current_dir / "logs")
    set_seed(42)  # FIX #3: Reproducibility

    gpu_info = get_gpu_info()
    logger.info(f"Hardware: {gpu_info}")

    config = SegmentationConfig()
    if quick_test:
        config.training.epochs = 1
        config.data.img_size = (64, 64)

    print(f"  Model      : {config.model.name}")
    print(f"  Resolution : {config.data.img_size}")
    print(f"  Filters    : {config.model.encoder_filters}")
    print(f"  Epochs     : {config.training.epochs}")
    print("=" * 60 + "\n")

    data = _resolve_data(config)
    if data is None:
        print("[!] Data not available. Skipping training.")
        return True

    train_imgs, train_masks, val_imgs, val_masks, data_dir = data
    train_gen, val_gen = _build_generators(
        train_imgs, train_masks, val_imgs, val_masks, config, quick_test
    )
    print(f"[*] Train batches: {len(train_gen)},  Val batches: {len(val_gen)}")

    # Build model
    model_key = config.model.name.lower()
    if model_key not in SEGMENTATION_MODELS:
        logger.error(f"Unknown model: {model_key}. Available: {list(SEGMENTATION_MODELS.keys())}")
        return False

    model = SEGMENTATION_MODELS[model_key](
        input_shape=(*config.data.img_size, 3),
        num_classes=config.model.num_classes,
        encoder_filters=config.model.encoder_filters,
        dropout_rate=config.model.dropout_rate,
        ghost_ratio=config.model.ghost_ratio,
        use_skip_attention=config.model.use_skip_attention,
        use_aspp=config.model.use_aspp,
        deep_supervision=config.model.deep_supervision,
    )

    results_dir = current_dir / "results"
    history, info = _train_model(
        model, train_gen, val_gen, config, results_dir,
        quick_test=quick_test, model_prefix="main",
    )

    # --- TEST PHASE (FIX #1) ---
    if history is not None:
        _evaluate_test(model, config, results_dir, "Ghost_CAS_UNet_v2", quick_test)

    return history is not None


# =============================================================================
# ABLATION STUDY
# =============================================================================

class SegmentationAblationStudy(BaseAblationStudy):
    """Ablation study for Ghost-CAS-UNet architecture components."""

    def __init__(self, config: SegmentationConfig, quick_test: bool = False, **kwargs):
        results_dir = current_dir / "results" / "ablation"
        super().__init__(results_dir=results_dir, **kwargs)
        self.base_config = config
        self.quick_test = quick_test

    def get_parameters(self) -> List[AblationParameter]:
        return [
            AblationParameter(
                name="ghost_ratio",
                values=[2, 4],
                description="Ghost Module expansion ratio — higher = fewer params",
            ),
            AblationParameter(
                name="use_skip_attention",
                values=[True, False],
                description="Attention Gate on skip connections",
            ),
            AblationParameter(
                name="use_aspp",
                values=[True, False],
                description="DW-ASPP at bottleneck for multi-scale context",
            ),
            AblationParameter(
                name="resolution",
                values=[128, 256, 512],
                description="Input patch resolution — higher captures finer lesion details",
            ),
        ]

    def get_metrics(self) -> List[str]:
        return ["val_loss", "val_accuracy", "val_iou", "val_dice", "params", "train_time_s"]

    def run_single_experiment(self, exp_config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        import tensorflow as tf

        set_seed(seed)

        # Override config from ablation parameters
        cfg = SegmentationConfig()
        cfg.model.ghost_ratio = exp_config["ghost_ratio"]
        cfg.model.use_skip_attention = exp_config["use_skip_attention"]
        cfg.model.use_aspp = exp_config["use_aspp"]
        res = exp_config["resolution"]
        cfg.data.img_size = (res, res)

        if self.quick_test:
            cfg.training.epochs = 1
            cfg.training.patches_per_image = 2
            cfg.training.patches_per_image_val = 2

        # Data
        data = _resolve_data(cfg)
        if data is None:
            return {"error": "data_not_found"}
        train_imgs, train_masks, val_imgs, val_masks, _ = data
        train_gen, val_gen = _build_generators(
            train_imgs, train_masks, val_imgs, val_masks, cfg, self.quick_test
        )

        # Model — always Ghost-CAS-UNet v2 for ablation
        model = SEGMENTATION_MODELS["ghost_cas_unet_v2"](
            input_shape=(*cfg.data.img_size, 3),
            num_classes=cfg.model.num_classes,
            encoder_filters=cfg.model.encoder_filters,
            dropout_rate=cfg.model.dropout_rate,
            ghost_ratio=cfg.model.ghost_ratio,
            use_skip_attention=cfg.model.use_skip_attention,
            use_aspp=cfg.model.use_aspp,
            deep_supervision=cfg.model.deep_supervision,
        )

        exp_label = (
            f"gr{exp_config['ghost_ratio']}_"
            f"sa{int(exp_config['use_skip_attention'])}_"
            f"aspp{int(exp_config['use_aspp'])}_"
            f"r{res}_s{seed}"
        )
        exp_dir = self.results_dir / exp_label
        history, info = _train_model(
            model, train_gen, val_gen, cfg, exp_dir,
            quick_test=self.quick_test, model_prefix="ablation",
        )

        if history is None:
            return {"error": "training_failed"}

        # Collect final metrics
        h = history.history
        result = {
            "ghost_ratio": exp_config["ghost_ratio"],
            "use_skip_attention": exp_config["use_skip_attention"],
            "use_aspp": exp_config["use_aspp"],
            "resolution": res,
            "params": int(model.count_params()),
            "train_time_s": info.get("train_time_s", 0),
            "val_loss": float(h.get("val_loss", [float("nan")])[-1]),
        }

        # Extract metrics (handle deep supervision key prefixes)
        for key in h:
            if "val_" in key and ("accuracy" in key or "iou" in key or "dice" in key):
                short_key = key.replace("val_main_out_", "val_").replace("val_", "val_")
                result[short_key] = float(h[key][-1])

        return result


def run_ablation(quick_test: bool = False) -> bool:
    """Run the full ablation study."""
    print("\n" + "=" * 60)
    print("  SEGMENTATION — ABLATION STUDY")
    print("=" * 60)

    setup_logging(log_dir=current_dir / "logs")

    config = SegmentationConfig()
    study = SegmentationAblationStudy(config, quick_test=quick_test)
    df = study.run_full_ablation(quick_test=quick_test)
    summary = study.compute_summary(df)

    print("\n[*] Ablation Summary:")
    print(summary.to_string(index=False))
    print(f"\n[*] Results saved to: {study.results_dir}")
    return True


# =============================================================================
# BASELINES
# =============================================================================

def run_baselines(quick_test: bool = False) -> bool:
    """Train each baseline model with the same data pipeline + test evaluation."""
    print("\n" + "=" * 60)
    print("  SEGMENTATION — BASELINE COMPARISON")
    print("=" * 60)

    setup_logging(log_dir=current_dir / "logs")
    set_seed(42)

    try:
        from baselines import BASELINE_MODELS
    except ImportError as e:
        logger.error(f"Cannot load baselines: {e}")
        print("[!] Install segmentation-models:  pip install segmentation-models")
        return False

    config = SegmentationConfig()
    if quick_test:
        config.training.epochs = 1
        config.data.img_size = (64, 64)

    data = _resolve_data(config)
    if data is None:
        print("[!] Data not available.")
        return True

    train_imgs, train_masks, val_imgs, val_masks, _ = data
    train_gen, val_gen = _build_generators(
        train_imgs, train_masks, val_imgs, val_masks, config, quick_test
    )

    all_results = []

    for name, model_fn in BASELINE_MODELS.items():
        print(f"\n{'─'*50}")
        print(f"  Baseline: {name}")
        print(f"{'─'*50}")

        set_seed(42)  # Same seed for each baseline

        model = model_fn(
            input_shape=(*config.data.img_size, 3),
            num_classes=config.model.num_classes,
        )

        # FIX: Rebuild generators for EACH baseline — generators have mutable shuffled
        # index state (on_epoch_end). Sharing one pair across baselines corrupts sampling.
        train_gen_b, val_gen_b = _build_generators(
            train_imgs, train_masks, val_imgs, val_masks, config, quick_test
        )

        baseline_dir = current_dir / "results" / "baselines" / name
        history, info = _train_model(
            model, train_gen_b, val_gen_b, config, baseline_dir,
            quick_test=quick_test, model_prefix=name,
        )

        result = {"model": name, "params": int(model.count_params())}
        if history is not None:
            h = history.history
            result["val_loss"] = float(h.get("val_loss", [float("nan")])[-1])
            for key in h:
                if "val_" in key and ("accuracy" in key or "iou" in key or "dice" in key):
                    result[key] = float(h[key][-1])

            # --- TEST PHASE for baseline ---
            test_metrics = _evaluate_test(
                model, config, baseline_dir, name, quick_test
            )
            if test_metrics:
                result["test"] = test_metrics

        result.update(info)
        all_results.append(result)

    # Save combined results
    baselines_dir = current_dir / "results" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    with open(baselines_dir / "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None if isinstance(x, float) and __import__('math').isnan(x) else x)

    print(f"\n[*] Baseline results saved to: {baselines_dir}")
    return True


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Runner")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (1 epoch)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--baselines", action="store_true", help="Run baseline comparison")
    args = parser.parse_args()

    try:
        if args.ablation:
            success = run_ablation(quick_test=args.quick)
        elif args.baselines:
            success = run_baselines(quick_test=args.quick)
        else:
            success = run_main(quick_test=args.quick)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Runner failed: {e}", exc_info=True)
        sys.exit(1)
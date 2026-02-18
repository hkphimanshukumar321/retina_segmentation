# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Universal Training Script
=========================

Trains any model in the registry with configurable parameters.
Used for:
  1. Baselines (UNet, MobileUNet)
  2. Ablation Studies (Ghost-CA, Ghost-CAS-v2 with different settings)
  3. Final Model Training

Usage:
    python -m segmentation.train --model unet --name baseline_unet
    python -m segmentation.train --model ghost_cas_unet_v2 --prob_lesion 0.7
"""

import sys
import argparse
import json
import time
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()           # segmentation/
ROOT_DIR   = SCRIPT_DIR.parent.resolve()               # retina_scan/

sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from common.logger import setup_logging
from common.data_loader import (
    PatchDataGenerator, AugmentationPipeline,
    RandomFlip, RandomRotation, RandomBrightness,
    RandomContrast, RandomElasticDeform
)
from segmentation.src.models import SEGMENTATION_MODELS
from segmentation.src.losses import combined_loss_v2
from segmentation.experiments.run_iou_analysis import (
    compute_iou, compute_dice, compute_clinical_metrics, compute_positive_only_iou
)
from segmentation.config import SegmentationConfig

# Logger setup
logger = logging.getLogger("train")


class ExperimentConfig:
    """Dynamic config based on CLI args."""
    def __init__(self, args):
        self.data = self.DataConfig(args)
        self.model = self.ModelConfig(args)
        
    class DataConfig:
        def __init__(self, args):
            self.img_size = tuple(args.resolution)
            self.label_ids = (127, 63, 255)  # Refined IDRiD
            self.bit_values = None
            self.prob_lesion = args.prob_lesion
            
    class ModelConfig:
        def __init__(self, args):
            self.num_classes = 3

def set_seed(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Universal Training Script")
    
    # Experiment settings
    parser.add_argument("--model", type=str, default="ghost_cas_unet_v2", 
                        choices=list(SEGMENTATION_MODELS.keys()), help="Model architecture")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patches_train", type=int, default=100, help="Patches per image (train)")
    parser.add_argument("--patches_val", type=int, default=20, help="Patches per image (val)")
    
    # Data params
    parser.add_argument("--resolution", type=int, nargs=2, default=[256, 256], help="Image resolution (H W)")
    parser.add_argument("--prob_lesion", type=float, default=0.5, help="Lesion sampling probability")
    
    # Model params
    parser.add_argument("--filters", type=int, nargs="+", default=[32, 64, 128, 256], help="Encoder filters")
    parser.add_argument("--ghost_ratio", type=int, default=2, help="Ghost module ratio")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--no_aspp", action="store_true", help="Disable ASPP")
    parser.add_argument("--no_ds", action="store_true", help="Disable deep supervision")
    
    args = parser.parse_args()
    
    # Determine experiment name
    if args.name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.name = f"{args.model}_{timestamp}"
        
    # Setup directories
    output_dir = SCRIPT_DIR / "results" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging
    setup_logging(log_dir=output_dir)
    logger.info(f"Starting Experiment: {args.name}")
    logger.info(f"Args: {vars(args)}")
    
    set_seed(args.seed)
    
    # Select GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
            logger.info(f"Using GPU: {gpus[args.gpu]}")
        except RuntimeError as e:
            logger.error(e)
            
    # ---- Data Loading ----
    data_dir = ROOT_DIR / "data" / "segmentation"
    img_dir  = "Images" if (data_dir / "Images").exists() else "images"
    mask_dir = "Labels" if (data_dir / "Labels").exists() else "masks"

    image_files = sorted(list((data_dir / img_dir).glob("*.[jp][pn][g]")))
    mask_files  = sorted(list((data_dir / mask_dir).glob("*.[jp][pn][g]")))
    
    assert len(image_files) == len(mask_files), "Image/mask count mismatch"
    
    # Split
    split_idx = int(len(image_files) * 0.8)
    train_imgs = image_files[:split_idx]
    val_imgs = image_files[split_idx:]
    train_masks = mask_files[:split_idx]
    val_masks = mask_files[split_idx:]
    
    # Config wrapper for generator
    config = ExperimentConfig(args)
    
    # Augmentations
    train_augs = AugmentationPipeline([
        RandomFlip(horizontal=True, vertical=True),
        RandomRotation(max_angle=180),
        RandomBrightness(delta=0.1),
        RandomContrast(limit=0.1),
        RandomElasticDeform(alpha=50, sigma=5, p=0.3),
    ])
    
    train_gen = PatchDataGenerator(
        train_imgs, train_masks, config,
        patches_per_image=args.patches_train,
        batch_size=args.batch_size,
        augmentitons=train_augs
    )
    val_gen = PatchDataGenerator(
        val_imgs, val_masks, config,
        patches_per_image=args.patches_val,
        batch_size=args.batch_size,
        augmentitons=None
    )
    
    logger.info(f"Train samples: {len(train_gen) * args.batch_size}, Val samples: {len(val_gen) * args.batch_size}")
    
    # ---- Model Building ----
    model_fn = SEGMENTATION_MODELS[args.model]
    
    # Build kwargs dynamically based on model signature
    # (Simplified: assume all take similar args or kwargs ignore extras)
    # Actually models have different signatures. We need to be careful.
    
    # Common args
    model_kwargs = {
        'input_shape': (*args.resolution, 3),
        'num_classes': 3,
        'encoder_filters': args.filters,
        'dropout_rate': args.dropout,
    }
    
    # Ghost-specific args
    if "ghost" in args.model:
        model_kwargs['ghost_ratio'] = args.ghost_ratio
        
    # v2 specific args
    if args.model == "ghost_cas_unet_v2":
        model_kwargs['use_skip_attention'] = True
        model_kwargs['use_aspp'] = not args.no_aspp
        model_kwargs['deep_supervision'] = not args.no_ds
    elif args.model == "ghost_ca_unet":
        model_kwargs['use_skip_attention'] = True # v1 had skip attention? yes, typically
        
    logger.info(f"Building model: {args.model} with kwargs: {model_kwargs}")
    model = model_fn(**model_kwargs)
    
    model.summary()
    total_params = model.count_params()
    logger.info(f"Total Params: {total_params:,}")
    
    # ---- Compilation ----
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
    
    # Loss selection
    # Using v2 loss for everything for fair comparison? 
    # Or should baselines use standard losses?
    # User wants to overhaul Ghost-CAS-UNet. Benchmarking against baselines usually implies SAME loss.
    loss_fn = combined_loss_v2(
        w_lovasz=0.5, w_focal_tversky=0.3, w_bce=0.2
    )
    
    deep_sup = (args.model == "ghost_cas_unet_v2" and not args.no_ds)
    
    if deep_sup:
        loss_dict = {'main_out': loss_fn, 'aux_up_1': loss_fn, 'aux_up_2': loss_fn}
        loss_weights = {'main_out': 1.0, 'aux_up_1': 0.4, 'aux_up_2': 0.2}
        model.compile(optimizer=optimizer, loss=loss_dict, loss_weights=loss_weights, metrics={'main_out': ['accuracy']})
    else:
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        
    # ---- Data Wrappers (for Deep Supervision) ----
    if deep_sup:
        def _wrap_gen(gen):
            for X, Y in gen:
                yield X, {'main_out': Y, 'aux_up_1': Y, 'aux_up_2': Y}
        
        # We need tf.data.Dataset for proper signature handling or just pass generator
        # Passing generator directly to fit works in recent TF but signature inference can be tricky.
        # Let's use the explicit Dataset method from pilot_test to be safe
        
        output_signature = (
            tf.TensorSpec(shape=(None, *args.resolution, 3), dtype=tf.float32),
            {
                'main_out': tf.TensorSpec(shape=(None, *args.resolution, 3), dtype=tf.float32),
                'aux_up_1': tf.TensorSpec(shape=(None, *args.resolution, 3), dtype=tf.float32),
                'aux_up_2': tf.TensorSpec(shape=(None, *args.resolution, 3), dtype=tf.float32),
            }
        )
        
        train_data = tf.data.Dataset.from_generator(lambda: _wrap_gen(train_gen), output_signature=output_signature)
        val_data = tf.data.Dataset.from_generator(lambda: _wrap_gen(val_gen), output_signature=output_signature)
    else:
        train_data = train_gen
        val_data = val_gen

    # ---- Training ----
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.weights.h5"),
            monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        callbacks=callbacks
    )
    
    # ---- Save & Evaluate ----
    model.save_weights(str(output_dir / "final.weights.h5"))
    try:
        model.save(str(output_dir / "final_model.keras"))
    except:
        pass
        
    # Evaluation
    logger.info("Evaluating...")
    y_true_list, y_pred_list = [], []
    batch_idx = 0
    for batch_x, batch_y in val_gen:
        y_true_list.append(batch_y)
        try:
            preds = model.predict(batch_x, verbose=0)
        except Exception as e:
            logger.error(f"Prediction failed at batch {batch_idx}: {e}")
            raise e
            
        if batch_idx == 0:
            logger.info(f"Preds type: {type(preds)}")
            
        if isinstance(preds, list) or (deep_sup and isinstance(preds, dict)): 
            if isinstance(preds, dict): preds = preds['main_out']
            elif isinstance(preds, list): preds = preds[0]
            
        y_pred_list.append(preds)
        batch_idx += 1
        
    logger.info(f"Collected {len(y_true_list)} batches.")
    if len(y_true_list) == 0:
        logger.warning("No validation data collected!")
        
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    logger.info(f"Y_true shape: {y_true.shape}, Y_pred shape: {y_pred.shape}")
    
    logger.info("Computing IoU...")
    iou = compute_iou(y_true, y_pred, 3)
    logger.info("Computing Dice...")
    dice = compute_dice(y_true, y_pred, 3)
    
    results = {
        "args": vars(args),
        "iou": {k: float(v) for k,v in iou.items()},
        "dice": {k: float(v) for k,v in dice.items()},
        "params": int(total_params)
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Mean IoU: {iou['mean_iou']:.4f}")
    logger.info(f"Experiment verified successfully.")
    
if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        logger.error(f"Experiment Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

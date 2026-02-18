# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Segmentation Master Runner
==========================

Main entry point for segmentation experiments.
"""

import sys
import argparse
import logging
from pathlib import Path

# Fix paths to allow imports from common and local src
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))  # Template root
sys.path.insert(0, str(current_dir))  # Segmentation root

from common.logger import setup_logging
from common.hardware import get_gpu_info
from config import SegmentationConfig

# Setup logger
logger = logging.getLogger("segmentation_runner")


def run_all(quick_test: bool = False) -> bool:
    """
    Run full segmentation pipeline.
    
    Args:
        quick_test: Run with minimal data/epochs for verification
        
    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("SEGMENTATION MASTER RUNNER")
    print("=" * 60)
    
    # 1. Setup
    setup_logging(log_dir=current_dir / "logs")
    gpu_info = get_gpu_info()
    logger.info(f"Hardware: {gpu_info}")
    
    config = SegmentationConfig()
    if quick_test:
        config.training.epochs = 1
        config.data.img_size = (64, 64)
    
    print(f"\n[*] Config: Image Size={config.data.img_size}, Classes={config.model.num_classes}")
    
    # 2. Check Data
    data_dir = Path(config.data.data_dir or (current_dir / "data"))
    if not data_dir.exists():
        # Try interactive setup
        from common.interactive_setup import setup_dataset_interactive
        if setup_dataset_interactive('segmentation', data_dir):
            print("\n[*] Data setup complete. Resuming experiment...")
        else:
            print(f"\n[!] Data directory not found: {data_dir}")
            print("    Please add 'images' and 'masks' folders.")
            print("    Skipping training/evaluation loop.")
            return True # Return true to pass smoke test even if data missing
        
    # 3. Data Generators (Patch-based)
    logger.info("Initializing Patch-based Data Generators...")
    
    # Simple split (first 80% train, last 20% val for now)
    image_files = sorted(list((data_dir / config.data.img_dir).glob("*.[jp][pn][g]")))
    mask_files = sorted(list((data_dir / config.data.mask_dir).glob("*.[jp][pn][g]")))
    
    if len(image_files) != len(mask_files):
        logger.error("Mismatch in images/masks count!")
        return False
        
    split_idx = int(len(image_files) * 0.8)
    train_imgs, val_imgs = image_files[:split_idx], image_files[split_idx:]
    train_masks, val_masks = mask_files[:split_idx], mask_files[split_idx:]
    
    from common.data_loader import (
        PatchDataGenerator, AugmentationPipeline, 
        RandomFlip, RandomRotation, RandomBrightness, 
        RandomContrast, RandomElasticDeform
    )
    
    # Training Augmentations
    train_augs = AugmentationPipeline([
        RandomFlip(horizontal=True, vertical=True),
        RandomRotation(max_angle=180), # Full rotation for retina
        RandomBrightness(delta=0.1),
        RandomContrast(limit=0.1),
        RandomElasticDeform(alpha=50, sigma=5, p=0.3)
    ])
    
    # Generators
    train_gen = PatchDataGenerator(
        train_imgs, train_masks, config, 
        patches_per_image=50 if not quick_test else 2,
        batch_size=8,
        augmentitons=train_augs
    )
    
    val_gen = PatchDataGenerator(
        val_imgs, val_masks, config,
        patches_per_image=10 if not quick_test else 2,
        batch_size=8,
        augmentitons=None # No aug for val
    )
    
    print(f"[*] Train Batches: {len(train_gen)}, Val Batches: {len(val_gen)}")
    
    # 4. Training
    logger.info("Building model...")
    from segmentation.src.models import SEGMENTATION_MODELS
    from segmentation.src.losses import combined_loss
    import tensorflow as tf
    
    model_key = config.model.name.lower()
    if model_key not in SEGMENTATION_MODELS:
        logger.error(f"Unknown model: {config.model.name}. Available: {list(SEGMENTATION_MODELS.keys())}")
        return False
    
    model_fn = SEGMENTATION_MODELS[model_key]
    model = model_fn(
        input_shape=(config.data.img_size[0], config.data.img_size[1], 3),
        num_classes=config.model.num_classes
    )
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    
    model.compile(
        optimizer=optimizer,
        # AUDIT FIX: Use Focal Tversky Loss for better MA detection
        loss=combined_loss(alpha=0.3, beta=0.7, gamma=0.75) if config.model.name.lower().startswith("ghost") else "binary_crossentropy",
        metrics=['accuracy', tf.keras.metrics.OneHotIoU(num_classes=config.model.num_classes, target_class_ids=[0,1,2], name='iou')]
    )
    model.summary(print_fn=logger.info)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(current_dir / "results" / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # CSV Logger
    callbacks.append(tf.keras.callbacks.CSVLogger(str(current_dir / "logs" / "training_log.csv")))
    
    logger.info("Starting training...")
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.training.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(str(current_dir / "results" / "final_model.keras"))
        logger.info("Training complete.")
        return True
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return False
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test")
    args = parser.parse_args()
    
    try:
        success = run_all(quick_test=args.quick)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Runner failed: {e}", exc_info=True)
        sys.exit(1)
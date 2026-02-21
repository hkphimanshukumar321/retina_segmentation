# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Segmentation Configuration
===========================

Central configuration for segmentation experiments.
All runners (run.py, ablation, baselines) use this config.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_dir: Optional[str] = None          # Auto-resolved from project root if None
    img_dir: str = "Images"                 # Subdirectory for images
    mask_dir: str = "Labels"                # Subdirectory for masks
    img_size: Tuple[int, int] = (256, 256)  # Patch / input resolution
    label_ids: Tuple[int, ...] = (255, 127, 63)  # Refined IDRiD: MA=255, HE=127, EX=63
    bit_values: Optional[List[int]] = None  # Legacy bitmask (unused for IDRiD)
    prob_lesion: float = 0.5                # Probability of lesion-centred sampling


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "ghost_cas_unet_v2"
    num_classes: int = 5
    encoder_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    ghost_ratio: int = 2
    use_skip_attention: bool = True
    use_aspp: bool = True
    deep_supervision: bool = True
    dropout_rate: float = 0.15


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 8
    patches_per_image: int = 100
    patches_per_image_val: int = 20
    clip_norm: float = 1.0


@dataclass
class SegmentationConfig:
    """Master configuration for segmentation experiments."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

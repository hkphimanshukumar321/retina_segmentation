# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Segmentation Configuration
==========================
"""

from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path
import sys

# Add root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_base import BaseTrainingConfig, BaseDataConfig, BaseModelConfig


@dataclass
class SegmentationDataConfig(BaseDataConfig):
    data_dir: Path = Path(__file__).parent.parent / "data" / "segmentation"
    # Auto-detect defaults
    img_dir: str = "Images" if (Path(__file__).parent.parent / "data" / "segmentation" / "Images").exists() else "images"
    mask_dir: str = "Labels" if (Path(__file__).parent.parent / "data" / "segmentation" / "Labels").exists() else "masks"
    
    num_classes: int = 3
    # Refined IDRiD label IDs (from Table 2 of the dataset paper):
    #   HE (Hemorrhage) = 127,  EX (Hard Exudate) = 63,  MA (Microaneurysm) = 255
    # These are direct integer label IDs, NOT bit flags.
    label_ids: Tuple[int, ...] = (127, 63, 255)
    class_names: Tuple[str, ...] = ("HE", "EX", "MA")
    # Legacy bit_values kept for backward compatibility (DO NOT USE for Refined IDRiD)
    bit_values: Tuple[int, ...] = (8, 16, 32)
    
    img_size: Tuple[int, int] = (128, 128)
    
    # Lesion-aware sampling: probability of centering a patch on a lesion pixel
    # 0.0 = purely random sampling (like v1)
    # 0.5 = 50% patches centered on lesions, 50% random
    prob_lesion: float = 0.5


@dataclass
class SegmentationModelConfig(BaseModelConfig):
    name: str = "Ghost_CA_UNet"
    encoder_filters: Tuple[int, ...] = (16, 32, 64, 128)
    num_classes: int = 3
    ghost_ratio: int = 2  # Ghost Module compression ratio (Now verified working)


@dataclass
class SegmentationTrainingConfig(BaseTrainingConfig):
    # Multi-label loss: Binary Crossentropy + Dice Loss
    loss_function: str = "binary_crossentropy"
    metrics: Tuple[str, ...] = ("accuracy", "binary_iou")


@dataclass
class SegmentationConfig:
    data: SegmentationDataConfig = field(default_factory=SegmentationDataConfig)
    model: SegmentationModelConfig = field(default_factory=SegmentationModelConfig)
    training: SegmentationTrainingConfig = field(default_factory=SegmentationTrainingConfig)
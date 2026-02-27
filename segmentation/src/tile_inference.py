# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# ==============================================================================

"""
Tile-Based Inference at Full Resolution
==========================================

Rank-1 IDRiD strategy: inference at 1024×1024 tiles on full-resolution
IDRID images (4288×2848). Tiles overlap by 256px, predictions are merged
using Gaussian-weighted blending to eliminate seam artifacts.

Usage::

    from segmentation.src.tile_inference import tile_predict
    pred = tile_predict(model, img_path, tile_size=1024, overlap=256)
"""

import numpy as np
import cv2
from pathlib import Path


def _gaussian_weight_map(h: int, w: int, sigma_frac: float = 0.25) -> np.ndarray:
    """Create 2D Gaussian weight map — centre weighted, edges fade to 0.

    Used to blend overlapping tile predictions without hard seam artefacts.
    """
    cy, cx = h / 2, w / 2
    sy, sx = h * sigma_frac, w * sigma_frac
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-((X - cx)**2 / (2 * sx**2) + (Y - cy)**2 / (2 * sy**2)))
    return g


def tile_predict(
    model,
    image: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 256,
    batch_size: int = 4,
    num_classes: int = 5,
    use_tta: bool = False,
) -> np.ndarray:
    """Sliding-window full-resolution prediction with Gaussian blending.

    Args:
        model:       Compiled Keras model (input can be any HxW multiple of 32).
        image:       (H, W, 3) float32 [0,1] preprocessed full-res image.
        tile_size:   Tile side length (default 1024).
        overlap:     Overlap between adjacent tiles (default 256).
        batch_size:  Batch size for model.predict.
        num_classes: Number of output channels.
        use_tta:     Apply TTA on each tile (8× averaging).

    Returns:
        (H, W, num_classes) float32 prediction map.
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap
    weight = _gaussian_weight_map(tile_size, tile_size)

    # Output accumulator
    pred_sum    = np.zeros((h, w, num_classes), dtype=np.float64)
    weight_sum  = np.zeros((h, w, 1), dtype=np.float64)

    # Collect tile positions
    tiles = []
    positions = []
    for y in range(0, max(1, h - tile_size + 1), stride):
        for x in range(0, max(1, w - tile_size + 1), stride):
            # Clamp to image boundary
            y = min(y, h - tile_size)
            x = min(x, w - tile_size)
            tile = image[y:y+tile_size, x:x+tile_size]

            # Resize if model needs different input size
            tiles.append(tile)
            positions.append((y, x))

    # Also handle right/bottom edges
    if h > tile_size:
        y_last = h - tile_size
    else:
        y_last = 0
    if w > tile_size:
        x_last = w - tile_size
    else:
        x_last = 0

    # Deduplicate positions
    pos_set = set(positions)
    for y in [y_last]:
        for x in [x_last]:
            if (y, x) not in pos_set:
                tiles.append(image[y:y+tile_size, x:x+tile_size])
                positions.append((y, x))

    # Predict in batches
    if use_tta:
        try:
            from segmentation.src.tta import tta_predict
        except ImportError:
            from src.tta import tta_predict

    for i in range(0, len(tiles), batch_size):
        batch = np.array(tiles[i:i+batch_size], dtype=np.float32)

        if use_tta:
            preds = tta_predict(model, batch, batch_size=batch_size)
        else:
            preds = model.predict(batch, verbose=0, batch_size=batch_size)
            if isinstance(preds, list):
                preds = preds[0]

        for j, (y, x) in enumerate(positions[i:i+batch_size]):
            pred_tile = preds[j]  # (tile_size, tile_size, C)
            w_tile = weight[..., np.newaxis]  # (tile_size, tile_size, 1)

            pred_sum[y:y+tile_size,   x:x+tile_size] += pred_tile * w_tile
            weight_sum[y:y+tile_size, x:x+tile_size] += w_tile

    # Normalise by accumulated weights
    weight_sum = np.maximum(weight_sum, 1e-8)
    result = (pred_sum / weight_sum).astype(np.float32)

    return result


def predict_full_image(
    model,
    image_path,
    tile_size: int = 1024,
    overlap: int = 256,
    num_classes: int = 5,
    use_tta: bool = True,
    preprocess: bool = True,
) -> np.ndarray:
    """Full pipeline: load image → preprocess → tile predict → return mask.

    Args:
        model:       Compiled Keras model.
        image_path:  Path to IDRID image (.jpg).
        tile_size:   Tile size for sliding window.
        overlap:     Overlap between tiles.
        num_classes: Number of output classes.
        use_tta:     Apply TTA on each tile.
        preprocess:  Apply Ben Graham + CLAHE preprocessing.

    Returns:
        (H, W, num_classes) float32 prediction in [0, 1].
    """
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    if preprocess:
        try:
            from segmentation.src.retinal_preprocessing import retinal_preprocess
        except ImportError:
            from src.retinal_preprocessing import retinal_preprocess
        img_f = retinal_preprocess(img)
    else:
        img_f = img.astype(np.float32) / 255.0

    return tile_predict(
        model, img_f,
        tile_size=tile_size,
        overlap=overlap,
        num_classes=num_classes,
        use_tta=use_tta,
    )

# -*- coding: utf-8 -*-
"""
MA Overfit Sanity Check — Can the model memorize 1 image?
==========================================================

This is the classic ML debugging experiment:
  - Train on 1 image + 1 MA mask (binary: MA vs background)
  - Repeat the same sample every batch (no augmentation, no noise)
  - If the model overfits (Dice → 1.0 within 50 epochs) → architecture is FINE
  - If it can't overfit → something is fundamentally broken

Usage:
    python segmentation/scripts/overfit_ma_test.py                # default: IDRiD_01
    python segmentation/scripts/overfit_ma_test.py --image IDRiD_24  # specific image
    python segmentation/scripts/overfit_ma_test.py --epochs 100      # more epochs

Expected result: MA Dice should reach > 0.9 within 30-50 epochs.
"""

import sys
import os
import argparse
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2

ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "segmentation"))

import tensorflow as tf

# ── Paths ──────────────────────────────────────────────────────────────────────
IDRID      = ROOT / "data" / "IDRID" / "A. Segmentation"
TRAIN_IMGS = IDRID / "1. Original Images"               / "a. Training Set"
TRAIN_MASK = IDRID / "2. All Segmentation Groundtruths"  / "a. Training Set"
RESULTS    = ROOT / "segmentation" / "results" / "overfit_ma"

# ── Model ──────────────────────────────────────────────────────────────────────

def build_mini_unet(input_shape=(256, 256, 3), num_classes=1):
    """Tiny U-Net for single-class overfit test. ~200K params.
    
    Intentionally small to prove even a tiny model can overfit 1 sample.
    If this works → Ghost-CAS-UNet will also work → problem is data, not model.
    """
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    c1 = layers.Conv2D(32, 3, padding="same", activation="relu")(c1)
    p1 = layers.MaxPool2D(2)(c1)

    c2 = layers.Conv2D(64, 3, padding="same", activation="relu")(p1)
    c2 = layers.Conv2D(64, 3, padding="same", activation="relu")(c2)
    p2 = layers.MaxPool2D(2)(c2)

    # Bottleneck
    bn = layers.Conv2D(128, 3, padding="same", activation="relu")(p2)
    bn = layers.Conv2D(128, 3, padding="same", activation="relu")(bn)

    # Decoder
    u2 = layers.UpSampling2D(2)(bn)
    u2 = layers.Concatenate()([u2, c2])
    d2 = layers.Conv2D(64, 3, padding="same", activation="relu")(u2)
    d2 = layers.Conv2D(64, 3, padding="same", activation="relu")(d2)

    u1 = layers.UpSampling2D(2)(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = layers.Conv2D(32, 3, padding="same", activation="relu")(u1)
    d1 = layers.Conv2D(32, 3, padding="same", activation="relu")(d1)

    out = layers.Conv2D(num_classes, 1, activation="sigmoid")(d1)

    return Model(inputs, out, name="mini_unet_overfit")


def build_ghost_cas_unet(input_shape=(256, 256, 3), num_classes=1):
    """Our actual Ghost-CAS-UNet v2, BUT with num_classes=1 (MA only)."""
    from src.models import SEGMENTATION_MODELS
    model = SEGMENTATION_MODELS["ghost_cas_unet_v2"](
        input_shape=input_shape,
        num_classes=num_classes,
        encoder_filters=[32, 64, 128, 256],
        dropout_rate=0.0,       # No dropout for overfit test
        ghost_ratio=2,
        use_skip_attention=True,
        use_aspp=True,
        deep_supervision=False,  # Single output for simplicity
    )
    return model


# ── Data ───────────────────────────────────────────────────────────────────────

def load_single_sample(image_stem, patch_size=256):
    """Load one image + one MA mask → multiple overlapping patches.
    
    Returns arrays of shape (N, H, W, 3) and (N, H, W, 1).
    Uses a sliding window to extract many patches from the same image,
    specifically centred on MA lesion locations.
    """
    img_path = TRAIN_IMGS / f"{image_stem}.jpg"
    ma_path  = TRAIN_MASK / "1. Microaneurysms" / f"{image_stem}_MA.tif"

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not ma_path.exists():
        raise FileNotFoundError(f"MA mask not found: {ma_path}")

    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    ma  = cv2.imread(str(ma_path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    if ma.shape != (h, w):
        ma = cv2.resize(ma, (w, h), interpolation=cv2.INTER_NEAREST)

    ma_bin = (ma > 0).astype(np.uint8)
    total_ma = int(ma_bin.sum())
    print(f"\n  Image: {image_stem}")
    print(f"  Size: {w}×{h}")
    print(f"  MA pixels: {total_ma:,}")

    # Extract patches centred on MA lesions
    patches_img, patches_mask = [], []
    ys, xs = np.where(ma_bin > 0)

    if len(ys) == 0:
        raise ValueError(f"Image {image_stem} has no MA pixels! Pick a different image.")

    # Take patches at every MA pixel cluster centroid
    # Plus random offsets around each to get diversity
    n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ma_bin, connectivity=8
    )

    for i in range(1, n_components):  # skip background=0
        cy, cx = int(centroids[i][1]), int(centroids[i][0])

        # Multiple patches around each lesion (slight offsets for variety)
        for dy in [-32, 0, 32]:
            for dx in [-32, 0, 32]:
                y = max(0, min(h - patch_size, cy - patch_size // 2 + dy))
                x = max(0, min(w - patch_size, cx - patch_size // 2 + dx))

                p_img = img[y:y+patch_size, x:x+patch_size]
                p_ma  = ma_bin[y:y+patch_size, x:x+patch_size]

                if p_ma.sum() > 0:  # only keep patches that have MA
                    patches_img.append(p_img)
                    patches_mask.append(p_ma)

    # Also add the SAME centre patch duplicated (for very small images)
    if len(patches_img) < 8:
        cy, cx = int(np.median(ys)), int(np.median(xs))
        y = max(0, min(h - patch_size, cy - patch_size // 2))
        x = max(0, min(w - patch_size, cx - patch_size // 2))
        p_img = img[y:y+patch_size, x:x+patch_size]
        p_ma  = ma_bin[y:y+patch_size, x:x+patch_size]
        for _ in range(8):
            patches_img.append(p_img)
            patches_mask.append(p_ma)

    X = np.array(patches_img, dtype=np.float32) / 255.0
    Y = np.array(patches_mask, dtype=np.float32)[..., np.newaxis]  # (N, H, W, 1)

    print(f"  Patches extracted: {len(X)}")
    print(f"  MA pixels per patch (avg): {Y.sum() / len(Y):.1f}")
    print(f"  MA fraction per patch: {Y.mean() * 100:.3f}%")

    return X, Y


# ── Training ───────────────────────────────────────────────────────────────────

def run_overfit_test(image_stem, epochs, use_ghost, patch_size):
    """Train on 1 image, evaluate on the SAME image. Must overfit."""

    RESULTS.mkdir(parents=True, exist_ok=True)

    X, Y = load_single_sample(image_stem, patch_size)

    # Use both mini-UNet and Ghost-CAS-UNet
    if use_ghost:
        print("\n  Building Ghost-CAS-UNet v2 (num_classes=1, MA only)...")
        model = build_ghost_cas_unet((patch_size, patch_size, 3), num_classes=1)
    else:
        print("\n  Building Mini-UNet (overfit test)...")
        model = build_mini_unet((patch_size, patch_size, 3), num_classes=1)

    params = model.count_params()
    print(f"  Parameters: {params:,}")

    # BCE fails for MA (0.15% pixel area → model predicts all zeros, Dice=0)
    # Use Focal Tversky which penalises false negatives heavily
    from src.losses import focal_tversky_loss
    ft_loss = focal_tversky_loss(alpha=0.3, beta=0.7, gamma=0.75)

    # Combine with Dice loss for direct overlap optimisation
    def dice_bce_loss(y_true, y_pred):
        # Dice component
        smooth = 1e-6
        y_t = tf.reshape(y_true, [-1])
        y_p = tf.reshape(y_pred, [-1])
        inter = tf.reduce_sum(y_t * y_p)
        dice = 1.0 - (2.0 * inter + smooth) / (tf.reduce_sum(y_t) + tf.reduce_sum(y_p) + smooth)
        # Focal Tversky component
        ft = ft_loss(y_true, y_pred)
        return dice + ft

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=dice_bce_loss,
        metrics=["accuracy"],
    )

    # Custom callback to compute MA Dice every epoch
    class DiceLogger(tf.keras.callbacks.Callback):
        def __init__(self, X, Y):
            super().__init__()
            self.X = X; self.Y = Y
            self.best_dice = 0.0
            self.dice_history = []

        def on_epoch_end(self, epoch, logs=None):
            preds = self.model.predict(self.X, verbose=0)
            preds_bin = (preds > 0.5).astype(np.float32)

            tp = float(np.sum(self.Y * preds_bin))
            fp = float(np.sum((1 - self.Y) * preds_bin))
            fn = float(np.sum(self.Y * (1 - preds_bin)))

            dice = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
            iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            self.dice_history.append(dice)
            if dice > self.best_dice:
                self.best_dice = dice

            status = ""
            if dice > 0.9:
                status = "✓ OVERFIT SUCCESS"
            elif dice > 0.5:
                status = "… learning"
            elif dice > 0.1:
                status = "… starting"
            else:
                status = "✗ not learning yet"

            loss = logs.get("loss", 0)
            print(f"  Epoch {epoch+1:>3}/{epochs}  "
                  f"loss={loss:.4f}  "
                  f"MA Dice={dice:.4f}  IoU={iou:.4f}  "
                  f"Prec={prec:.4f}  Rec={rec:.4f}  "
                  f"[{status}]")

    dice_cb = DiceLogger(X, Y)

    print(f"\n{'='*70}")
    print(f"  OVERFIT TEST: {image_stem} → 1 image, {len(X)} patches, {epochs} epochs")
    print(f"  Model: {'Ghost-CAS-UNet v2' if use_ghost else 'Mini-UNet'}")
    print(f"  Expectation: MA Dice should reach > 0.9 if model can learn MA")
    print(f"{'='*70}\n")

    t0 = time.time()
    model.fit(
        X, Y,
        epochs=epochs,
        batch_size=min(16, len(X)),
        verbose=0,
        callbacks=[dice_cb],
    )
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  RESULT")
    print(f"{'='*70}")
    print(f"  Best MA Dice: {dice_cb.best_dice:.4f}")
    print(f"  Final MA Dice: {dice_cb.dice_history[-1]:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    print()

    if dice_cb.best_dice > 0.9:
        print("  ✅ MODEL CAN OVERFIT MA — architecture is sound")
        print("     Problem is purely data quantity/balance — NOT the model")
    elif dice_cb.best_dice > 0.5:
        print("  🟡 MODEL PARTIALLY LEARNS MA — architecture works but slowly")
        print("     May need more epochs, or MA patches are too sparse")
    else:
        print("  ❌ MODEL CANNOT LEARN MA — check architecture or loss function")
        print("     This means something fundamental is broken")

    print()

    # Save dice curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(range(1, len(dice_cb.dice_history)+1), dice_cb.dice_history,
            "r-o", markersize=3, label="MA Dice")
    ax.axhline(0.9, color="green", linestyle="--", alpha=0.5, label="Overfit target (0.9)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MA Dice")
    ax.set_title(f"Overfit Test: {image_stem} ({'Ghost-CAS-UNet' if use_ghost else 'Mini-UNet'})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = RESULTS / f"overfit_dice_{image_stem}_{'ghost' if use_ghost else 'mini'}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Dice curve saved → {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MA Overfit Sanity Check")
    parser.add_argument("--image", default="IDRiD_01",
                        help="IDRID image stem (default: IDRiD_01)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--ghost", action="store_true",
                        help="Use Ghost-CAS-UNet v2 instead of mini UNet")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Patch size (default: 256)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  MA OVERFIT SANITY CHECK")
    print(f"  Can our model memorize a single MA image?")
    print(f"{'='*70}")

    # Test 1: Mini-UNet (should overfit easily)
    if not args.ghost:
        run_overfit_test(args.image, args.epochs, use_ghost=False, patch_size=args.patch_size)

    # Test 2: Ghost-CAS-UNet v2 (our actual model)
    if args.ghost:
        run_overfit_test(args.image, args.epochs, use_ghost=True, patch_size=args.patch_size)

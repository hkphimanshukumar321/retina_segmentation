# -*- coding: utf-8 -*-
"""
MA Class Diagnostic — Pipeline Integrity Check
=================================================

Checks three hypotheses for MA Dice=0:

  1. PIPELINE BROKEN  — MA pixels not reaching the model at all?
  2. MASK OVERLAP     — MA pixels overwritten by HE/EX/SE/OD in multi-class mask?
  3. PIXEL STARVATION — MA ground truth too sparse for the model to learn?

Run locally:
    cd retina_scan
    python segmentation/scripts/diagnose_ma.py

Outputs detailed stats to terminal + saves PNG to results/dataset_vis/
"""

import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "segmentation"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
IDRID      = ROOT / "data" / "IDRID" / "A. Segmentation"
TRAIN_IMGS = IDRID / "1. Original Images" / "a. Training Set"
TRAIN_MASK = IDRID / "2. All Segmentation Groundtruths" / "a. Training Set"
OUT_DIR    = ROOT / "segmentation" / "results" / "dataset_vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES    = ["MA", "HE", "EX", "SE", "OD"]
SUBFOLDERS = ["1. Microaneurysms", "2. Haemorrhages",
              "3. Hard Exudates", "4. Soft Exudates", "5. Optic Disc"]
SUFFIXES   = ["_MA", "_HE", "_EX", "_SE", "_OD"]
COLOURS    = [(255,0,0), (0,255,0), (255,255,0), (0,255,255), (255,128,0)]

SEP = "=" * 70


def load_class_masks(stem, h, w):
    """Load all 5 class masks as raw grayscale + binarised."""
    raw, binary = {}, {}
    for c, (sub, suf) in enumerate(zip(SUBFOLDERS, SUFFIXES)):
        p = TRAIN_MASK / sub / f"{stem}{suf}.tif"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                raw[c] = m
                binary[c] = (m > 0).astype(np.uint8)
    return raw, binary


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1: Are MA pixels present in the raw masks?
# ═══════════════════════════════════════════════════════════════════════════════

def check_1_raw_masks():
    print(f"\n{SEP}")
    print("CHECK 1: Are MA pixels present in the raw IDRID mask files?")
    print(SEP)
    img_paths = sorted(TRAIN_IMGS.glob("*.jpg"))
    total_ma_px = 0
    imgs_with_ma = 0
    ma_stats = []

    for ip in img_paths:
        h, w = cv2.imread(str(ip)).shape[:2]
        raw, binary = load_class_masks(ip.stem, h, w)
        ma_px = int(binary.get(0, np.zeros((h,w))).sum())
        total_ma_px += ma_px
        if ma_px > 0:
            imgs_with_ma += 1
            unique = np.unique(raw[0]) if 0 in raw else []
            ma_stats.append((ip.stem, ma_px, list(unique)))
            print(f"  {ip.stem}: {ma_px:>8,} MA pixels  (raw values: {list(unique)})")

    print(f"\n  TOTAL MA pixels across train set: {total_ma_px:,}")
    print(f"  Images with MA: {imgs_with_ma}/{len(img_paths)}")
    if total_ma_px == 0:
        print("  ⚠ CRITICAL: Zero MA pixels found — mask files may be corrupt!")
    else:
        print(f"  ✓ MA pixels exist. Average per image: {total_ma_px/max(imgs_with_ma,1):,.0f}")
    return ma_stats


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2: Does MA overlap with other classes? (are pixels overwritten?)
# ═══════════════════════════════════════════════════════════════════════════════

def check_2_overlap():
    print(f"\n{SEP}")
    print("CHECK 2: MA overlap with other classes (HE, EX, SE, OD)")
    print(SEP)
    img_paths = sorted(TRAIN_IMGS.glob("*.jpg"))

    overlap_counts = {c: 0 for c in range(1, 5)}  # overlap with HE/EX/SE/OD
    total_ma = 0
    any_overlap = 0

    for ip in img_paths:
        h, w = cv2.imread(str(ip)).shape[:2]
        _, binary = load_class_masks(ip.stem, h, w)
        ma = binary.get(0, np.zeros((h, w), dtype=np.uint8))
        ma_px = int(ma.sum())
        total_ma += ma_px
        if ma_px == 0:
            continue

        for c in range(1, 5):
            other = binary.get(c, np.zeros((h, w), dtype=np.uint8))
            overlap = int(np.sum(ma & other))
            overlap_counts[c] += overlap
            if overlap > 0:
                pct = 100 * overlap / ma_px
                print(f"  {ip.stem}: MA overlaps {CLASSES[c]} — {overlap} px ({pct:.1f}% of MA)")
                any_overlap += overlap

    print(f"\n  Summary:")
    for c in range(1, 5):
        pct = 100 * overlap_counts[c] / max(total_ma, 1)
        print(f"    MA ∩ {CLASSES[c]}: {overlap_counts[c]:>8,} px ({pct:.2f}% of total MA)")
    print(f"    Total MA pixels: {total_ma:,}")
    print(f"    Total overlapping: {any_overlap:,} ({100*any_overlap/max(total_ma,1):.2f}%)")

    if any_overlap == 0:
        print("  ✓ NO overlap — MA masks are independent of other classes")
    else:
        print("  ⚠ Some overlap exists (this is NORMAL in IDRID — lesions can co-exist)")
        print("    Our loader uses independent channels so overlap does NOT erase MA")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3: MA pixel fraction in actual training patches
# ═══════════════════════════════════════════════════════════════════════════════

def check_3_patch_ma_fraction():
    print(f"\n{SEP}")
    print("CHECK 3: MA pixels surviving into 256x256 patches (simulated)")
    print(SEP)
    img_paths = sorted(TRAIN_IMGS.glob("*.jpg"))
    patch_size = 256
    n_patches_per_img = 100     # same as training config
    prob_lesion = 0.8           # our config
    prob_ma_target = 0.3        # our config

    total_patches = 0
    patches_with_ma = 0
    total_ma_px_in_patches = 0
    class_px_in_patches = [0, 0, 0, 0, 0]

    for ip in img_paths[:20]:  # check first 20 images
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        _, binary = load_class_masks(ip.stem, h, w)

        # Build merged mask
        merged = np.zeros((h, w, 5), dtype=np.uint8)
        for c in range(5):
            if c in binary:
                merged[..., c] = binary[c]

        for _ in range(n_patches_per_img):
            rand = np.random.random()
            placed = False
            ph = pw = patch_size

            # Hierarchical sampling (same as our idrid_loader.py)
            if rand < prob_ma_target:
                for cls_idx in [0, 3]:  # MA, SE
                    ys, xs = np.where(merged[..., cls_idx] > 0)
                    if len(ys) > 0:
                        k = np.random.randint(len(ys))
                        cy, cx = int(ys[k]), int(xs[k])
                        y = max(0, min(h - ph, cy - ph//2))
                        x = max(0, min(w - pw, cx - pw//2))
                        placed = True
                        break

            if not placed and rand < prob_lesion:
                lesion_map = merged.sum(axis=-1) > 0
                ys, xs = np.where(lesion_map)
                if len(ys) > 0:
                    k = np.random.randint(len(ys))
                    cy, cx = int(ys[k]), int(xs[k])
                    y = max(0, min(h - ph, cy - ph // 2))
                    x = max(0, min(w - pw, cx - pw // 2))
                    placed = True

            if not placed:
                y = np.random.randint(0, max(1, h - ph))
                x = np.random.randint(0, max(1, w - pw))

            patch_mask = merged[y:y+ph, x:x+pw]
            total_patches += 1

            for c in range(5):
                class_px_in_patches[c] += int(patch_mask[..., c].sum())

            if patch_mask[..., 0].sum() > 0:
                patches_with_ma += 1
                total_ma_px_in_patches += int(patch_mask[..., 0].sum())

    print(f"  Total patches sampled: {total_patches}")
    print(f"  Patches containing MA: {patches_with_ma} ({100*patches_with_ma/total_patches:.1f}%)")
    print(f"  Total MA pixels in patches: {total_ma_px_in_patches:,}")
    print(f"\n  Avg pixels per patch by class:")
    for c in range(5):
        avg = class_px_in_patches[c] / total_patches
        print(f"    {CLASSES[c]}: {avg:>8.1f} px/patch  "
              f"({100*avg/(patch_size*patch_size):.3f}% of patch area)")

    ma_frac = total_ma_px_in_patches / (total_patches * patch_size * patch_size)
    print(f"\n  MA fraction of total patch area: {ma_frac:.6f} ({ma_frac*100:.4f}%)")
    if ma_frac < 0.001:
        print("  ⚠ MA < 0.1% of patch area — this is WHY the model ignores MA")
        print("    Even with targeted sampling, MA signal is drowned by background")
    else:
        print("  ✓ MA has reasonable presence in patches")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 4: Visualise MA mask at full resolution (zoom into MA lesions)
# ═══════════════════════════════════════════════════════════════════════════════

def check_4_visualise_ma():
    print(f"\n{SEP}")
    print("CHECK 4: Visualising MA (zoomed) to verify mask quality")
    print(SEP)
    img_paths = sorted(TRAIN_IMGS.glob("*.jpg"))

    # Find first 4 images with MA
    ma_images = []
    for ip in img_paths:
        ma_path = TRAIN_MASK / "1. Microaneurysms" / f"{ip.stem}_MA.tif"
        if ma_path.exists():
            ma_raw = cv2.imread(str(ma_path), cv2.IMREAD_GRAYSCALE)
            if ma_raw is not None and ma_raw.sum() > 0:
                ma_images.append((ip, ma_raw))
        if len(ma_images) >= 4:
            break

    if not ma_images:
        print("  ⚠ No images with MA found!")
        return

    fig, axes = plt.subplots(len(ma_images), 4, figsize=(20, 5*len(ma_images)))
    fig.suptitle("MA Mask Diagnostic — Is MA Visible?\n"
                 "Col 1: Full image | Col 2: MA mask (full) | "
                 "Col 3: Zoomed MA region | Col 4: MA pixels overlaid on image",
                 fontsize=10)

    for row, (ip, ma_raw) in enumerate(ma_images):
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Binary MA mask
        ma_bin = (ma_raw > 0).astype(np.uint8)

        # Find centroid of largest MA cluster for zoom
        ys, xs = np.where(ma_bin > 0)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        zoom = 256
        y0 = max(0, cy - zoom); y1 = min(h, cy + zoom)
        x0 = max(0, cx - zoom); x1 = min(w, cx + zoom)

        # Col 1: Full image (small)
        axes[row, 0].imshow(cv2.resize(img, (512, 384)))
        axes[row, 0].set_title(f"{ip.stem}\n{int(ma_bin.sum())} MA px total", fontsize=8)
        axes[row, 0].axis("off")

        # Col 2: Full MA mask
        axes[row, 1].imshow(ma_bin * 255, cmap="Reds", vmin=0, vmax=255)
        axes[row, 1].set_title(f"MA mask (full)\nunique raw: {np.unique(ma_raw)}", fontsize=8)
        axes[row, 1].axis("off")

        # Col 3: Zoomed MA region
        zoom_img = img[y0:y1, x0:x1].copy()
        zoom_mask = ma_bin[y0:y1, x0:x1]
        # Overlay MA in red
        zoom_overlay = zoom_img.copy()
        zoom_overlay[zoom_mask > 0] = [255, 0, 0]
        axes[row, 2].imshow(zoom_overlay)
        axes[row, 2].set_title(f"Zoomed ({zoom_mask.sum()} MA px in crop)", fontsize=8)
        axes[row, 2].axis("off")

        # Col 4: MA diameter analysis
        # Connected components to measure individual MA sizes
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ma_bin, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
        if len(areas) > 0:
            title_txt = (f"MA lesions: {n_labels-1}\n"
                        f"sizes: min={areas.min()} max={areas.max()} "
                        f"mean={areas.mean():.1f} px")
        else:
            title_txt = "No MA components found"

        # Show histogram of MA lesion sizes
        if len(areas) > 0:
            axes[row, 3].hist(areas, bins=min(30, len(areas)), color='red', alpha=0.7)
            axes[row, 3].set_xlabel("MA lesion size (pixels)")
            axes[row, 3].set_ylabel("Count")
        axes[row, 3].set_title(title_txt, fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "5_ma_diagnostic.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("  MA CLASS DIAGNOSTIC — Why is MA Dice = 0?")
    print(f"{'='*70}")

    check_1_raw_masks()
    check_2_overlap()
    check_3_patch_ma_fraction()
    check_4_visualise_ma()

    print(f"\n{'='*70}")
    print("  DONE — Review results above + check 5_ma_diagnostic.png")
    print(f"{'='*70}\n")

# -*- coding: utf-8 -*-
"""
Dataset Visualisation Script
==============================
Generates local PNG grids verifying:
  1. Thresholding correctness (raw mask pixels vs binarised GT)
  2. Patch sampling quality (lesion-centred vs random)
  3. Train / Val / Test split coverage (class pixel counts)
  4. Augmentation pipeline output

Usage::

    cd /path/to/retina_segmentation
    python segmentation/scripts/visualise_dataset.py

Outputs saved to: segmentation/results/dataset_vis/
"""

import sys
from pathlib import Path
import numpy as np
import cv2

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "segmentation"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
IDRID_ROOT   = ROOT_DIR / "data" / "IDRID" / "A. Segmentation"
TRAIN_IMGS   = IDRID_ROOT / "1. Original Images"               / "a. Training Set"
TRAIN_MASKS  = IDRID_ROOT / "2. All Segmentation Groundtruths" / "a. Training Set"
VAL_IMGS     = IDRID_ROOT / "1. Original Images"               / "a. Val Set"
VAL_MASKS    = IDRID_ROOT / "2. All Segmentation Groundtruths" / "a. Val Set"
TEST_IMGS    = IDRID_ROOT / "1. Original Images"               / "b. Testing Set"
TEST_MASKS   = IDRID_ROOT / "2. All Segmentation Groundtruths" / "b. Testing Set"

OUT_DIR = ROOT_DIR / "segmentation" / "results" / "dataset_vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES     = ["MA", "HE", "EX", "SE", "OD"]
SUBFOLDERS  = ["1. Microaneurysms", "2. Haemorrhages",
               "3. Hard Exudates",  "4. Soft Exudates", "5. Optic Disc"]
SUFFIXES    = ["_MA", "_HE", "_EX", "_SE", "_OD"]
# Colours per class for overlays
COLOURS = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 128, 0)]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def load_merged_mask(img_stem, mask_root, orig_hw):
    h, w = orig_hw
    merged = np.zeros((h, w, 5), dtype=np.uint8)
    raw_vals = {}
    for c, (sub, suf) in enumerate(zip(SUBFOLDERS, SUFFIXES)):
        p = mask_root / sub / f"{img_stem}{suf}.tif"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                raw_vals[CLASSES[c]] = m
                merged[..., c] = (m > 0).astype(np.uint8)  # threshold: >0
    return merged, raw_vals


def overlay_masks(img_rgb, mask_5ch, alpha=0.45):
    """Return RGB image with coloured class overlays."""
    out = img_rgb.copy().astype(np.float32)
    for c, colour in enumerate(COLOURS):
        region = mask_5ch[..., c] > 0
        for ch, v in enumerate(colour):
            out[..., ch][region] = (1 - alpha) * out[..., ch][region] + alpha * v
    return np.clip(out, 0, 255).astype(np.uint8)


def extract_patch(img, mask, cy, cx, ph=256, pw=256):
    h, w = img.shape[:2]
    y = max(0, min(h - ph, cy - ph // 2))
    x = max(0, min(w - pw, cx - pw // 2))
    return img[y:y+ph, x:x+pw], mask[y:y+ph, x:x+pw]


# ══════════════════════════════════════════════════════════════════════════════
# 1. THRESHOLDING CHECK — Raw pixel value histogram vs binarised mask
# ══════════════════════════════════════════════════════════════════════════════

def vis_thresholding(num_images=3):
    print("[1/4] Thresholding verification ...")
    img_paths = sorted(TRAIN_IMGS.glob("*.jpg"))[:num_images]

    fig, axes = plt.subplots(num_images, 6, figsize=(24, num_images * 4))
    fig.suptitle("Thresholding Verification\n"
                 "Col 1: Original image | Col 2-6: Binarised GT (>0 threshold) per class\n"
                 "Red = lesion pixels. Title shows unique raw pixel values in mask.",
                 fontsize=10, y=1.01)

    for row, ip in enumerate(img_paths):
        img  = load_image(ip)
        h, w = img.shape[:2]
        merged, raw_vals = load_merged_mask(ip.stem, TRAIN_MASKS, (h, w))

        # Show resized image
        img_sm = cv2.resize(img, (512, 384))
        axes[row, 0].imshow(img_sm)
        axes[row, 0].set_title(f"{ip.stem}", fontsize=7)
        axes[row, 0].axis("off")

        for c, cls in enumerate(CLASSES):
            ax = axes[row, c + 1]
            if cls in raw_vals:
                raw = raw_vals[cls]
                unique_vals = np.unique(raw)
                binary = (raw > 0).astype(np.uint8) * 255
                # Show binarised mask with colour
                disp = cv2.resize(binary, (512, 384))
                ax.imshow(disp, cmap="Reds", vmin=0, vmax=255)
                ax.set_title(f"{cls} | unique={unique_vals}", fontsize=7)
            else:
                ax.imshow(np.zeros((384, 512), dtype=np.uint8), cmap="gray")
                ax.set_title(f"{cls} | NO MASK", fontsize=7, color="red")
            ax.axis("off")

    plt.tight_layout()
    out = OUT_DIR / "1_thresholding_check.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PATCH SAMPLING — Lesion-centred vs random patches
# ══════════════════════════════════════════════════════════════════════════════

def vis_patch_sampling(n_patches=6, patch_size=256):
    print("[2/4] Patch sampling verification ...")
    img_paths = sorted(TRAIN_IMGS.glob("*.jpg"))[:3]

    fig, axes = plt.subplots(3, n_patches + 1, figsize=(3 * (n_patches + 1), 10))
    fig.suptitle("Patch Sampling: Lesion-centred patches vs Random patches\n"
                 "Col 1: Full image with overlay | Green box = lesion patch | Yellow box = random patch",
                 fontsize=10, y=1.02)

    for row, ip in enumerate(img_paths):
        img = load_image(ip)
        h, w = img.shape[:2]
        merged, _ = load_merged_mask(ip.stem, TRAIN_MASKS, (h, w))
        overlay = overlay_masks(img, merged)
        ph = pw = patch_size

        # Show thumbnail
        thumb = cv2.resize(overlay, (512, 384))
        axes[row, 0].imshow(thumb)
        axes[row, 0].set_title(ip.stem, fontsize=7)
        axes[row, 0].axis("off")

        lesion_map = merged.sum(axis=-1) > 0
        ys, xs = np.where(lesion_map)

        for col in range(1, n_patches + 1):
            use_lesion = col <= n_patches // 2
            if use_lesion and len(ys) > 0:
                idx = np.random.randint(len(ys))
                cy, cx = int(ys[idx]), int(xs[idx])
                pi, pm = extract_patch(img, merged, cy, cx, ph, pw)
                label = f"Lesion patch {col}"
                border = (0, 200, 0)
            else:
                ry = np.random.randint(0, max(1, h - ph))
                rx = np.random.randint(0, max(1, w - pw))
                pi = img[ry:ry+ph, rx:rx+pw]
                pm = merged[ry:ry+ph, rx:rx+pw]
                label = f"Random patch {col}"
                border = (200, 200, 0)

            po = overlay_masks(pi, pm)
            # Add coloured border
            po = cv2.copyMakeBorder(po, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=border)
            axes[row, col].imshow(po)
            lesion_frac = pm.sum() / (ph * pw * 5 + 1e-6)
            axes[row, col].set_title(f"{label}\nlesion={lesion_frac:.3f}", fontsize=7)
            axes[row, col].axis("off")

    plt.tight_layout()
    out = OUT_DIR / "2_patch_sampling.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLASS DISTRIBUTION — Split coverage: pixel counts per class  
# ══════════════════════════════════════════════════════════════════════════════

def vis_class_distribution():
    print("[3/4] Class distribution across splits ...")
    splits = {
        "Train (43)": (TRAIN_IMGS, TRAIN_MASKS),
        "Val   (11)": (VAL_IMGS,   VAL_MASKS)   if VAL_IMGS.exists() else None,
        "Test  (27)": (TEST_IMGS,  TEST_MASKS),
    }

    stats = {}  # split -> {class: [img_count, total_pixels]}
    for split_name, paths in splits.items():
        if paths is None:
            print(f"   Skipping {split_name} (val split not on this machine)")
            continue
        img_dir, mask_dir = paths
        img_paths = sorted(img_dir.glob("*.jpg"))
        cls_px   = {c: 0 for c in CLASSES}
        cls_imgs = {c: 0 for c in CLASSES}
        for ip in img_paths:
            img = cv2.imread(str(ip))
            if img is None: continue
            h, w = img.shape[:2]
            merged, _ = load_merged_mask(ip.stem, mask_dir, (h, w))
            for ci, cls in enumerate(CLASSES):
                px = int(merged[..., ci].sum())
                if px > 0:
                    cls_px[cls]   += px
                    cls_imgs[cls] += 1
        stats[split_name] = {"px": cls_px, "imgs": cls_imgs, "n": len(img_paths)}

    if not stats:
        print("   No data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Class Distribution per Split", fontsize=13)
    x = np.arange(len(CLASSES))
    width = 0.25

    colours_bar = ["#3498db", "#2ecc71", "#f39c12"]
    for i, (split_name, data) in enumerate(stats.items()):
        px_vals   = [data["px"].get(c, 0)   for c in CLASSES]
        imgs_vals = [data["imgs"].get(c, 0) for c in CLASSES]
        n = data["n"]
        axes[0].bar(x + i * width, px_vals,   width, label=split_name, color=colours_bar[i], alpha=0.85)
        axes[1].bar(x + i * width, imgs_vals, width, label=split_name, color=colours_bar[i], alpha=0.85)

    axes[0].set_title("Total GT pixels per class")
    axes[0].set_xticks(x + width); axes[0].set_xticklabels(CLASSES)
    axes[0].set_ylabel("Pixel count"); axes[0].legend(); axes[0].set_yscale("log")

    axes[1].set_title("Images with non-zero GT per class")
    axes[1].set_xticks(x + width); axes[1].set_xticklabels(CLASSES)
    axes[1].set_ylabel("Image count"); axes[1].legend()

    plt.tight_layout()
    out = OUT_DIR / "3_class_distribution.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. OVERLAY GALLERY — Full image overlays for first 6 training images
# ══════════════════════════════════════════════════════════════════════════════

def vis_overlay_gallery(num_images=6):
    print("[4/4] Overlay gallery ...")
    img_paths = sorted(TRAIN_IMGS.glob("*.jpg"))[:num_images]

    fig, axes = plt.subplots(2, num_images // 2, figsize=(5 * (num_images // 2), 10))
    axes = axes.flatten()
    legend_patches = [mpatches.Patch(color=[c/255 for c in COLOURS[i]], label=CLASSES[i])
                      for i in range(5)]

    for idx, ip in enumerate(img_paths):
        img = load_image(ip)
        h, w = img.shape[:2]
        merged, _ = load_merged_mask(ip.stem, TRAIN_MASKS, (h, w))
        overlay = overlay_masks(img, merged)
        thumb = cv2.resize(overlay, (640, 480))
        axes[idx].imshow(thumb)
        axes[idx].set_title(ip.stem, fontsize=8)
        axes[idx].axis("off")

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=5, fontsize=10, frameon=True)
    fig.suptitle("Ground Truth Overlays — Training Images\n"
                 "MA=Red, HE=Green, EX=Yellow, SE=Cyan, OD=Orange", fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = OUT_DIR / "4_overlay_gallery.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n[Dataset Visualiser] Output dir: {OUT_DIR}\n")
    vis_thresholding(num_images=3)
    vis_patch_sampling(n_patches=6)
    vis_class_distribution()
    vis_overlay_gallery(num_images=6)
    print(f"\n[Done] All visualisations saved to: {OUT_DIR}")

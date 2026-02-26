# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
prepare_val_split.py
=====================

One-time script to create an IDRID validation split from the training set.

Split strategy (fixed seed, reproducible):
    - IDRID Training Set: 54 images (IDRiD_01 – IDRiD_54)
    - Train split (43 images): IDRiD_01 – IDRiD_43  → kept in 'a. Training Set'
    - Val   split (11 images): IDRiD_44 – IDRiD_54  → COPIED to 'a. Val Set'

No files are moved or deleted. The original 'a. Training Set' is UNCHANGED.
Only images IDRiD_44–54 are additionally copied into the new validation folders.

Usage (on server):
    cd /home/abhays/retina_segmentation
    python segmentation/scripts/prepare_val_split.py

    # Dry-run (no files written):
    python segmentation/scripts/prepare_val_split.py --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()          # segmentation/scripts/
SEG_DIR    = SCRIPT_DIR.parent.resolve()               # segmentation/
ROOT_DIR   = SEG_DIR.parent.resolve()                  # retina_segmentation/

IDRID_ROOT = ROOT_DIR / "data" / "IDRID" / "A. Segmentation"

ORIG_IMG_TRAIN   = IDRID_ROOT / "1. Original Images"   / "a. Training Set"
NEW_IMG_VAL      = IDRID_ROOT / "1. Original Images"   / "a. Val Set"

ORIG_MASK_TRAIN  = IDRID_ROOT / "2. All Segmentation Groundtruths" / "a. Training Set"
NEW_MASK_VAL     = IDRID_ROOT / "2. All Segmentation Groundtruths" / "a. Val Set"

# Subfolders (class mask folders) that must exist in both train and val
CLASS_SUBFOLDERS = [
    "1. Microaneurysms",
    "2. Haemorrhages",
    "3. Hard Exudates",
    "4. Soft Exudates",
    "5. Optic Disc",
]

# Class suffix used in mask filenames (same order)
CLASS_SUFFIXES   = ["_MA", "_HE", "_EX", "_SE", "_OD"]

# Validation image IDs (1-indexed numbers to use as val)
VAL_IDS = list(range(44, 55))   # 44, 45, ..., 54 → 11 images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy(src: Path, dst: Path, dry_run: bool):
    if dry_run:
        print(f"    [DRY] copy {src.name} → {dst}")
    else:
        shutil.copy2(str(src), str(dst))


def prepare_split(dry_run: bool = False):
    print("=" * 60)
    print("  IDRID VAL SPLIT — prepare_val_split.py")
    print("=" * 60)
    print(f"  Root          : {IDRID_ROOT}")
    print(f"  Val image dir : {NEW_IMG_VAL}")
    print(f"  Val mask dir  : {NEW_MASK_VAL}")
    print(f"  Val IDs       : IDRiD_{VAL_IDS[0]:02d} – IDRiD_{VAL_IDS[-1]:02d} ({len(VAL_IDS)} images)")
    print(f"  Dry run       : {dry_run}")
    print()

    # Validate source directories
    if not ORIG_IMG_TRAIN.exists():
        print(f"[ERROR] Training image dir not found: {ORIG_IMG_TRAIN}")
        sys.exit(1)
    if not ORIG_MASK_TRAIN.exists():
        print(f"[ERROR] Training mask dir not found: {ORIG_MASK_TRAIN}")
        sys.exit(1)

    # Create destination directories
    if not dry_run:
        NEW_IMG_VAL.mkdir(parents=True, exist_ok=True)
        for sub in CLASS_SUBFOLDERS:
            (NEW_MASK_VAL / sub).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Val directories created (or already existed).")
    else:
        print(f"[DRY] Would create: {NEW_IMG_VAL}")
        for sub in CLASS_SUBFOLDERS:
            print(f"[DRY] Would create: {NEW_MASK_VAL / sub}")

    print()

    # ---- Copy images ----
    print("  [1/2] Copying validation images ...")
    img_count = 0
    missing_imgs = []

    for id_ in VAL_IDS:
        stem = f"IDRiD_{id_:02d}"
        src = ORIG_IMG_TRAIN / f"{stem}.jpg"

        if not src.exists():
            missing_imgs.append(str(src))
            print(f"    [WARN] Image not found: {src.name}")
            continue

        dst = NEW_IMG_VAL / src.name
        _copy(src, dst, dry_run)
        img_count += 1

    print(f"       → {img_count}/{len(VAL_IDS)} images copied")
    if missing_imgs:
        print(f"       → {len(missing_imgs)} images MISSING (see above)")

    print()

    # ---- Copy masks ----
    print("  [2/2] Copying validation masks ...")
    mask_count  = 0
    mask_missing = []

    for id_ in VAL_IDS:
        stem = f"IDRiD_{id_:02d}"
        for sub, suffix in zip(CLASS_SUBFOLDERS, CLASS_SUFFIXES):
            mask_name = f"{stem}{suffix}.tif"
            src  = ORIG_MASK_TRAIN / sub / mask_name
            dst  = NEW_MASK_VAL / sub / mask_name

            if not src.exists():
                # Some classes (especially SE) may not have masks for every image
                mask_missing.append(mask_name)
                continue

            _copy(src, dst, dry_run)
            mask_count += 1

    expected_masks = len(VAL_IDS) * len(CLASS_SUBFOLDERS)
    print(f"       → {mask_count}/{expected_masks} masks copied")
    if mask_missing:
        print(f"       → {len(mask_missing)} masks not present (normal for SE/OD if images lack that lesion):")
        for m in mask_missing:
            print(f"           {m}")

    print()

    # ---- Summary ----
    print("=" * 60)
    if dry_run:
        print("  DRY RUN COMPLETE — no files were written.")
    else:
        print("  SPLIT COMPLETE")
        print(f"  Train : IDRiD_01 – IDRiD_43  (in original 'a. Training Set')")
        print(f"  Val   : IDRiD_44 – IDRiD_54  (in new 'a. Val Set')")
        print(f"  Test  : IDRiD_55 – IDRiD_81  (in existing 'b. Testing Set')")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create IDRID validation split from the training set."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing any files.",
    )
    args = parser.parse_args()
    prepare_split(dry_run=args.dry_run)

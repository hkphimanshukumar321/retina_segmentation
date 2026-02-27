
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import tensorflow as tf
from common.data_loader import PatchDataGenerator, decode_labelmap

logger = logging.getLogger(__name__)


class IDRIDPatchDataGenerator(PatchDataGenerator):
    """
    Data Generator for IDRID dataset — upgraded with literature-backed
    data representation improvements (Porwal et al. MedIA 2020,
    EAD-Net TMI 2021, MLI copy-paste 2023).

    Improvements over v1:
      1. Ben Graham + CLAHE preprocessing — fixes illumination, boosts MA visibility
      2. FOV mask — excludes black circular border from random patch sampling
      3. Class-targeted sampling — dedicates 30% of patches to MA/SE pixels
      4. Copy-paste augmentation — pastes MA/SE lesions across images

    IDRID Structure:
    - Images: "1. Original Images/a. Training Set/*.jpg"
    - Masks:  "2. All Segmentation Groundtruths/a. Training Set/"
               subfolders: "1. Microaneurysms", "2. Haemorrhages", etc.
    """

    # Internal class order: [MA, HE, EX, SE, OD]
    SUBFOLDERS = [
        "1. Microaneurysms",
        "2. Haemorrhages",
        "3. Hard Exudates",
        "4. Soft Exudates",
        "5. Optic Disc"
    ]
    SUFFIXES = ["_MA", "_HE", "_EX", "_SE", "_OD"]

    # Classes considered "starving" — specifically targeted in sampling & copy-paste
    SPARSE_CLASS_INDICES = [0, 3]   # MA=0, SE=3

    def __init__(
        self,
        image_dir: Path,
        mask_root_dir: Path,
        config,
        patches_per_image: int = 50,
        batch_size: int = 8,
        augmentations=None,
        check_files: bool = True,
        enable_copy_paste: bool = True,   # Technique 4 toggle
    ):
        """
        Args:
            image_dir:          Path to '1. Original Images/a. Training Set'
            mask_root_dir:      Path to '2. All Segmentation Groundtruths/a. Training Set'
            config:             SegmentationConfig (DataConfig attributes used)
            patches_per_image:  Patches extracted per image per epoch
            batch_size:         Batch size returned per __getitem__ call
            augmentations:      Optional callable applied to uint8 patches
            check_files:        Raise if no images found
            enable_copy_paste:  Enable Technique 4 (MA/SE cross-image paste)
        """
        self.image_dir       = image_dir
        self.mask_root_dir   = mask_root_dir
        self.enable_copy_paste = enable_copy_paste

        # Collect image paths
        self.image_paths = sorted(list(image_dir.glob("*.jpg")))
        if not self.image_paths:
            self.image_paths = sorted(list(image_dir.rglob("*.jpg")))
        if check_files and not self.image_paths:
            raise ValueError(f"No .jpg images found in {image_dir}")

        super().__init__(
            self.image_paths,
            [],   # mask_paths unused — we load per-class masks dynamically
            config,
            patches_per_image,
            batch_size,
            augmentations
        )

        self.patch_size = config.data.img_size

        # Read preprocessing flags from config (with safe defaults)
        self._use_ben_graham  = getattr(config.data, 'use_ben_graham', True)
        self._use_clahe       = getattr(config.data, 'use_clahe', True)
        self._clahe_clip      = getattr(config.data, 'clahe_clip', 2.0)
        self._prob_lesion     = getattr(config.data, 'prob_lesion', 0.8)
        self._prob_ma_target  = getattr(config.data, 'prob_ma_targeted', 0.3)

        # --- Cache everything into RAM (uint8 to save memory) ---
        self.cache_x   = {}   # img_idx → uint8 (H, W, 3)  ← preprocessed
        self.cache_y   = {}   # img_idx → uint8 (H, W, 5)  ← binary masks
        self.cache_fov = {}   # img_idx → bool  (H, W)      ← FOV mask

        logger.info(f"Pre-loading {len(self.image_paths)} IDRID samples into RAM...")
        logger.info(f"  Ben Graham: {self._use_ben_graham}, CLAHE: {self._use_clahe}, "
                    f"prob_lesion: {self._prob_lesion}, prob_ma_target: {self._prob_ma_target}")

        # Import with fallback for both package and direct script usage
        try:
            from segmentation.src.retinal_preprocessing import retinal_preprocess, compute_fov_mask
        except ImportError:
            from src.retinal_preprocessing import retinal_preprocess, compute_fov_mask

        for idx, img_path in enumerate(self.image_paths):
            raw = cv2.imread(str(img_path))
            if raw is None:
                logger.warning(f"Failed to load {img_path}")
                continue
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            h, w    = raw_rgb.shape[:2]

            # Technique 1+2: Preprocessing → float32 [0,1] → back to uint8 for cache
            proc_f   = retinal_preprocess(raw_rgb,
                                          use_ben_graham=self._use_ben_graham,
                                          use_clahe=self._use_clahe,
                                          clahe_clip=self._clahe_clip)
            proc_u8  = (proc_f * 255).astype(np.uint8)

            # Technique 3b: FOV mask (computed from raw, not preprocessed)
            fov = compute_fov_mask(raw_rgb)

            # Binary merged mask (H, W, 5), uint8
            merged = self._load_merged_mask(img_path.stem, (h, w))

            self.cache_x[idx]   = proc_u8
            self.cache_y[idx]   = merged
            self.cache_fov[idx] = fov

        logger.info("IDRID Cache complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Mask loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_merged_mask(self, img_stem: str, original_hw: Tuple[int, int]) -> np.ndarray:
        """Load all 5 class masks and merge into (H, W, 5) uint8 binary.
        
        MA (class 0) and SE (class 3) masks are dilated by 7px to expand
        tiny lesion targets — "soft boundary expansion" from top IDRiD teams.
        """
        h, w = original_hw
        merged = np.zeros((h, w, 5), dtype=np.uint8)
        # 7px elliptical kernel for MA/SE dilation
        _dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        for c, (subfolder, suffix) in enumerate(zip(self.SUBFOLDERS, self.SUFFIXES)):
            p = self.mask_root_dir / subfolder / f"{img_stem}{suffix}.tif"
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                binary = (m > 0).astype(np.uint8)
                # Dilate sparse classes — MA and SE lesions are too small at 256-512px
                if c in self.SPARSE_CLASS_INDICES:
                    binary = cv2.dilate(binary, _dilate_kernel, iterations=1)
                merged[..., c] = binary
        return merged

    # ─────────────────────────────────────────────────────────────────────────
    # Keras Sequence API
    # ─────────────────────────────────────────────────────────────────────────

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self._data_generation(batch_indices)

    # ─────────────────────────────────────────────────────────────────────────
    # Patch sampling & batch generation (core of all 4 techniques)
    # ─────────────────────────────────────────────────────────────────────────

    def _data_generation(self, batch_indices):
        batch_X, batch_Y = [], []
        ph, pw = self.patch_size
        n_images = len(self.cache_x)

        for i in batch_indices:
            img_idx = i // self.patches_per_image
            if img_idx not in self.cache_x:
                continue

            img_u8   = self.cache_x[img_idx]
            mask_u8  = self.cache_y[img_idx]
            fov_mask = self.cache_fov.get(img_idx)

            h, w = img_u8.shape[:2]

            # Resize if image is somehow smaller than patch
            if h < ph or w < pw:
                scale  = max(ph / h, pw / w) + 0.01
                img_u8  = cv2.resize(img_u8,  (int(w*scale), int(h*scale)))
                mask_u8 = cv2.resize(mask_u8, (int(w*scale), int(h*scale)),
                                     interpolation=cv2.INTER_NEAREST)
                fov_mask = None  # can't resize bool mask reliably, skip FOV
                h, w = img_u8.shape[:2]

            # ── Technique 3: Hierarchical patch sampling ─────────────────────
            rand = np.random.random()
            patch_placed = False

            # (a) MA/SE targeted — force patch onto a sparse-class pixel
            if rand < self._prob_ma_target:
                for cls_idx in self.SPARSE_CLASS_INDICES:
                    ys, xs = np.where(mask_u8[..., cls_idx] > 0)
                    if len(ys) > 0:
                        k = np.random.randint(len(ys))
                        cy, cx = int(ys[k]), int(xs[k])
                        y = np.clip(cy - ph // 2, 0, h - ph)
                        x = np.clip(cx - pw // 2, 0, w - pw)
                        patch_placed = True
                        break

            # (b) Any-lesion centred
            if not patch_placed and rand < self._prob_lesion:
                lesion_map = mask_u8.sum(axis=-1) > 0
                ys, xs = np.where(lesion_map)
                if len(ys) > 0:
                    k = np.random.randint(len(ys))
                    cy, cx = int(ys[k]), int(xs[k])
                    y = np.clip(cy - ph // 2, 0, h - ph)
                    x = np.clip(cx - pw // 2, 0, w - pw)
                    patch_placed = True

            # (c) Random — but restricted to FOV region (Technique 2)
            if not patch_placed:
                y, x = self._sample_fov_position(h, w, ph, pw, fov_mask)

            # Extract patches
            patch_img  = img_u8 [y:y+ph, x:x+pw].copy()
            patch_mask = mask_u8[y:y+ph, x:x+pw].copy()

            # ── Technique 4: Copy-paste for MA/SE ────────────────────────────
            if self.enable_copy_paste and n_images > 1:
                patch_img, patch_mask = self._copy_paste_sparse(
                    patch_img, patch_mask, img_idx, ph, pw
                )

            # ── Augmentation (on uint8) ───────────────────────────────────────
            if self.augmentations:
                seed = np.random.randint(0, 2**31)
                np.random.seed(seed)
                patch_img  = self.augmentations(patch_img)
                np.random.seed(seed)
                patch_mask = self.augmentations(patch_mask)

            # Float32 normalise (already preprocessed in cache)
            batch_X.append(patch_img.astype(np.float32) / 255.0)
            batch_Y.append(patch_mask.astype(np.float32))

        if not batch_X:
            # Fallback: return zeros to prevent crash
            dummy_x = np.zeros((1, ph, pw, 3), dtype=np.float32)
            dummy_y = np.zeros((1, ph, pw, 5), dtype=np.float32)
            return dummy_x, dummy_y

        return (np.array(batch_X, dtype=np.float32),
                np.array(batch_Y, dtype=np.float32))

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: FOV-restricted random position (Technique 2)
    # ─────────────────────────────────────────────────────────────────────────

    def _sample_fov_position(self, h, w, ph, pw, fov_mask):
        """Sample top-left corner for random patch inside FOV.

        If FOV mask is available, tries up to 10 times to find a position
        where the patch centre lies inside the mask. Falls back to uniform
        random if no valid position found in 10 trials (rare).
        """
        if fov_mask is not None:
            for _ in range(10):
                y = np.random.randint(0, max(1, h - ph))
                x = np.random.randint(0, max(1, w - pw))
                # Accept if patch centre is inside FOV
                cy, cx = y + ph // 2, x + pw // 2
                if 0 <= cy < h and 0 <= cx < w and fov_mask[cy, cx]:
                    return y, x
        # Fallback — plain random
        y = np.random.randint(0, max(1, h - ph))
        x = np.random.randint(0, max(1, w - pw))
        return y, x

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: Copy-paste augmentation for MA / SE (Technique 4)
    # ─────────────────────────────────────────────────────────────────────────

    def _copy_paste_sparse(self, patch_img, patch_mask, src_idx, ph, pw):
        """Paste MA or SE lesion region from a random donor image.

        With probability 0.3, selects a sparse class (MA or SE), picks a
        random donor image that has that class, crops a small region around
        a lesion pixel, and blends it into the current patch using
        seamlessClone (Poisson blending).

        If seamlessClone fails (small lesion, edge case), falls back to
        alpha-blend to ensure no crash.

        Reference:
            Nguyen et al., "Multiple Lesions Insertion", 2023.
            Semi-supervised copy-paste, IDRiD 2022.
        """
        if np.random.random() > 0.3:
            return patch_img, patch_mask

        # Pick a sparse class to copy
        cls_idx = int(np.random.choice(self.SPARSE_CLASS_INDICES))

        # Find a donor image that has this class (≠ current image)
        candidates = [i for i in self.cache_y
                      if i != src_idx and self.cache_y[i][..., cls_idx].sum() > 0]
        if not candidates:
            return patch_img, patch_mask

        don_idx  = int(np.random.choice(candidates))
        don_img  = self.cache_x[don_idx]
        don_mask = self.cache_y[don_idx]

        # Find a lesion pixel in donor
        dys, dxs = np.where(don_mask[..., cls_idx] > 0)
        if len(dys) == 0:
            return patch_img, patch_mask

        k  = np.random.randint(len(dys))
        dy = int(dys[k]); dx = int(dxs[k])

        # Crop a small region around the lesion (up to 64×64, at most ph/2)
        crop_h = min(64, ph // 2)
        crop_w = min(64, pw // 2)
        dh, dw = don_img.shape[:2]

        y0 = max(0, dy - crop_h // 2); y1 = min(dh, y0 + crop_h)
        x0 = max(0, dx - crop_w // 2); x1 = min(dw, x0 + crop_w)
        if y1 - y0 < 4 or x1 - x0 < 4:
            return patch_img, patch_mask

        src_crop   = don_img [y0:y1, x0:x1].copy()
        mask_crop  = don_mask[y0:y1, x0:x1, cls_idx].copy()  # (h, w) binary

        ch, cw = src_crop.shape[:2]

        # Random placement inside current patch
        ty = np.random.randint(0, max(1, ph - ch))
        tx = np.random.randint(0, max(1, pw - cw))
        cy_center = ty + ch // 2
        cx_center = tx + cw // 2

        result_img  = patch_img.copy()
        result_mask = patch_mask.copy()

        # Expand lesion binary mask for seamlessClone (needs uint8 0/255)
        clone_mask = (mask_crop * 255).astype(np.uint8)

        # Only blend where mask has lesion pixels — skip if no lesion
        if clone_mask.sum() == 0:
            return patch_img, patch_mask

        try:
            blended = cv2.seamlessClone(
                src_crop,
                patch_img,
                clone_mask,
                (cx_center, cy_center),
                cv2.NORMAL_CLONE,
            )
            result_img = blended
        except cv2.error:
            # Poisson blend fails at image boundary — fall back to alpha blend
            alpha = 0.7
            roi = result_img[ty:ty+ch, tx:tx+cw]
            if roi.shape == src_crop.shape:
                result_img[ty:ty+ch, tx:tx+cw] = (
                    alpha * src_crop + (1 - alpha) * roi
                ).astype(np.uint8)

        # Update mask — OR in the copied lesion region
        result_mask[ty:ty+ch, tx:tx+cw, cls_idx] = np.maximum(
            result_mask[ty:ty+ch, tx:tx+cw, cls_idx],
            mask_crop.astype(np.uint8),
        )

        return result_img, result_mask

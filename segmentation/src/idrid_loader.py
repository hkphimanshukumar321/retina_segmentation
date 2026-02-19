
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
    Data Generator specifically for IDRID dataset structure.
    
    IDRID Structure:
    - Images: "1. Original Images/a. Training Set/*.jpg"
    - Masks: "2. All Segmentation Groundtruths/a. Training Set/" 
             containing subfolders: "1. Microaneurysms", "2. Haemorrhages", etc.
             
    This loader matches images to their corresponding multiple mask files
    and merges them into a single (H, W, 5) multi-class mask.
    """
    
    # Mapping of official IDRID subfolders to our internal class order
    # Order: [MA, HE, EX, SE, OD]
    SUBFOLDERS = [
        "1. Microaneurysms",
        "2. Haemorrhages",
        "3. Hard Exudates", 
        "4. Soft Exudates",
        "5. Optic Disc"
    ]
    
    # Suffixes found in IDRID filenames (e.g., IDRiD_01_MA.tif)
    SUFFIXES = ["_MA", "_HE", "_EX", "_SE", "_OD"]

    def __init__(
        self,
        image_dir: Path,
        mask_root_dir: Path,
        config,
        patches_per_image: int = 50,
        batch_size: int = 8,
        augmentations=None,
        check_files: bool = True
    ):
        """
        Args:
            image_dir: Path to '1. Original Images/a. Training Set'
            mask_root_dir: Path to '2. All Segmentation Groundtruths/a. Training Set'
        """
        self.image_dir = image_dir
        self.mask_root_dir = mask_root_dir
        
        # 1. Collect all image paths
        self.image_paths = sorted(list(image_dir.glob("*.jpg")))
        if not self.image_paths:
             # Try recursive if not found immediately (though structure is flat)
             self.image_paths = sorted(list(image_dir.rglob("*.jpg")))
             
        if check_files and not self.image_paths:
            raise ValueError(f"No .jpg images found in {image_dir}")

        # 2. Verify mask availability for each image
        # We don't store single mask paths; we store the ID (filename stem)
        # and construct paths on the fly in __data_generation
        
        # Base constructor needs lists, but we override generation logic
        # so we pass dummy lists to satisfy __init__ if needed, or just standard ones.
        # PatchDataGenerator stores self.image_paths and self.mask_paths.
        # We will use image_paths as is, and generate mask logic dynamically.
        
        super().__init__(
            self.image_paths, 
            [], # Empty mask paths, we won't use this list directly from parent logic
            config, 
            patches_per_image, 
            batch_size, 
            augmentations
        )
        
        self.patch_size = config.data.img_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indices of the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X, y = self._data_generation(batch_indices)
        return X, y

    def _load_merged_mask(self, img_stem: str, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Load masks for all 5 classes and merge into (H, W, 5).
        
        Args:
            img_stem: Filename stem like 'IDRiD_01'
            original_shape: (H, W) of the corresponding image
        """
        h, w = original_shape
        merged_mask = np.zeros((h, w, 5), dtype=np.float32)
        
        for i, (subfolder, suffix) in enumerate(zip(self.SUBFOLDERS, self.SUFFIXES)):
            # IDRID convention: IDRiD_01.jpg -> IDRiD_01_MA.tif
            mask_name = f"{img_stem}{suffix}.tif"
            mask_path = self.mask_root_dir / subfolder / mask_name
            
            if mask_path.exists():
                # Load mask
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    # corrupted or unreadable
                    continue
                
                # Resize if differs from image (should not happen in IDRID usually)
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Binarize and assign to channel i
                # IDRID masks are 0/255 usually, sometimes labeled.
                # safely assume > 0 is foreground
                merged_mask[..., i] = (m > 0).astype(np.float32)
            else:
                # File missing means empty mask for this class (e.g. no Hemorrhages in this image)
                pass
                
        return merged_mask

    def _data_generation(self, batch_indices):
        """Generates data containing batch_size samples"""
        batch_X = []
        batch_Y = []
        
        for i in batch_indices:
            # Map flattened index back to image index
            img_idx = i // self.patches_per_image
            
            img_path = self.image_paths[img_idx]
            img_stem = img_path.stem  # e.g. "IDRiD_01"
            
            # Load Image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            h, w = img.shape[:2]
            
            # Load Merged Mask
            # Mask shape: (H, W, 5)
            full_mask = self._load_merged_mask(img_stem, (h, w))
            
            # --- PATCH EXTRACTION (Adapted from parent) ---
            ph, pw = self.patch_size
            
            # Safeguard resize
            if h < ph or w < pw:
                scale = max(ph / h, pw / w) + 0.01
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                full_mask = cv2.resize(full_mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
                h, w = img.shape[:2]
                
            # Lesion-aware sampling
            prob_lesion = getattr(self.config.data, 'prob_lesion', 0.5)
            use_lesion_center = False
            
            if np.random.random() < prob_lesion:
                # Any class active
                # full_mask is (H, W, 5), sum over channels to find any lesion
                lesion_map = full_mask.sum(axis=-1) > 0
                ys, xs = np.where(lesion_map)
                
                if len(ys) > 0:
                    use_lesion_center = True
                    idx = np.random.randint(0, len(ys))
                    cy, cx = ys[idx], xs[idx]
                    
                    y_min = max(0, cy - ph + 1)
                    y_max = min(h - ph, cy)
                    if y_max < y_min: y_max = y_min
                    
                    x_min = max(0, cx - pw + 1)
                    x_max = min(w - pw, cx)
                    if x_max < x_min: x_max = x_min
                    
                    y = np.random.randint(y_min, y_max + 1) if y_max >= y_min else 0
                    x = np.random.randint(x_min, x_max + 1) if x_max >= x_min else 0
            
            if not use_lesion_center:
                y = np.random.randint(0, max(1, h - ph))
                x = np.random.randint(0, max(1, w - pw))
                
            patch_img = img[y:y+ph, x:x+pw]
            patch_mask = full_mask[y:y+ph, x:x+pw]
            
            # Augmentation
            if self.augmentations:
                seed = np.random.randint(0, 2**31)
                np.random.seed(seed)
                patch_img = self.augmentations(patch_img)
                np.random.seed(seed)
                # mask is (H, W, 5), simple affine transforms work fine on multi-channel
                patch_mask = self.augmentations(patch_mask)
                
            batch_X.append(patch_img.astype(np.float32))
            batch_Y.append(patch_mask.astype(np.float32))
            
        return np.array(batch_X, dtype=np.float32), np.array(batch_Y, dtype=np.float32)


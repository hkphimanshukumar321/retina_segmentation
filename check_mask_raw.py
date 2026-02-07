
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from segmentation.config import SegmentationConfig

def check_raw_masks():
    print("Checking Raw Masks...")
    config = SegmentationConfig()
    data_dir = config.data.data_dir
    images_dir = data_dir / config.data.img_dir
    masks_dir = data_dir / config.data.mask_dir
    
    print(f"Masks Dir: {masks_dir}")
    
    if not masks_dir.exists():
        print("Masks directory not found.")
        return

    mask_files = sorted(list(masks_dir.glob("*.[jp][pn][g]")))
    if not mask_files:
        print("No mask files found.")
        return
        
    sample_path = mask_files[0]
    print(f"Analyzing: {sample_path}")
    
    # Load RAW
    img = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to load image.")
        return
        
    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    
    unique, counts = np.unique(img, return_counts=True)
    print(f"Unique Values Count: {len(unique)}")
    
    sorted_indices = np.argsort(-counts)
    print("Top 20 Values:")
    for i in sorted_indices[:20]:
        print(f"  Value: {unique[i]} (Count: {counts[i]})")
        
    if len(img.shape) == 3:
        print("Image has 3 channels. Checking channel spread...")
        for c in range(img.shape[2]):
             u, _ = np.unique(img[:,:,c], return_counts=True)
             print(f"  Channel {c} unique count: {len(u)}")

if __name__ == "__main__":
    check_raw_masks()

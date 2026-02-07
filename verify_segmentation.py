
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from segmentation.src.models import create_unet_model
from common.data_loader import load_image, decode_bitmask
from segmentation.config import SegmentationConfig

def verify_segmentation():
    print("="*60)
    print("SEGMENTATION VERIFICATION (MULTI-LABEL)")
    print("="*60)
    sys.stdout.flush()

    # 1. Check Directories
    config = SegmentationConfig()
    data_dir = config.data.data_dir
    print(f"Data Directory: {data_dir}")
    
    # Use config paths (which now auto-detect)
    images_dir = data_dir / config.data.img_dir
    masks_dir = data_dir / config.data.mask_dir

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return
    
    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        print(f"Expected: {config.data.img_dir}")
        return
        
    if not masks_dir.exists():
        print(f"[ERROR] Masks directory not found: {masks_dir}")
        print(f"Expected: {config.data.mask_dir}")
        return

    # 2. Check Files
    image_files = sorted(list(images_dir.glob("*.[jp][pn][g]")))
    mask_files = sorted(list(masks_dir.glob("*.[jp][pn][g]"))) 
    
    print(f"Found {len(image_files)} images.")
    print(f"Found {len(mask_files)} masks.")

    if len(image_files) == 0:
        print("[ERROR] No images found.")
        return

    # 3. Check Matching
    if len(image_files) != len(mask_files):
        print(f"[WARNING] Mismatch in count: {len(image_files)} images vs {len(mask_files)} masks.")
    
    # 4. Try Loading
    print("\nTest Loading...")
    try:
        sample_img_path = image_files[0]
        sample_mask_path = mask_files[0]
        
        # Load Image (Standard)
        img = load_image(sample_img_path, img_size=config.data.img_size)
        
        # Load Mask (CRITICAL: normalize=False)
        mask = load_image(sample_mask_path, img_size=config.data.img_size, color_mode='grayscale', normalize=False)
        
        if img is None:
             print(f"[ERROR] Failed to load image: {sample_img_path}")
        else:
             print(f"[PASS] Image loaded. Shape: {img.shape}, Range: [{img.min():.2f}, {img.max():.2f}]")

        if mask is None:
             print(f"[ERROR] Failed to load mask: {sample_mask_path}")
        else:
             unique, counts = np.unique(mask, return_counts=True)
             
             with open("mask_analysis.txt", "w") as f:
                 f.write(f"Mask Shape: {mask.shape}\n")
                 f.write(f"Unique values count: {len(unique)}\n")
                 f.write("Top 20 most frequent values:\n")
                 sorted_indices = np.argsort(-counts)
                 for i in sorted_indices[:20]:
                     f.write(f"Value: {unique[i]} (Count: {counts[i]})\n")
                 
                 if len(unique) <= config.model.num_classes and unique.max() < config.model.num_classes:
                     f.write("Mask seems valid (values roughly 0-2).\n")
                 else:
                     f.write("Mask values are NOT simple class indices.\n")
                     
             print("[PASS] Analysis written to mask_analysis.txt")
             
             # 4b. Analyze RAW Mask (No resize)
             print("\nAnalyzing Raw Mask (No Resize)...")
             raw_mask = cv2.imread(str(sample_mask_path), cv2.IMREAD_GRAYSCALE)
             if raw_mask is None:
                 # Try IMREAD_UNCHANGED to see if it implies alpha or 16-bit
                 raw_mask = cv2.imread(str(sample_mask_path), cv2.IMREAD_UNCHANGED)
             
             if raw_mask is not None:
                 raw_unique, raw_counts = np.unique(raw_mask, return_counts=True)
                 print(f"       Raw Shape: {raw_mask.shape}")
                 print(f"       Raw Unique values count: {len(raw_unique)}")
                 with open("mask_analysis_raw.txt", "w") as f:
                     f.write(f"Raw Shape: {raw_mask.shape}\n")
                     f.write(f"Raw Unique count: {len(raw_unique)}\n")
                     f.write(f"Top 20 raw values:\n")
                     raw_indices = np.argsort(-raw_counts)
                     for i in raw_indices[:20]:
                         f.write(f"Value: {raw_unique[i]} (Count: {raw_counts[i]})\n")
             else:
                 print("[ERROR] Failed to load raw mask.")

             # 5. Test Decoding
             if raw_mask is not None:
                 print("\nTesting Bit-Mask Decoding...")
                 bit_values = config.data.bit_values
                 print(f"Config Bit Values: {bit_values}")
                 
                 decoded = decode_bitmask(raw_mask, num_classes=config.model.num_classes, bit_values=bit_values)
                 print(f"Decoded Shape: {decoded.shape} (Expected: {raw_mask.shape} + {config.model.num_classes} channels)")
                 
                 for i in range(config.model.num_classes):
                     layer = decoded[..., i]
                     unique = np.unique(layer)
                     print(f"  Channel {i} (Bit {bit_values[i]}): Unique values {unique}")
                     if not np.all(np.isin(unique, [0.0, 1.0])):
                         print(f"    [FAIL] Channel {i} contains non-binary values.")
                     else:
                         print(f"    [PASS] Channel {i} is binary.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Data loading failed with exception: {e}")

    # 6. Check Model (With Sigmoid)
    print("\nChecking Model...")
    try:
        model = create_unet_model(
            input_shape=(config.data.img_size[0], config.data.img_size[1], 3),
            num_classes=config.model.num_classes
        )
        print(f"[PASS] Model instantiated: {model.name}")
        model.summary(print_fn=lambda x: None) # Suppress output, just check if it runs
        print("[PASS] Model summary generated (structure is valid).")
        
    except Exception as e:
        print(f"[ERROR] Model instantiation failed: {e}")

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    verify_segmentation()

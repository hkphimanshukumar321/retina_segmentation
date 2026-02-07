
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# Setup paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from segmentation.config import SegmentationConfig
from common.data_loader import load_image, decode_bitmask

def visualize_results(model_path, output_dir="results/viz"):
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Try custom objects if needed (usually not for standard UNet)
        return

    config = SegmentationConfig()
    data_dir = config.data.data_dir
    images_dir = data_dir / config.data.img_dir
    masks_dir = data_dir / config.data.mask_dir
    
    # Get all validation images (simple split logic from run.py: last 20%)
    image_files = sorted(list(images_dir.glob("*.[jp][pn][g]")))
    mask_files = sorted(list(masks_dir.glob("*.[jp][pn][g]")))
    
    split_idx = int(len(image_files) * 0.8)
    val_imgs = image_files[split_idx:]
    val_masks = mask_files[split_idx:]
    
    print(f"Found {len(val_imgs)} validation images.")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process a few samples
    indices = np.linspace(0, len(val_imgs)-1, 5, dtype=int)
    
    for idx in indices:
        img_path = val_imgs[idx]
        mask_path = val_masks[idx]
        filename = img_path.name
        
        # Load and Preprocess
        # Note: Model expects 128x128 patches or resized images. 
        # For visualization, we'll resize the full image to 512x512 for better visibility
        # or keep it 1024x1024 if the model is fully convolutional and can handle it.
        # UNet usually handles any size divisible by 16/32. 
        # Let's try 512x512 to start.
        
        target_size = (512, 512)
        
        # 1. Image
        original_img = cv2.imread(str(img_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        inp_img = cv2.resize(original_img, target_size)
        inp = inp_img.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, axis=0) # (1, 512, 512, 3)
        
        # 2. Ground Truth
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        raw_mask = cv2.resize(raw_mask, target_size, interpolation=cv2.INTER_NEAREST)
        gt_decoded = decode_bitmask(raw_mask, config.model.num_classes, config.data.bit_values)
        
        # 3. Predict
        try:
            pred = model.predict(inp, verbose=0) # (1, 512, 512, 3)
            pred = pred[0]
        except Exception:
            # Fallback for shape mismatch (if model has fixed input size)
            # Resize to trained size (128x128) then upscale
            train_size = config.data.img_size
            inp_small = cv2.resize(original_img, train_size).astype(np.float32) / 255.0
            inp_small = np.expand_dims(inp_small, axis=0)
            pred_small = model.predict(inp_small, verbose=0)[0]
            pred = cv2.resize(pred_small, target_size)

        # 4. Visualize Side-by-Side
        # Create a collage: [Original] [GT Class 0] [Pred Class 0]
        #                   [          ] [GT Class 1] [Pred Class 1] ...
        
        plt.figure(figsize=(15, 5 * config.model.num_classes))
        
        # Row 0: Original Image
        plt.subplot(config.model.num_classes + 1, 3, 2)
        plt.imshow(inp_img)
        plt.title(f"Original: {filename}")
        plt.axis('off')
        
        for i in range(config.model.num_classes):
            # GT
            plt.subplot(config.model.num_classes + 1, 3, 4 + (i*3))
            plt.imshow(gt_decoded[..., i], cmap='gray', vmin=0, vmax=1)
            plt.title(f"GT Class {i} (Bit {config.data.bit_values[i]})")
            plt.axis('off')
            
            # Pred
            plt.subplot(config.model.num_classes + 1, 3, 5 + (i*3))
            plt.imshow(pred[..., i], cmap='jet', vmin=0, vmax=1)
            plt.title(f"Pred Class {i}")
            plt.axis('off')
            
            # Pred Thresholded
            plt.subplot(config.model.num_classes + 1, 3, 6 + (i*3))
            plt.imshow(pred[..., i] > 0.5, cmap='gray', vmin=0, vmax=1)
            plt.title(f"Pred > 0.5")
            plt.axis('off')

        save_path = os.path.join(output_dir, f"viz_{filename}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    # Check for best model first, then final
    if Path("segmentation/results/best_model.keras").exists():
        model_p = "segmentation/results/best_model.keras"
    else:
        model_p = "segmentation/results/final_model.keras"
        
    visualize_results(model_p)

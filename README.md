# Retina Segmentation — Ghost-CAS-UNet v2

Lightweight semantic segmentation for **diabetic retinopathy lesion detection** on fundus images.

> **Architecture**: Ghost-CAS-UNet v2 — Ghost Modules + Coordinate Attention + Attention Gates + DW-ASPP + Deep Supervision

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train main model (Ghost-CAS-UNet v2)
python segmentation/run.py --quick          # smoke test (1 epoch)
python segmentation/run.py                  # full training

# 3. Train baselines (DeepLabV3+, U-Net, LinkNet, FPN)
pip install segmentation-models             # one-time
python segmentation/run.py --baselines

# 4. Run ablation study
python segmentation/run.py --ablation

# 5. Analyse & compare results
python segmentation/analysis.py

# 6. Export for edge deployment
python segmentation/export_model.py --quantize float16
```

## Architecture

| Component | Role | Benefit |
|-----------|------|---------|
| **Ghost Module** | Replace standard conv → cheap linear ops | 2× parameter reduction |
| **Coordinate Attention** | Channel + spatial attention in encoder | Better feature selection |
| **Attention Gate** | Gate skip connections before fusion | Suppress noisy features |
| **DW-ASPP** | Multi-scale depthwise convs at bottleneck | Capture lesions at all scales |
| **Deep Supervision** | Auxiliary losses at decoder stages | Faster convergence |

## Directory Structure

```
retina_scan/
├── common/                     # Shared utilities
│   ├── data_loader.py          # PatchDataGenerator, augmentations
│   ├── ablation.py             # BaseAblationStudy framework
│   ├── hardware.py             # GPU setup
│   └── logger.py               # Logging
│
├── data/
│   ├── segmentation/           # Local dataset (Images/ + Labels/)
│   └── IDRID/                  # IDRiD dataset (auto-detected)
│
├── segmentation/
│   ├── config.py               # SegmentationConfig (central config)
│   ├── run.py                  # Master runner (main/ablation/baselines)
│   ├── baselines.py            # DeepLabV3+, U-Net, LinkNet, FPN
│   ├── analysis.py             # Comparison tables, statistical tests
│   ├── export_model.py         # TFLite / ONNX conversion
│   ├── src/
│   │   ├── models.py           # Ghost-CAS-UNet v2 + variants
│   │   ├── losses.py           # Lovász-Softmax, Focal Tversky, Combined
│   │   ├── metrics.py          # DiceScore, IoUScore
│   │   └── idrid_loader.py     # IDRiD-specific data generator
│   ├── experiments/
│   │   ├── pilot_test.py       # Single experiment runner
│   │   ├── benchmark_ghost.py  # Model efficiency benchmarks
│   │   └── run_iou_analysis.py # Per-class metric analysis
│   └── results/                # Training outputs
│       ├── main_*              # Main model results
│       ├── ablation/           # Ablation study results
│       └── baselines/          # Baseline comparison results
│
└── requirements.txt
```

## Training Pipeline

1. **Data** — Train on IDRiD (auto-downloaded), validate on local dataset
2. **Train** — `run.py` with combined loss (Lovász + Focal Tversky + BCE)
3. **Test** — Evaluate on IDRiD test set with per-class IoU/Dice/Sensitivity
4. **Compare** — `analysis.py` generates LaTeX tables + Wilcoxon significance tests
5. **Export** — `export_model.py` converts to TFLite with optional float16/int8 quantization

## Baselines

| Model | Encoder | Pretrained | Library |
|-------|---------|------------|---------|
| DeepLabV3+ | ResNet50 | ImageNet | tf.keras.applications |
| U-Net | ResNet34 | ImageNet | segmentation-models |
| LinkNet | ResNet34 | ImageNet | segmentation-models |
| FPN | ResNet34 | ImageNet | segmentation-models |

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
- OpenCV, tqdm, psutil
- `segmentation-models` (for baselines)
- `scipy` (for statistical tests)

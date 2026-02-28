# -*- coding: utf-8 -*-
"""
export_for_rpi.py — Export all models as FULL .h5 for Raspberry Pi
===================================================================
Run this ONCE on the training machine.
It rebuilds each model architecture, loads the trained weights,
and saves a FULL (architecture + weights) .h5 file that can be loaded
on any device with just:  tf.keras.models.load_model('model.h5')

NOTE: Ghost-CAS-UNet uses custom layers. The custom layers file
      (rpi_custom_layers.py) is also exported to the deploy folder.
      On the RPi, you must pass custom_objects when loading Ghost model.

USAGE:
    python export_for_rpi.py
"""

import os, sys, shutil, gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()         # experiments/
SEG_DIR    = SCRIPT_DIR.parent.resolve()             # segmentation/
ROOT       = SEG_DIR.parent.resolve()                # retina_scan/

sys.path.insert(0, str(ROOT))   # so 'segmentation' package is importable

H5_DIR  = ROOT / '.h5'
DEPLOY  = ROOT / 'rpi_deploy'
DEPLOY.mkdir(exist_ok=True)

NUM_CLASSES = 5
RESOLUTION  = 256  # match training resolution

print(f"\n{'='*58}")
print("  EXPORTING MODELS FOR RPi DEPLOYMENT")
print(f"  Output folder: {DEPLOY}")
print(f"{'='*58}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. DeepLabV3+ ResNet50
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[1/3] DeepLabV3+ ResNet50 ...")
from segmentation.baselines import BASELINE_MODELS

model = BASELINE_MODELS['deeplabv3plus_resnet50'](
    input_shape=(RESOLUTION, RESOLUTION, 3), num_classes=NUM_CLASSES
)
wp = H5_DIR / 'deeplabv3plus_resnet50_final.weights.h5'
model.load_weights(str(wp))
out = DEPLOY / 'deeplabv3plus_resnet50_full.h5'
model.save(str(out), save_format='h5')
print(f"  ✓ Saved: {out.name}  ({out.stat().st_size/(1024*1024):.1f} MB)")
del model
import gc; gc.collect()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. SM-UNet ResNet34
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[2/3] SM-UNet ResNet34 ...")
os.environ['SM_FRAMEWORK'] = 'tf.keras'
# encoder_weights=None so we don't download ImageNet (our weights file has full weights)
model = BASELINE_MODELS['sm_unet_resnet34'](
    input_shape=(RESOLUTION, RESOLUTION, 3), num_classes=NUM_CLASSES, encoder_weights=None
)
wp = H5_DIR / 'sm_unet_resnet34_best.weights.h5'
model.load_weights(str(wp))
out = DEPLOY / 'sm_unet_resnet34_full.h5'
model.save(str(out), save_format='h5')
print(f"  ✓ Saved: {out.name}  ({out.stat().st_size/(1024*1024):.1f} MB)")
del model; gc.collect()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Ghost-CAS-UNet v2 (Proposed model)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[3/3] Ghost-CAS-UNet v2 (Proposed) ...")
from segmentation.src.models import SEGMENTATION_MODELS

model = SEGMENTATION_MODELS['ghost_cas_unet_v2'](
    input_shape=(RESOLUTION, RESOLUTION, 3),
    num_classes=NUM_CLASSES,
    encoder_filters=[32, 64, 128, 256],
    dropout_rate=0.15,
    ghost_ratio=2,
    use_skip_attention=True,
    use_aspp=True,
    deep_supervision=True,
)
wp = H5_DIR / 'pilot_best.weights.h5'
model.load_weights(str(wp))
out = DEPLOY / 'ghost_cas_unet_v2_full.h5'
model.save(str(out), save_format='h5')
print(f"  ✓ Saved: {out.name}  ({out.stat().st_size/(1024*1024):.1f} MB)")
del model; gc.collect()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Copy benchmark scripts into rpi_deploy/
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
scripts_dir = Path(__file__).parent
for script in ['rpi_benchmark.py', 'batch_benchmark.py']:
    src = scripts_dir / script
    if src.exists():
        shutil.copy(src, DEPLOY / script)
        print(f"\n  ✓ Copied: {script}")

# Update batch_benchmark.py's DEFAULT_SEARCH_DIR to point to rpi_deploy itself
batch_file = DEPLOY / 'batch_benchmark.py'
if batch_file.exists():
    content = batch_file.read_text()
    content = content.replace(
        'DEFAULT_SEARCH_DIR = Path(__file__).parent.parent / "results"',
        'DEFAULT_SEARCH_DIR = Path(__file__).parent  # scan this folder'
    )
    batch_file.write_text(content)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Write a README
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
readme = DEPLOY / 'README.txt'
readme.write_text("""RPi Deploy Folder — Retinal Lesion Segmentation Benchmark
============================================================

FILES:
  deeplabv3plus_resnet50_full.h5   — Baseline 1: DeepLabV3+ ResNet50
  sm_unet_resnet34_full.h5         — Baseline 2: SM-UNet ResNet34
  ghost_cas_unet_v2_full.h5        — Proposed: Ghost-CAS-UNet v2
  rpi_benchmark.py                 — Single model benchmark
  batch_benchmark.py               — Run ALL models in this folder

INSTALL ON RPi:
  pip install tensorflow

BENCHMARK ALL MODELS (one shot):
  python batch_benchmark.py

BENCHMARK ONE MODEL:
  python rpi_benchmark.py --model deeplabv3plus_resnet50_full.h5
  python rpi_benchmark.py --model ghost_cas_unet_v2_full.h5 --runs 50

NOTE: ghost_cas_unet_v2_full.h5 uses custom layers. If loading fails,
      provide custom_objects when calling tf.keras.models.load_model().
""")
print(f"\n  ✓ Written: README.txt")

print(f"\n{'='*58}")
print("  ALL DONE!")
print(f"  Deploy folder ready at: {DEPLOY}")
print(f"{'='*58}")

RPi Deploy Folder — Retinal Lesion Segmentation Benchmark
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

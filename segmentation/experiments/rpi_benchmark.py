# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
RPi Inference Benchmark
=======================

Rebuilds the Ghost-CAS-UNet architecture and loads saved weights,
then measures CPU-only inference latency.

Designed to run identically on a local PC (baseline) and on a Raspberry Pi.

Usage (local baseline):
    python -m segmentation.experiments.rpi_benchmark

    python -m segmentation.experiments.rpi_benchmark \
        --weights results/pilot/pilot_model.weights.h5 \
        --runs 100 \
        --resolution 128

Usage (on RPi — copy this file + weights + src/models.py):
    python rpi_benchmark.py --weights pilot_model.weights.h5 --runs 50

Outputs → inference_benchmark.json
"""

import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
SEG_DIR    = SCRIPT_DIR.parent.resolve()
ROOT_DIR   = SEG_DIR.parent.resolve()

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SEG_DIR))

# ===========================================================================
# Defaults (must match pilot_test.py sweet-spot config)
# ===========================================================================
DEFAULT_WEIGHTS   = SEG_DIR / "results" / "pilot" / "pilot_model.weights.h5"
DEFAULT_KERAS     = SEG_DIR / "results" / "pilot" / "pilot_model.keras"
DEFAULT_RESOLUTION = 128
DEFAULT_NUM_RUNS   = 100
DEFAULT_WARMUP     = 10

# Architecture config (matches pilot_test.py sweet-spot)
NUM_CLASSES     = 3
ENCODER_FILTERS = [16, 32, 64, 128]
GHOST_RATIO     = 2
DROPOUT         = 0.15
USE_SKIP_ATTN   = True


def main(
    weights_path: str = None,
    resolution: int = DEFAULT_RESOLUTION,
    n_runs: int = DEFAULT_NUM_RUNS,
    n_warmup: int = DEFAULT_WARMUP,
    output_dir: str = None,
):
    import tensorflow as tf

    # ---- Force CPU (simulate RPi) ----
    tf.config.set_visible_devices([], "GPU")
    print("[*] GPU disabled — running on CPU only")

    # ---- Resolve weights path ----
    weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS

    # Fallback: if user pointed to .keras file, check if .weights.h5 exists
    if weights_path.suffix == ".keras":
        w_alt = weights_path.with_suffix(".weights.h5")
        if w_alt.exists():
            weights_path = w_alt
        else:
            print(f"[WARN] .keras file given but no .weights.h5 found next to it.")
            print(f"       Tried: {w_alt}")

    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        print("        Run pilot_test.py first to generate the weights.")
        return False

    print(f"\n{'='*60}")
    print("  RPi INFERENCE BENCHMARK")
    print(f"{'='*60}")
    print(f"  Weights    : {weights_path.name}")
    print(f"  Resolution : {resolution}×{resolution}")
    print(f"  Warmup     : {n_warmup}")
    print(f"  Runs       : {n_runs}")

    # ---- Rebuild model architecture ----
    from segmentation.src.models import SEGMENTATION_MODELS

    model_fn = SEGMENTATION_MODELS["ghost_ca_unet"]
    model = model_fn(
        input_shape=(resolution, resolution, 3),
        num_classes=NUM_CLASSES,
        encoder_filters=ENCODER_FILTERS,
        dropout_rate=DROPOUT,
        ghost_ratio=GHOST_RATIO,
        use_skip_attention=USE_SKIP_ATTN,
    )

    # ---- Load weights ----
    model.load_weights(str(weights_path))
    print("[*] Weights loaded successfully")

    total_params  = model.count_params()
    model_size_mb = weights_path.stat().st_size / (1024 * 1024)

    # Also report .keras size if available
    keras_path = weights_path.with_suffix("").with_suffix(".keras")
    if keras_path.exists():
        model_size_mb = keras_path.stat().st_size / (1024 * 1024)

    print(f"  Parameters : {total_params:,}")
    print(f"  Model Size : {model_size_mb:.3f} MB")
    print(f"{'='*60}\n")

    input_shape = (1, resolution, resolution, 3)
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # ---- Warmup ----
    print(f"[*] Warming up ({n_warmup} passes) …")
    with tf.device("/CPU:0"):
        for _ in range(n_warmup):
            model.predict(dummy_input, verbose=0)

    # ---- Timed runs ----
    print(f"[*] Benchmarking ({n_runs} passes) …")
    times_ms = []
    with tf.device("/CPU:0"):
        for i in range(n_runs):
            start = time.perf_counter()
            model.predict(dummy_input, verbose=0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed_ms)
            if (i + 1) % 25 == 0:
                print(f"    [{i+1}/{n_runs}]  {elapsed_ms:.2f} ms")

    times_ms = np.array(times_ms)
    avg_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    min_ms = float(np.min(times_ms))
    max_ms = float(np.max(times_ms))
    p50_ms = float(np.percentile(times_ms, 50))
    p95_ms = float(np.percentile(times_ms, 95))
    fps    = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    # ---- Results ----
    results = {
        "model_name": "Ghost_CAS_UNet",
        "weights_file": weights_path.name,
        "resolution": f"{resolution}x{resolution}",
        "total_params": int(total_params),
        "model_size_mb": round(model_size_mb, 3),
        "num_runs": n_runs,
        "inference_ms": {
            "mean": round(avg_ms, 2),
            "std": round(std_ms, 2),
            "min": round(min_ms, 2),
            "max": round(max_ms, 2),
            "p50": round(p50_ms, 2),
            "p95": round(p95_ms, 2),
        },
        "fps": round(fps, 1),
        "device": "CPU",
    }

    # ---- Save ----
    out_dir = Path(output_dir) if output_dir else weights_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inference_benchmark.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---- Print ----
    print(f"\n{'='*60}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Model       : Ghost_CAS_UNet")
    print(f"  Parameters  : {total_params:,}")
    print(f"  Size        : {model_size_mb:.3f} MB")
    print(f"  Inference   : {avg_ms:.2f} ± {std_ms:.2f} ms")
    print(f"  P50 / P95   : {p50_ms:.2f} / {p95_ms:.2f} ms")
    print(f"  FPS         : {fps:.1f}")
    print(f"{'='*60}")
    print(f"\n  Saved → {out_path}")

    return True


# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPi Inference Benchmark")
    parser.add_argument("--weights", type=str, default=None,
                        help=f"Path to .weights.h5 file (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                        help=f"Input resolution (default: {DEFAULT_RESOLUTION})")
    parser.add_argument("--runs", type=int, default=DEFAULT_NUM_RUNS,
                        help=f"Number of inference runs (default: {DEFAULT_NUM_RUNS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"Number of warmup passes (default: {DEFAULT_WARMUP})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    try:
        success = main(
            weights_path=args.weights,
            resolution=args.resolution,
            n_runs=args.runs,
            n_warmup=args.warmup,
            output_dir=args.output,
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)

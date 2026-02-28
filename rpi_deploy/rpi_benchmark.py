# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# ==============================================================================
"""
RPi Standalone Inference Benchmark
====================================
Copy ONLY this file + your .keras/.h5 model file to the Raspberry Pi.
No other project files needed. Just TensorFlow must be installed.

USAGE:
  # Option 1: Edit MODEL_PATH below and run directly
  python rpi_benchmark.py

  # Option 2: Pass the model path as an argument
  python rpi_benchmark.py --model path/to/model.keras
  python rpi_benchmark.py --model path/to/model.keras --runs 50 --resolution 256

  # Option 3: If model is in the same folder as this script
  python rpi_benchmark.py --model model_filename.keras

INSTALL REQUIREMENTS ON RPi:
  pip install tensorflow
  pip install psutil          # optional, for better memory measurement

OUTPUT:
  Prints latency + memory stats to screen.
  Saves results to <model_name>_benchmark.json in the same folder.
"""

import sys
import json
import time
import argparse
import traceback
from pathlib import Path

import numpy as np

# ===========================================================================
# ▼▼▼ EDIT THIS IF YOU DON'T WANT TO USE --model ARGUMENT ▼▼▼
# Set to the full path of your .keras or .h5 model file
# Use None to require the --model argument
# ===========================================================================
MODEL_PATH = None          # Example: "deeplabv3plus_resnet50_final.keras"
                           # Example: "/home/pi/models/ghost_cas_unet.keras"

# ===========================================================================
# Default benchmark settings (override with command-line args if needed)
# ===========================================================================
DEFAULT_RESOLUTION = 256   # Change if your model uses a different input size
DEFAULT_NUM_RUNS   = 100
DEFAULT_WARMUP     = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_memory_mb() -> float:
    """Return current process memory usage in MB."""
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB→MB on Linux
    except Exception:
        return -1.0


def load_any_model(model_path: Path):
    """
    Load a Keras model from .keras or .h5 file.
    Handles Lambda/custom-layer models by extracting embedded weights.
    Does NOT require any project code.
    """
    import tensorflow as tf

    suffix = model_path.suffix.lower()

    # ------------------------------------------------------------------ .h5
    if suffix == ".h5":
        print(f"[*] Loading H5 model: {model_path.name}")
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
            print("[*] Loaded OK.")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load H5 model: {e}")

    # ---------------------------------------------------------------- .keras
    if suffix == ".keras":
        print(f"[*] Loading .keras model: {model_path.name}")

        # Attempt 1: plain load (works for standard models)
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
            print("[*] Loaded OK.")
            return model
        except Exception:
            pass

        # Attempt 2: safe_mode=False
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
            print("[*] Loaded OK (safe_mode=False).")
            return model
        except Exception:
            pass

        # Attempt 3: .keras is a ZIP. Extract model.weights.h5, load as weights-only
        # into a newly rebuilt architecture from the config stored inside the zip.
        print("[!] Standard load failed (likely Lambda/custom layers).")
        print("[*] Trying to extract & load weights from inside the .keras archive ...")
        import zipfile, tempfile, json as _json

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with zipfile.ZipFile(str(model_path), 'r') as zf:
                zf.extractall(str(tmpdir))

            config_file = tmpdir / "config.json"
            weights_file = tmpdir / "model.weights.h5"

            if not config_file.exists() or not weights_file.exists():
                raise RuntimeError(
                    f"Cannot load {model_path.name}.\n"
                    "The model uses custom layers that cannot be deserialized on this device.\n"
                    "FIX: On your training machine, re-save the model after removing Lambda layers,\n"
                    "     then copy the new .keras file to the RPi."
                )

            # Reconstruct model from the serialized config
            with open(config_file) as f:
                cfg = _json.load(f)

            try:
                model = tf.keras.models.model_from_json(_json.dumps(cfg["config"]))
            except Exception as e:
                raise RuntimeError(
                    f"Could not reconstruct model from config: {e}\n"
                    "The model has custom objects that block deserialization on this device."
                )

            model.load_weights(str(weights_file))
            print("[*] Model reconstructed from config + weights. OK.")
            return model

    raise ValueError(f"Unsupported file type: {suffix}. Use .keras or .h5")


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------
def run_benchmark(
    model_path: str,
    resolution: int  = DEFAULT_RESOLUTION,
    n_runs: int      = DEFAULT_NUM_RUNS,
    n_warmup: int    = DEFAULT_WARMUP,
    output_dir: str  = None,
) -> bool:
    import tensorflow as tf

    # Force CPU only — simulates/matches RPi hardware
    tf.config.set_visible_devices([], "GPU")
    print("[*] GPU disabled — CPU only (RPi mode)")

    model_path = Path(model_path).resolve()
    if not model_path.exists():
        print(f"[ERROR] File not found: {model_path}")
        print("        Make sure the model file is in the same folder as this script,")
        print("        or pass the full path with --model.")
        return False

    print(f"\n{'='*58}")
    print("  RPi INFERENCE BENCHMARK")
    print(f"{'='*58}")
    print(f"  Model      : {model_path.name}")
    print(f"  Resolution : {resolution} x {resolution}")
    print(f"  Warmup     : {n_warmup} passes")
    print(f"  Runs       : {n_runs} passes")
    print(f"{'='*58}\n")

    # --- Load model ---
    mem_baseline = get_memory_mb()
    try:
        model = load_any_model(model_path)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        return False

    mem_after_load = get_memory_mb()
    load_mem_delta = max(0.0, mem_after_load - mem_baseline)

    total_params  = model.count_params()
    file_size_mb  = model_path.stat().st_size / (1024 * 1024)

    print(f"\n  Parameters : {total_params:,}")
    print(f"  File size  : {file_size_mb:.2f} MB")
    print(f"  Mem (load) : +{load_mem_delta:.1f} MB\n")

    # --- Prepare dummy input ---
    dummy = np.random.rand(1, resolution, resolution, 3).astype(np.float32)

    # --- Warmup ---
    print(f"[*] Warmup ({n_warmup} passes) ...")
    with tf.device("/CPU:0"):
        for _ in range(n_warmup):
            model.predict(dummy, verbose=0)

    # --- Timed runs ---
    print(f"[*] Benchmarking ({n_runs} passes) ...")
    times = []
    peak_mem = get_memory_mb()

    with tf.device("/CPU:0"):
        for i in range(n_runs):
            t0 = time.perf_counter()
            model.predict(dummy, verbose=0)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)
            peak_mem = max(peak_mem, get_memory_mb())
            if n_runs <= 20 or (i + 1) % 25 == 0:
                print(f"    [{i+1:3d}/{n_runs}]  {elapsed_ms:.1f} ms")

    times = np.array(times)
    avg   = float(np.mean(times))
    std   = float(np.std(times))
    p50   = float(np.percentile(times, 50))
    p95   = float(np.percentile(times, 95))
    fps   = round(1000.0 / avg, 1) if avg > 0 else 0.0

    # --- Print results ---
    print(f"\n{'='*58}")
    print("  RESULTS")
    print(f"{'='*58}")
    print(f"  Model         : {model_path.stem}")
    print(f"  Parameters    : {total_params:,}")
    print(f"  File size     : {file_size_mb:.2f} MB")
    print(f"  Mem footprint : ~{load_mem_delta:.0f} MB  (load delta)")
    print(f"  Peak mem      : ~{peak_mem:.0f} MB  (during inference)")
    print(f"  Avg latency   : {avg:.1f} ± {std:.1f} ms")
    print(f"  P50 / P95     : {p50:.1f} / {p95:.1f} ms")
    print(f"  FPS           : {fps}")
    print(f"{'='*58}")

    # --- Save JSON ---
    results = {
        "model": model_path.stem,
        "file": model_path.name,
        "resolution": f"{resolution}x{resolution}",
        "total_params": int(total_params),
        "file_size_mb": round(file_size_mb, 2),
        "device": "CPU",
        "num_runs": n_runs,
        "inference_ms": {
            "mean": round(avg, 2),
            "std": round(std, 2),
            "min": round(float(np.min(times)), 2),
            "max": round(float(np.max(times)), 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
        },
        "fps": fps,
        "mem_load_delta_mb": round(load_mem_delta, 1),
        "mem_peak_mb": round(peak_mem, 1),
    }

    out_dir  = Path(output_dir).resolve() if output_dir else model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_path.stem}_benchmark.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved → {out_path}\n")
    return True


# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone RPi Benchmark — no project code needed. Just TensorFlow."
    )
    parser.add_argument(
        "--model", "-m", type=str, default=MODEL_PATH,
        help="Path to .keras or .h5 model file. "
             f"(default: MODEL_PATH variable = '{MODEL_PATH}')"
    )
    parser.add_argument(
        "--resolution", "-r", type=int, default=DEFAULT_RESOLUTION,
        help=f"Input patch resolution H=W (default: {DEFAULT_RESOLUTION})"
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_NUM_RUNS,
        help=f"Number of timed inference passes (default: {DEFAULT_NUM_RUNS})"
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Number of warmup passes (default: {DEFAULT_WARMUP})"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Directory to save the JSON results (default: same folder as model)"
    )
    args = parser.parse_args()

    if args.model is None:
        print("[ERROR] No model path provided.")
        print("        Either set MODEL_PATH at the top of this script,")
        print("        or use:  python rpi_benchmark.py --model your_model.keras")
        sys.exit(1)

    try:
        ok = run_benchmark(
            model_path=args.model,
            resolution=args.resolution,
            n_runs=args.runs,
            n_warmup=args.warmup,
            output_dir=args.output,
        )
        sys.exit(0 if ok else 1)
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
        sys.exit(1)

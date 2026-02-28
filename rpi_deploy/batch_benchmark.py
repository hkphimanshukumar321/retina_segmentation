# -*- coding: utf-8 -*-
"""
batch_benchmark.py — CPU Benchmark Runner (ALL models in one shot)
===================================================================
Scans a directory for ALL .keras and .h5 model files and benchmarks
each one on CPU-only mode. Prints a comparison table + saves JSON.

USAGE:
    # Scan default folder (results/ next to this script)
    python batch_benchmark.py

    # Scan a specific folder
    python batch_benchmark.py --dir path/to/models/

    # Custom settings
    python batch_benchmark.py --dir path/to/models/ --runs 50 --resolution 256

REQUIREMENTS:
    pip install tensorflow
    pip install psutil          # optional, for better memory accuracy
"""

import sys, json, time, argparse, traceback
from pathlib import Path
import numpy as np

# ===========================================================================
# ▼ Edit these defaults if not passing command-line args ▼
# ===========================================================================
DEFAULT_SEARCH_DIR = Path(__file__).parent  # scan this folder  # segmentation/results/
DEFAULT_RESOLUTION = 256    # H = W input size
DEFAULT_NUM_RUNS   = 50
DEFAULT_WARMUP     = 5
# ===========================================================================


def get_memory_mb() -> float:
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return -1.0


def try_load_model(model_path: Path):
    """Try to load a .keras or .h5 model. Returns (model, error_str)."""
    import tensorflow as tf

    path_str = str(model_path)

    # For Ghost model: load with custom objects
    if 'ghost' in model_path.name.lower():
        import sys as _sys
        _here = str(model_path.parent)
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        try:
            from ghost_custom_layers import GHOST_CUSTOM_OBJECTS
            with tf.keras.utils.custom_object_scope(GHOST_CUSTOM_OBJECTS):
                model = tf.keras.models.load_model(path_str, compile=False)
            return model, None
        except Exception as e:
            return None, str(e)

    # Standard load
    try:
        model = tf.keras.models.load_model(path_str, compile=False)
        return model, None
    except Exception as e1:
        pass

    # safe_mode=False fallback for .keras
    if model_path.suffix.lower() == '.keras':
        try:
            model = tf.keras.models.load_model(path_str, compile=False, safe_mode=False)
            return model, None
        except Exception as e2:
            return None, str(e2)

    return None, str(e1)


def benchmark_one(model_path: Path, resolution: int, n_runs: int, n_warmup: int) -> dict:
    """Benchmark a single model. Returns a results dict."""
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    name = model_path.stem
    file_size_mb = model_path.stat().st_size / (1024 * 1024)

    print(f"\n  {'─'*54}")
    print(f"  Model : {model_path.name}")
    print(f"  Size  : {file_size_mb:.2f} MB")

    mem_before = get_memory_mb()
    model, err = try_load_model(model_path)

    if model is None:
        print(f"  [SKIP] Load failed: {err[:120]}")
        return {
            "model": name, "file": model_path.name,
            "status": "FAILED", "error": err[:200],
            "file_size_mb": round(file_size_mb, 2),
        }

    mem_after_load = get_memory_mb()
    load_mem = max(0.0, mem_after_load - mem_before)
    total_params = model.count_params()
    print(f"  Params: {total_params:,}   Load mem delta: +{load_mem:.0f} MB")

    dummy = np.random.rand(1, resolution, resolution, 3).astype(np.float32)

    # Warmup
    with tf.device("/CPU:0"):
        for _ in range(n_warmup):
            model.predict(dummy, verbose=0)

    # Timed runs
    times = []
    peak_mem = get_memory_mb()
    with tf.device("/CPU:0"):
        for i in range(n_runs):
            t0 = time.perf_counter()
            model.predict(dummy, verbose=0)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)
            peak_mem = max(peak_mem, get_memory_mb())
            if (i + 1) % 10 == 0 or n_runs <= 10:
                print(f"    [{i+1:3d}/{n_runs}]  {elapsed_ms:.1f} ms")

    times = np.array(times)
    avg = float(np.mean(times))
    std = float(np.std(times))

    print(f"\n  Avg  : {avg:.1f} ± {std:.1f} ms   |   FPS: {1000/avg:.1f}   |   Peak mem: ~{peak_mem:.0f} MB")

    # Free model to avoid OOM on next iteration
    del model
    import gc; gc.collect()

    return {
        "model": name,
        "file": model_path.name,
        "status": "OK",
        "resolution": f"{resolution}x{resolution}",
        "total_params": int(total_params),
        "file_size_mb": round(file_size_mb, 2),
        "inference_ms": {
            "mean": round(avg, 2),
            "std": round(std, 2),
            "min": round(float(np.min(times)), 2),
            "max": round(float(np.max(times)), 2),
            "p50": round(float(np.percentile(times, 50)), 2),
            "p95": round(float(np.percentile(times, 95)), 2),
        },
        "fps": round(1000.0 / avg, 1),
        "mem_load_delta_mb": round(load_mem, 1),
        "mem_peak_mb": round(peak_mem, 1),
    }


def main(search_dir: str, resolution: int, n_runs: int, n_warmup: int, output_dir: str = None):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    print("[*] GPU disabled — running CPU only")

    search_dir = Path(search_dir).resolve()
    if not search_dir.exists():
        print(f"[ERROR] Directory not found: {search_dir}")
        return

    # Find all model files recursively
    model_files = sorted(
        list(search_dir.rglob("*.keras")) + list(search_dir.rglob("*.h5"))
    )

    if not model_files:
        print(f"[WARN] No .keras or .h5 files found in: {search_dir}")
        return

    print(f"\n{'='*58}")
    print("  CPU BATCH BENCHMARK")
    print(f"{'='*58}")
    print(f"  Scan dir   : {search_dir}")
    print(f"  Found      : {len(model_files)} model file(s)")
    print(f"  Resolution : {resolution}x{resolution}")
    print(f"  Warmup     : {n_warmup} passes")
    print(f"  Runs       : {n_runs} passes per model")
    print(f"{'='*58}")

    all_results = []
    for model_path in model_files:
        result = benchmark_one(model_path, resolution, n_runs, n_warmup)
        all_results.append(result)

    # Summary table
    print(f"\n\n{'='*58}")
    print("  SUMMARY TABLE")
    print(f"{'='*58}")
    header = f"  {'Model':<45} {'Avg(ms)':>8} {'FPS':>6} {'PeakMem':>10}"
    print(header)
    print(f"  {'─'*54}")
    for r in all_results:
        if r["status"] == "OK":
            name = r["model"][:44]
            avg  = r["inference_ms"]["mean"]
            fps  = r["fps"]
            mem  = r["mem_peak_mb"]
            print(f"  {name:<45} {avg:>8.1f} {fps:>6.1f} {mem:>9.0f}MB")
        else:
            print(f"  {r['model'][:44]:<45} {'FAILED':>8}")
    print(f"{'='*58}\n")

    # Save combined JSON
    out_dir = Path(output_dir).resolve() if output_dir else search_dir
    out_path = out_dir / "batch_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({"device": "CPU", "resolution": f"{resolution}x{resolution}",
                   "runs": n_runs, "results": all_results}, f, indent=2)
    print(f"  Full results saved → {out_path}\n")


# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU Batch Benchmark — all .keras/.h5 models in a folder.")
    parser.add_argument("--dir",  "-d", type=str, default=str(DEFAULT_SEARCH_DIR),
                        help=f"Directory to scan (default: {DEFAULT_SEARCH_DIR})")
    parser.add_argument("--resolution", "-r", type=int, default=DEFAULT_RESOLUTION,
                        help=f"Input resolution H=W (default: {DEFAULT_RESOLUTION})")
    parser.add_argument("--runs",    type=int, default=DEFAULT_NUM_RUNS,
                        help=f"Timed runs per model (default: {DEFAULT_NUM_RUNS})")
    parser.add_argument("--warmup",  type=int, default=DEFAULT_WARMUP,
                        help=f"Warmup passes (default: {DEFAULT_WARMUP})")
    parser.add_argument("--output",  "-o", type=str, default=None,
                        help="Output dir for JSON (default: same as --dir)")
    args = parser.parse_args()

    main(
        search_dir=args.dir,
        resolution=args.resolution,
        n_runs=args.runs,
        n_warmup=args.warmup,
        output_dir=args.output,
    )

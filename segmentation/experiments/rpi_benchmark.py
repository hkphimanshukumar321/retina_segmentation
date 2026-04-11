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
import tensorflow as tf


def _build_local_custom_objects():
    """Provide custom layer classes even when the project package is unavailable."""

    class GhostModule(tf.keras.layers.Layer):
        def __init__(self, filters, kernel_size=1, ratio=2, dw_kernel=3, activation='relu', **kwargs):
            super().__init__(**kwargs)
            self.filters = filters
            self.ratio = ratio
            self.primary_filters = max(1, filters // ratio)
            self.ghost_filters = filters - self.primary_filters
            self.primary_conv = tf.keras.layers.Conv2D(self.primary_filters, kernel_size, padding='same', use_bias=False)
            self.primary_gn = tf.keras.layers.GroupNormalization(groups=min(8, self.primary_filters), axis=-1)
            self.primary_act = tf.keras.layers.Activation(activation)
            self.ghost_dw = tf.keras.layers.DepthwiseConv2D(dw_kernel, padding='same', use_bias=False)
            self.ghost_gn = tf.keras.layers.GroupNormalization(groups=min(8, self.primary_filters), axis=-1)
            self.ghost_act = tf.keras.layers.Activation(activation)

        def call(self, x, training=None):
            primary = self.primary_conv(x)
            primary = self.primary_gn(primary, training=training)
            primary = self.primary_act(primary)
            ghost = self.ghost_dw(primary)
            ghost = self.ghost_gn(ghost, training=training)
            ghost = self.ghost_act(ghost)
            out = tf.concat([primary, ghost], axis=-1)
            return out[:, :, :, :self.filters]

        def get_config(self):
            config = super().get_config()
            config.update({'filters': self.filters, 'ratio': self.ratio})
            return config

    class CoordinateAttention(tf.keras.layers.Layer):
        def __init__(self, reduction=4, **kwargs):
            super().__init__(**kwargs)
            self.reduction = reduction

        def build(self, input_shape):
            channels = input_shape[-1]
            mid_channels = max(8, channels // self.reduction)
            self.shared_conv = tf.keras.layers.Conv2D(mid_channels, 1, use_bias=False)
            self.shared_gn = tf.keras.layers.GroupNormalization(groups=min(8, mid_channels), axis=-1)
            self.shared_act = tf.keras.layers.Activation('relu')
            self.conv_h = tf.keras.layers.Conv2D(channels, 1, use_bias=False)
            self.conv_w = tf.keras.layers.Conv2D(channels, 1, use_bias=False)
            super().build(input_shape)

        def call(self, x, training=None):
            x_shape = tf.shape(x)
            h = x_shape[1]
            w = x_shape[2]
            pool_h = tf.reduce_mean(x, axis=2, keepdims=True)
            pool_w = tf.reduce_mean(x, axis=1, keepdims=True)
            pool_w_t = tf.transpose(pool_w, perm=[0, 2, 1, 3])
            combined = tf.concat([pool_h, pool_w_t], axis=1)
            combined = self.shared_conv(combined)
            combined = self.shared_gn(combined, training=training)
            combined = self.shared_act(combined)
            split_h, split_w = tf.split(combined, [h, w], axis=1)
            attn_h = tf.sigmoid(self.conv_h(split_h))
            split_w_back = tf.transpose(split_w, perm=[0, 2, 1, 3])
            attn_w = tf.sigmoid(self.conv_w(split_w_back))
            return x * attn_h * attn_w

        def get_config(self):
            config = super().get_config()
            config.update({'reduction': self.reduction})
            return config

    class AttentionGate(tf.keras.layers.Layer):
        def __init__(self, filters, **kwargs):
            super().__init__(**kwargs)
            self.filters = filters

        def build(self, input_shape):
            self.wl = tf.keras.layers.Conv2D(self.filters, 1, strides=1, padding='same', use_bias=True)
            self.wg = tf.keras.layers.Conv2D(self.filters, 1, strides=1, padding='same', use_bias=True)
            self.psi = tf.keras.layers.Conv2D(1, 1, strides=1, padding='same', use_bias=True)
            super().build(input_shape)

        def call(self, inputs, training=None):
            x, g = inputs
            xl = self.wl(x)
            gg = self.wg(g)
            gg = tf.image.resize(gg, (tf.shape(x)[1], tf.shape(x)[2]))
            joined = tf.add(xl, gg)
            act = tf.nn.relu(joined)
            psi = self.psi(act)
            coef = tf.nn.sigmoid(psi)
            return tf.multiply(x, coef)

        def get_config(self):
            config = super().get_config()
            config.update({'filters': self.filters})
            return config

    class GhostBottleneck(tf.keras.layers.Layer):
        def __init__(self, filters, ratio=2, use_attention=False, dilation_rate=1, **kwargs):
            super().__init__(**kwargs)
            self.ratio = ratio
            self.target_filters = filters
            self.use_attention = use_attention
            self.dilation_rate = dilation_rate
            self.ghost1 = GhostModule(filters, kernel_size=1, ratio=ratio)
            self.ghost2 = GhostModule(filters, kernel_size=3, ratio=ratio)
            if use_attention:
                self.attention = CoordinateAttention()
            self.residual_conv = None

        def build(self, input_shape):
            if input_shape[-1] != self.target_filters:
                self.residual_conv = tf.keras.layers.Conv2D(self.target_filters, 1, padding='same', use_bias=False)
            super().build(input_shape)

        def call(self, x, training=None):
            residual = x
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            out = self.ghost1(x, training=training)
            out = self.ghost2(out, training=training)
            if self.use_attention:
                out = self.attention(out, training=training)
            return out + residual

        def get_config(self):
            config = super().get_config()
            config.update({
                'filters': self.target_filters,
                'ratio': self.ratio,
                'use_attention': self.use_attention,
                'dilation_rate': self.dilation_rate,
            })
            return config

    class DW_ASPP(tf.keras.layers.Layer):
        def __init__(self, out_channels, rates=(2, 4, 6), **kwargs):
            super().__init__(**kwargs)
            self.out_channels = out_channels
            self.rates = rates

        def build(self, input_shape):
            ch = input_shape[-1]
            branch_ch = max(ch // 4, 16)
            self.b1_conv = tf.keras.layers.Conv2D(branch_ch, 1, padding='same', use_bias=False)
            self.b1_bn = tf.keras.layers.BatchNormalization()
            self.dw_branches = []
            for rate in self.rates:
                dw = tf.keras.layers.DepthwiseConv2D(3, padding='same', dilation_rate=rate, use_bias=False)
                pw = tf.keras.layers.Conv2D(branch_ch, 1, padding='same', use_bias=False)
                bn = tf.keras.layers.BatchNormalization()
                self.dw_branches.append((dw, pw, bn))
            self.gap_conv = tf.keras.layers.Conv2D(branch_ch, 1, use_bias=False)
            self.gap_bn = tf.keras.layers.BatchNormalization()
            self.proj_conv = tf.keras.layers.Conv2D(self.out_channels, 1, padding='same', use_bias=False)
            self.proj_bn = tf.keras.layers.BatchNormalization()
            super().build(input_shape)

        def call(self, x, training=None):
            branches = []
            b1 = tf.nn.relu(self.b1_bn(self.b1_conv(x), training=training))
            branches.append(b1)
            for dw, pw, bn in self.dw_branches:
                b = dw(x)
                b = pw(b)
                b = tf.nn.relu(bn(b, training=training))
                branches.append(b)
            gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            gap = tf.nn.relu(self.gap_bn(self.gap_conv(gap), training=training))
            gap = tf.image.resize(gap, (tf.shape(x)[1], tf.shape(x)[2]))
            branches.append(gap)
            out = tf.concat(branches, axis=-1)
            out = tf.nn.relu(self.proj_bn(self.proj_conv(out), training=training))
            return out

        def get_config(self):
            config = super().get_config()
            config.update({'out_channels': self.out_channels, 'rates': self.rates})
            return config

    return {
        'GhostModule': GhostModule,
        'CoordinateAttention': CoordinateAttention,
        'AttentionGate': AttentionGate,
        'GhostBottleneck': GhostBottleneck,
        'DW_ASPP': DW_ASPP,
    }

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
    custom_objects = _build_local_custom_objects()

    suffix = model_path.suffix.lower()

    # ------------------------------------------------------------------ .h5
    if suffix == ".h5":
        print(f"[*] Loading H5 model: {model_path.name}")
        try:
            model = tf.keras.models.load_model(
                str(model_path),
                compile=False,
                custom_objects=custom_objects or None,
            )
            print("[*] Loaded OK.")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load H5 model: {e}")

    # ---------------------------------------------------------------- .keras
    if suffix == ".keras":
        print(f"[*] Loading .keras model: {model_path.name}")

        # Attempt 1: plain load (works for standard models)
        try:
            model = tf.keras.models.load_model(
                str(model_path),
                compile=False,
                custom_objects=custom_objects or None,
            )
            print("[*] Loaded OK.")
            return model
        except Exception:
            pass

        # Attempt 2: safe_mode=False
        try:
            model = tf.keras.models.load_model(
                str(model_path),
                compile=False,
                safe_mode=False,
                custom_objects=custom_objects or None,
            )
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
                model = tf.keras.models.model_from_json(
                    _json.dumps(cfg["config"]),
                    custom_objects=custom_objects or None,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not reconstruct model from config: {e}\n"
                    "The model has custom objects that block deserialization on this device."
                )

            model.load_weights(str(weights_file))
            print("[*] Model reconstructed from config + weights. OK.")
            return model

    raise ValueError(f"Unsupported file type: {suffix}. Use .keras or .h5")


def _prepare_benchmark_input(dummy: np.ndarray, model, resolution: int) -> np.ndarray:
    """Resize benchmark input to the model's native size if the model expects a fixed shape."""
    import tensorflow as tf

    expected_shape = getattr(model, "input_shape", None)
    if not expected_shape or len(expected_shape) < 4:
        return dummy

    target_h, target_w = expected_shape[1], expected_shape[2]
    if target_h is None or target_w is None:
        return dummy

    if (resolution, resolution) == (target_h, target_w):
        return dummy

    resized = tf.image.resize(dummy, (target_h, target_w), method="bilinear")
    return resized.numpy() if hasattr(resized, "numpy") else np.asarray(resized)


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
    dummy_model_input = _prepare_benchmark_input(dummy, model, resolution)
    with tf.device("/CPU:0"):
        for _ in range(n_warmup):
            model.predict(dummy_model_input, verbose=0)

    # --- Timed runs ---
    print(f"[*] Benchmarking ({n_runs} passes) ...")
    times = []
    peak_mem = get_memory_mb()

    with tf.device("/CPU:0"):
        for i in range(n_runs):
            t0 = time.perf_counter()
            model.predict(dummy_model_input, verbose=0)
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

# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Model Export — TFLite Conversion
==================================

Converts trained Keras models to TensorFlow Lite format for edge deployment
(e.g., Raspberry Pi, Jetson Nano).

Usage::

    # Export main model
    python export_model.py --model results/main_final.keras

    # Export with float16 quantisation
    python export_model.py --model results/main_final.keras --quantize float16

    # Export from weights file (needs model rebuild)
    python export_model.py --weights results/main_best.weights.h5
"""

import argparse
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(current_dir))


def export_tflite(
    model_path: str = None,
    weights_path: str = None,
    output_path: str = None,
    quantize: str = "none",
    input_shape: tuple = (256, 256),
):
    """Convert a Keras model to TFLite format.

    Args:
        model_path:   Path to a .keras or .h5 saved model.
        weights_path: Path to .weights.h5 (model rebuilt from code).
        output_path:  Output .tflite path. Auto-generated if None.
        quantize:     "none", "float16", or "int8".
        input_shape:  (H, W) for model rebuild from weights.
    """
    import tensorflow as tf

    # --- Load model ---
    if model_path:
        print(f"[*] Loading model from: {model_path}")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "DiceScore": __import__("src.metrics", fromlist=["DiceScore"]).DiceScore,
                "IoUScore": __import__("src.metrics", fromlist=["IoUScore"]).IoUScore,
            },
            compile=False,
        )
    elif weights_path:
        print(f"[*] Rebuilding model and loading weights from: {weights_path}")
        from config import SegmentationConfig
        from src.models import SEGMENTATION_MODELS

        cfg = SegmentationConfig()
        cfg.data.img_size = input_shape

        model = SEGMENTATION_MODELS[cfg.model.name](
            input_shape=(*input_shape, 3),
            num_classes=cfg.model.num_classes,
            encoder_filters=cfg.model.encoder_filters,
            dropout_rate=cfg.model.dropout_rate,
            ghost_ratio=cfg.model.ghost_ratio,
            use_skip_attention=cfg.model.use_skip_attention,
            use_aspp=cfg.model.use_aspp,
            deep_supervision=False,  # TFLite needs single output
        )
        model.load_weights(weights_path)
    else:
        raise ValueError("Provide either --model or --weights")

    # Handle multi-output (deep supervision) -> take main output only
    if len(model.outputs) > 1:
        print(f"[*] Model has {len(model.outputs)} outputs. Taking main output only.")
        model = tf.keras.Model(inputs=model.input, outputs=model.outputs[0])

    print(f"[*] Model: {model.name}, params={model.count_params():,}")
    print(f"[*] Input shape: {model.input_shape}")
    print(f"[*] Output shape: {model.output_shape}")

    # --- Convert ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize == "float16":
        print("[*] Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        print("[*] Applying int8 quantization (dynamic range)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    else:
        print("[*] No quantization (float32).")

    tflite_model = converter.convert()

    # --- Save ---
    if output_path is None:
        src = Path(model_path or weights_path)
        suffix = f"_{quantize}" if quantize != "none" else ""
        output_path = str(src.parent / f"{src.stem}{suffix}.tflite")

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    model_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\n[*] TFLite model saved -> {output_path}")
    print(f"[*] Size: {model_size_mb:.2f} MB")

    # --- Verify ---
    print("[*] Verifying TFLite model...")
    import numpy as np
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()

    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    test_input = np.random.rand(*inp_det["shape"]).astype(inp_det["dtype"])
    interpreter.set_tensor(inp_det["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(out_det["index"])

    print(f"[*] Inference OK — output shape: {output.shape}")
    print(f"[*] Export complete.")

    return output_path


def export_onnx(
    model_path: str = None,
    weights_path: str = None,
    output_path: str = None,
    input_shape: tuple = (256, 256),
):
    """Convert a Keras model to ONNX format.

    Requires: pip install tf2onnx
    """
    import tensorflow as tf

    if model_path:
        model = tf.keras.models.load_model(model_path, compile=False)
    elif weights_path:
        from config import SegmentationConfig
        from src.models import SEGMENTATION_MODELS

        cfg = SegmentationConfig()
        cfg.data.img_size = input_shape
        model = SEGMENTATION_MODELS[cfg.model.name](
            input_shape=(*input_shape, 3),
            num_classes=cfg.model.num_classes,
            deep_supervision=False,
        )
        model.load_weights(weights_path)
    else:
        raise ValueError("Provide either --model or --weights")

    if len(model.outputs) > 1:
        model = tf.keras.Model(inputs=model.input, outputs=model.outputs[0])

    if output_path is None:
        src = Path(model_path or weights_path)
        output_path = str(src.parent / f"{src.stem}.onnx")

    print(f"[*] Converting to ONNX -> {output_path}")
    import tf2onnx
    spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

    onnx_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"[*] ONNX saved -> {output_path}, size: {onnx_size_mb:.2f} MB")
    return output_path


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Segmentation Model")
    parser.add_argument("--model", type=str, help="Path to .keras/.h5 saved model")
    parser.add_argument("--weights", type=str, help="Path to .weights.h5 file")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--quantize", choices=["none", "float16", "int8"], default="none")
    parser.add_argument("--format", choices=["tflite", "onnx", "both"], default="tflite")
    parser.add_argument("--resolution", type=int, default=256, help="Input resolution (square)")
    args = parser.parse_args()

    if not args.model and not args.weights:
        # Default: look for main model
        default = current_dir / "results" / "main_final.keras"
        if default.exists():
            args.model = str(default)
        else:
            print("[!] No model specified and no default found.")
            print("    Use: python export_model.py --model <path> or --weights <path>")
            sys.exit(1)

    shape = (args.resolution, args.resolution)

    if args.format in ("tflite", "both"):
        export_tflite(args.model, args.weights, args.output, args.quantize, shape)

    if args.format in ("onnx", "both"):
        export_onnx(args.model, args.weights, args.output, shape)

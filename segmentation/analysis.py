# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Analysis & Comparison Tables
==============================

Generates publication-ready comparison tables and statistical significance
tests from the results produced by ``run.py``.

Usage::

    python analysis.py                        # full analysis
    python analysis.py --results-dir results  # custom dir
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(current_dir))

CLASS_NAMES = ("MA", "HE", "EX", "SE", "OD")


# =============================================================================
# 1. COMPARISON TABLE
# =============================================================================

def build_comparison_table(results_dir: Path) -> pd.DataFrame:
    """Collect test_metrics.json from main model + all baselines into one table.

    Returns DataFrame with per-class IoU and Dice for each model.
    """
    rows = []

    # Main model
    main_metrics = results_dir / "Ghost_CAS_UNet_v2_test_metrics.json"
    if main_metrics.exists():
        rows.append(_parse_test_metrics(main_metrics))

    # Baselines
    baselines_dir = results_dir / "baselines"
    if baselines_dir.exists():
        for sub in sorted(baselines_dir.iterdir()):
            if sub.is_dir():
                for jf in sub.glob("*_test_metrics.json"):
                    rows.append(_parse_test_metrics(jf))

    if not rows:
        print("[!] No test metrics found. Run training first.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Reorder columns nicely
    fixed_cols = ["Model", "Params", "Mean_IoU", "Mean_Dice"]
    class_cols = [c for c in df.columns if c not in fixed_cols]
    df = df[fixed_cols + sorted(class_cols)]

    return df


def _parse_test_metrics(path: Path) -> Dict:
    """Parse a *_test_metrics.json file into a flat row dict."""
    with open(path) as f:
        data = json.load(f)

    row = {"Model": data.get("model", path.stem)}

    iou = data.get("iou", {})
    dice = data.get("dice", {})
    clinical = data.get("clinical", {})

    row["Mean_IoU"] = iou.get("mean_iou", float("nan"))
    row["Mean_Dice"] = dice.get("mean_dice", float("nan"))

    for i, name in enumerate(CLASS_NAMES):
        row[f"IoU_{name}"] = iou.get(f"iou_class_{i}", float("nan"))
        row[f"Dice_{name}"] = dice.get(f"dice_class_{i}", float("nan"))
        row[f"Sens_{name}"] = clinical.get(f"sensitivity_class_{i}", float("nan"))
        row[f"Prec_{name}"] = clinical.get(f"precision_class_{i}", float("nan"))

    # Try to get param count
    row["Params"] = data.get("params", "—")

    return row


def print_latex_table(df: pd.DataFrame) -> str:
    """Generate a LaTeX-formatted comparison table for IEEE papers."""
    if df.empty:
        return ""

    iou_cols = [c for c in df.columns if c.startswith("IoU_")]
    dice_cols = [c for c in df.columns if c.startswith("Dice_")]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of Segmentation Models on IDRiD Test Set}")
    lines.append(r"\label{tab:comparison}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")

    # IoU table
    header = "Model & " + " & ".join(c.replace("IoU_", "") for c in iou_cols) + r" & Mean \\"
    lines.append(r"\begin{tabular}{l" + "c" * (len(iou_cols) + 1) + "}")
    lines.append(r"\hline")
    lines.append(header)
    lines.append(r"\hline")

    # Find best per column for bolding
    for _, row in df.iterrows():
        vals = [f"{row[c]:.3f}" if not np.isnan(row[c]) else "—" for c in iou_cols]
        mean_val = f"{row['Mean_IoU']:.3f}" if not np.isnan(row['Mean_IoU']) else "—"
        lines.append(f"{row['Model']} & " + " & ".join(vals) + f" & {mean_val}" + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    return latex_str


# =============================================================================
# 2. STATISTICAL SIGNIFICANCE
# =============================================================================

def run_significance_tests(results_dir: Path) -> pd.DataFrame:
    """Run Wilcoxon signed-rank tests comparing main model vs each baseline.

    Uses per-class IoU values as paired samples.
    Returns DataFrame with p-values and significance flags.
    """
    from scipy import stats

    main_metrics = results_dir / "Ghost_CAS_UNet_v2_test_metrics.json"
    if not main_metrics.exists():
        print("[!] Main model test metrics not found.")
        return pd.DataFrame()

    with open(main_metrics) as f:
        main_data = json.load(f)

    main_iou = [main_data["iou"].get(f"iou_class_{i}", float("nan"))
                for i in range(len(CLASS_NAMES))]

    baselines_dir = results_dir / "baselines"
    if not baselines_dir.exists():
        print("[!] No baseline results found.")
        return pd.DataFrame()

    rows = []
    for sub in sorted(baselines_dir.iterdir()):
        if not sub.is_dir():
            continue
        for jf in sub.glob("*_test_metrics.json"):
            with open(jf) as f:
                bl_data = json.load(f)

            bl_name = bl_data.get("model", sub.name)
            bl_iou = [bl_data["iou"].get(f"iou_class_{i}", float("nan"))
                      for i in range(len(CLASS_NAMES))]

            # Wilcoxon signed-rank test (paired, two-sided)
            # Filter out NaN pairs
            pairs = [(m, b) for m, b in zip(main_iou, bl_iou)
                     if not (np.isnan(m) or np.isnan(b))]

            if len(pairs) < 3:
                p_val = float("nan")
            else:
                m_arr = np.array([p[0] for p in pairs])
                b_arr = np.array([p[1] for p in pairs])
                try:
                    stat, p_val = stats.wilcoxon(m_arr, b_arr, alternative="greater")
                except ValueError:
                    p_val = float("nan")

            delta_mean = np.nanmean(main_iou) - np.nanmean(bl_iou)

            rows.append({
                "Baseline": bl_name,
                "Main_Mean_IoU": round(np.nanmean(main_iou), 4),
                "Baseline_Mean_IoU": round(np.nanmean(bl_iou), 4),
                "Delta_IoU": round(delta_mean, 4),
                "p_value": round(p_val, 6) if not np.isnan(p_val) else "—",
                "Significant (p<0.05)": "Yes" if (isinstance(p_val, float) and p_val < 0.05) else "No",
            })

    return pd.DataFrame(rows)


# =============================================================================
# 3. ABLATION SUMMARY
# =============================================================================

def summarize_ablation(results_dir: Path) -> pd.DataFrame:
    """Read ablation results and produce contribution summary."""
    ablation_dir = results_dir / "ablation"

    # Check for CSV from BaseAblationStudy
    csv_files = list(ablation_dir.glob("ablation_results*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        return df

    # Otherwise gather from JSON
    rows = []
    for sub in sorted(ablation_dir.iterdir()):
        if sub.is_dir():
            for jf in sub.glob("*.json"):
                with open(jf) as f:
                    rows.append(json.load(f))

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Segmentation Results Analysis")
    parser.add_argument("--results-dir", type=str, default=str(current_dir / "results"),
                        help="Path to results directory")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    print("=" * 60)
    print("  SEGMENTATION — RESULTS ANALYSIS")
    print("=" * 60)

    # 1. Comparison Table
    print("\n--- Per-Class Comparison Table ---\n")
    df = build_comparison_table(results_dir)
    if not df.empty:
        print(df.to_string(index=False))
        df.to_csv(results_dir / "comparison_table.csv", index=False)
        print(f"\n  Saved -> comparison_table.csv")

        latex = print_latex_table(df)
        with open(results_dir / "comparison_table.tex", "w") as f:
            f.write(latex)
        print(f"  Saved -> comparison_table.tex")

    # 2. Statistical Significance
    print("\n--- Statistical Significance (Wilcoxon) ---\n")
    sig_df = run_significance_tests(results_dir)
    if not sig_df.empty:
        print(sig_df.to_string(index=False))
        sig_df.to_csv(results_dir / "significance_tests.csv", index=False)
        print(f"\n  Saved -> significance_tests.csv")

    # 3. Ablation Summary
    print("\n--- Ablation Summary ---\n")
    abl_df = summarize_ablation(results_dir)
    if not abl_df.empty:
        print(abl_df.to_string(index=False))
    else:
        print("  No ablation results found.")

    print(f"\n{'='*60}")
    print(f"  Analysis complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

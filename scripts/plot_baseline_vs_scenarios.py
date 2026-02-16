from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _latest_comparison_dir(outputs_root: Path) -> Path:
    base = outputs_root / "comparisons"
    dirs = sorted([p for p in base.glob("*") if p.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"No comparison folders found under {base}")
    return dirs[-1]


def _load_comparison_csv(comparison_dir: Path) -> pd.DataFrame:
    csv_path = comparison_dir / "baseline_vs_scenarios_headline_indicators.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing comparison file: {csv_path}")
    return pd.read_csv(csv_path)


def _pivot_ok(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in comparison file.")

    d = df.copy()
    d = d[d["status"] == "ok"]
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col])
    if d.empty:
        raise ValueError(f"No numeric rows with status=ok for '{value_col}'.")

    piv = d.pivot_table(index="scenario", columns="indicator", values=value_col, aggfunc="mean")
    piv = piv.reindex(piv.abs().sum(axis=1).sort_values(ascending=False).index)
    return piv


def _draw_heatmap(ax: plt.Axes, piv: pd.DataFrame, title: str) -> None:
    mat = piv.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 1e-12)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_title(title)
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=7, color="black")

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Delta", rotation=90, va="bottom")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline-vs-scenarios comparison heatmaps.")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help="Path to outputs/comparisons/<timestamp>. If omitted, latest is used.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Outputs root folder (default: outputs).",
    )
    args = parser.parse_args()

    comparison_dir = args.comparison_dir or _latest_comparison_dir(args.outputs_root)
    df = _load_comparison_csv(comparison_dir)

    piv_reporting = _pivot_ok(df, "delta_mean_reporting")
    piv_2100 = _pivot_ok(df, "delta_mean_2100")

    h = max(5.5, 0.45 * len(piv_reporting.index) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(18, h), constrained_layout=True)
    _draw_heatmap(axes[0], piv_reporting, "Delta Mean (2020-2100): Scenario - Baseline")
    _draw_heatmap(axes[1], piv_2100, "Delta at 2100: Scenario - Baseline")

    out_png = comparison_dir / "baseline_vs_scenarios_comparison_heatmaps.png"
    fig.suptitle("Baseline vs Scenarios Headline Comparison", fontsize=14)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()

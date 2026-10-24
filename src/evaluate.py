"""
Model evaluation metrics and diagnostic visualizations for PlateBack Lite.

Computes MAE, RMSE, and R² on the held-out chronological test set, exports
`results/model_comparison.csv`, and saves figures under `results/figures/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import TARGET_COL


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, and R² between actual and predicted values."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def evaluate_models(
    test_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    output_csv: str | Path = "results/model_comparison.csv",
) -> pd.DataFrame:
    """
    Evaluate all models on the held-out test set and save `model_comparison.csv`.
    """
    y_true = test_df[TARGET_COL].to_numpy(dtype=float)

    rows: List[Dict[str, object]] = []
    for model_name, y_pred in predictions.items():
        if len(y_pred) != len(y_true):
            raise ValueError(
                f"Prediction length mismatch for '{model_name}': "
                f"expected {len(y_true)}, got {len(y_pred)}."
            )
        metrics = compute_metrics(y_true, y_pred)
        rows.append(
            {
                "Model": model_name,
                "MAE": round(metrics["MAE"], 4),
                "RMSE": round(metrics["RMSE"], 4),
                "R2": round(metrics["R2"], 4),
            }
        )

    comparison_df = pd.DataFrame(rows)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out_path, index=False)

    return comparison_df


def select_best_ml_model(comparison_df: pd.DataFrame) -> str:
    """
    Select the non-baseline model with the lowest RMSE on the test set.
    Falls back to the overall lowest-RMSE model if only one model is present.
    """
    non_baseline = comparison_df[
        ~comparison_df["Model"].str.contains("Baseline", case=False, na=False)
    ]
    target_df = non_baseline if not non_baseline.empty else comparison_df
    best_idx = target_df["RMSE"].idxmin()
    return str(target_df.loc[best_idx, "Model"])


def plot_actual_vs_predicted(
    test_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    best_model_name: str,
    output_path: str | Path = "results/figures/actual_vs_predicted.png",
) -> Path:
    """
    Generate and save the Actual vs. Predicted demand plot:
    - Panel A: Weekly total actual vs. predicted demand across the test period.
    - Panel B: Sample-level actual vs. predicted scatter for the best ML model.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Weekly aggregated demand trajectory over test weeks
    weekly_df = pd.DataFrame(
        {"week": test_df["week"].to_numpy(), "Actual": test_df[TARGET_COL].to_numpy()}
    )
    for name, preds in predictions.items():
        weekly_df[name] = preds

    weekly_totals = weekly_df.groupby("week").sum()
    ax0 = axes[0]
    ax0.plot(
        weekly_totals.index,
        weekly_totals["Actual"],
        label="Actual Demand",
        color="#111827",
        linewidth=2.5,
        marker="o",
    )

    palette = ["#6B7280", "#2563EB", "#059669", "#D97706"]
    for idx, name in enumerate(predictions.keys()):
        ax0.plot(
            weekly_totals.index,
            weekly_totals[name],
            label=name,
            linestyle="--" if "Baseline" in name else "-",
            linewidth=1.8,
            color=palette[idx % len(palette)],
        )

    ax0.set_title("Weekly Total Demand in Test Period (Actual vs. Predicted)")
    ax0.set_xlabel("Week")
    ax0.set_ylabel("Total Orders")
    ax0.legend(fontsize=8.5, loc="best")
    ax0.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Scatter plot of actual vs predicted for best model
    ax1 = axes[1]
    y_true = test_df[TARGET_COL].to_numpy(dtype=float)
    y_best = predictions[best_model_name]

    # Subsample up to 3,000 points for visual clarity if test set is large
    rng = np.random.default_rng(42)
    n_pts = len(y_true)
    if n_pts > 3000:
        sample_idx = rng.choice(n_pts, size=3000, replace=False)
        x_plot, y_plot = y_true[sample_idx], y_best[sample_idx]
    else:
        x_plot, y_plot = y_true, y_best

    ax1.scatter(
        x_plot,
        y_plot,
        alpha=0.35,
        s=16,
        color="#2563EB",
        edgecolors="none",
        label=f"{best_model_name}",
    )
    max_val = float(max(np.max(x_plot), np.max(y_plot), 1.0))
    ax1.plot(
        [0, max_val],
        [0, max_val],
        color="#DC2626",
        linestyle="--",
        linewidth=1.8,
        label="Ideal (y = x)",
    )
    ax1.set_title(f"Actual vs. Predicted Orders ({best_model_name})")
    ax1.set_xlabel("Actual Orders (num_orders)")
    ax1.set_ylabel("Predicted Orders")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | Path = "results/figures/model_comparison.png",
) -> Path:
    """
    Generate and save a side-by-side bar chart comparing MAE, RMSE, and R²
    across all evaluated models.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("MAE", "Lower is better"), ("RMSE", "Lower is better"), ("R2", "Higher is better")]
    colors = ["#6B7280", "#3B82F6", "#10B981", "#F59E0B"]

    models = comparison_df["Model"].tolist()
    x_pos = np.arange(len(models))

    for ax, (metric, subtitle) in zip(axes, metrics):
        vals = comparison_df[metric].to_numpy(dtype=float)
        bars = ax.bar(
            x_pos,
            vals,
            color=[colors[i % len(colors)] for i in range(len(models))],
            width=0.6,
            edgecolor="#1F2937",
            linewidth=0.7,
        )
        ax.set_title(f"{metric} ({subtitle})")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=22, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.annotate(
                f"{val:.3f}" if metric == "R2" else f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    fig.suptitle("Held-Out Test Set Performance Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_residuals(
    test_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    best_model_name: str,
    output_path: str | Path = "results/figures/residuals.png",
) -> Path:
    """
    Generate and save prediction residual diagnostic plots:
    - Panel A: Residuals (Actual - Predicted) vs. Predicted demand for the best ML model.
    - Panel B: Residual distribution histogram (Best ML Model vs. Historical Baseline).
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    y_true = test_df[TARGET_COL].to_numpy(dtype=float)
    y_best = predictions[best_model_name]
    residuals_best = y_true - y_best

    baseline_name = next(
        (k for k in predictions if "Baseline" in k), best_model_name
    )
    residuals_base = y_true - predictions[baseline_name]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subsample scatter if needed
    rng = np.random.default_rng(42)
    n_pts = len(y_true)
    if n_pts > 3000:
        idx = rng.choice(n_pts, size=3000, replace=False)
        pred_sample = y_best[idx]
        res_sample = residuals_best[idx]
    else:
        pred_sample = y_best
        res_sample = residuals_best

    ax0 = axes[0]
    ax0.scatter(
        pred_sample,
        res_sample,
        alpha=0.35,
        s=16,
        color="#059669",
        edgecolors="none",
    )
    ax0.axhline(0.0, color="#DC2626", linestyle="--", linewidth=1.6)
    ax0.set_title(f"Residuals vs. Predicted Demand ({best_model_name})")
    ax0.set_xlabel("Predicted Orders")
    ax0.set_ylabel("Residual (Actual - Predicted)")
    ax0.grid(True, linestyle=":", alpha=0.6)

    ax1 = axes[1]
    # Clip extreme 1% tails for readable histogram bins
    low_q = float(np.percentile(np.concatenate([residuals_best, residuals_base]), 1))
    high_q = float(np.percentile(np.concatenate([residuals_best, residuals_base]), 99))
    if low_q >= high_q:
        low_q, high_q = -10.0, 10.0
    bins = np.linspace(low_q, high_q, 40)

    if baseline_name != best_model_name:
        ax1.hist(
            np.clip(residuals_base, low_q, high_q),
            bins=bins,
            alpha=0.45,
            label=baseline_name,
            color="#6B7280",
        )
    ax1.hist(
        np.clip(residuals_best, low_q, high_q),
        bins=bins,
        alpha=0.60,
        label=best_model_name,
        color="#2563EB",
    )
    ax1.axvline(0.0, color="#DC2626", linestyle="--", linewidth=1.6)
    ax1.set_title("Residual Error Distribution (Actual - Predicted)")
    ax1.set_xlabel("Residual (Orders)")
    ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file


def generate_all_evaluation_outputs(
    test_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    results_dir: str | Path = "results",
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """
    Run full evaluation: save `model_comparison.csv` and all three required
    diagnostic plots under `results/figures/`.
    """
    res_path = Path(results_dir)
    fig_dir = res_path / "figures"

    comparison_df = evaluate_models(
        test_df=test_df,
        predictions=predictions,
        output_csv=res_path / "model_comparison.csv",
    )
    best_model = select_best_ml_model(comparison_df)

    fig_paths = {
        "actual_vs_predicted": plot_actual_vs_predicted(
            test_df,
            predictions,
            best_model_name=best_model,
            output_path=fig_dir / "actual_vs_predicted.png",
        ),
        "model_comparison": plot_model_comparison(
            comparison_df,
            output_path=fig_dir / "model_comparison.png",
        ),
        "residual_analysis": plot_residuals(
            test_df,
            predictions,
            best_model_name=best_model,
            output_path=fig_dir / "residual_analysis.png",
        ),
        "residuals": plot_residuals(
            test_df,
            predictions,
            best_model_name=best_model,
            output_path=fig_dir / "residuals.png",
        ),
    }
    return comparison_df, fig_paths

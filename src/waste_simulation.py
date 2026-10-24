"""
Food preparation surplus and shortage simulation for PlateBack Lite.

IMPORTANT METHODOLOGICAL DISCLAIMER:
The public Genpact Food Demand Forecasting dataset contains historical order
counts (`num_orders`) but does NOT contain physical kitchen preparation logs or
measured food waste. All surplus and shortage quantities computed in this module
are *simulated operational proxies* under explicit preparation policy assumptions.
Simulated surplus must not be interpreted or reported as measured food waste.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_preprocessing import TARGET_COL
from src.train_models import HistoricalAverageBaseline


SIMULATION_ASSUMPTIONS: List[str] = [
    "1. Unit of analysis: Each test-set record represents weekly meal demand D_{c,m,t} "
    "for meal m at fulfilment centre c in week t.",
    "2. Information boundary (zero test leakage): Preparation quantities Q_{c,m,t} are "
    "chosen prior to observing test-period demand. The baseline policy uses only "
    "training-period (week <= split_week) historical averages per (center_id, meal_id).",
    "3. Preparation rule with safety buffer (alpha): Each policy prepares "
    "Q_{c,m,t} = round((1 + alpha) * forecast_{c,m,t}) non-negative meal units.",
    "4. Perishability / non-carryover: Prepared units exceeding weekly demand cannot be "
    "stored as fresh prepared meals for future weeks, resulting in simulated weekly "
    "surplus max(0, Q_{c,m,t} - D_{c,m,t}) and simulated shortage max(0, D_{c,m,t} - Q_{c,m,t}).",
    "5. Proxy disclaimer: Simulated surplus measures hypothetical over-preparation "
    "relative to realized orders and does NOT equal measured physical food waste.",
]


def compute_baseline_preparation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    safety_buffer: float = 0.05,
) -> np.ndarray:
    """
    Compute baseline preparation quantities for `test_df` using ONLY information
    available in `train_df` (before the test period begins).

    Actual test-period demand (`test_df['num_orders']`) is never accessed.
    """
    if train_df["week"].max() >= test_df["week"].min():
        raise ValueError(
            "Leakage violation in baseline policy: train_df weeks must strictly "
            "precede test_df weeks."
        )

    # Drop target column from test_df before passing to baseline predictor
    test_features_only = test_df[["center_id", "meal_id"]].copy()

    baseline_estimator = HistoricalAverageBaseline()
    baseline_estimator.fit(train_df[["center_id", "meal_id"]], train_df[TARGET_COL])

    raw_baseline_forecast = baseline_estimator.predict(test_features_only)
    prepared_units = np.round((1.0 + safety_buffer) * raw_baseline_forecast)
    return np.maximum(prepared_units, 0.0)


def evaluate_preparation_policy(
    actual_demand: np.ndarray,
    prepared_units: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate simulated surplus, simulated shortage, and percentage of demand served
    for a vector of prepared quantities against actual test-period demand.
    """
    d = np.asarray(actual_demand, dtype=float)
    q = np.maximum(np.asarray(prepared_units, dtype=float), 0.0)

    if len(d) != len(q):
        raise ValueError(
            f"Length mismatch between actual_demand ({len(d)}) and prepared_units ({len(q)})."
        )

    surplus = np.maximum(0.0, q - d)
    shortage = np.maximum(0.0, d - q)
    served = np.minimum(q, d)

    total_demand = float(np.sum(d))
    total_prepared = float(np.sum(q))
    total_surplus = float(np.sum(surplus))
    total_shortage = float(np.sum(shortage))

    demand_served_pct = (
        100.0 * float(np.sum(served)) / total_demand if total_demand > 0 else 100.0
    )

    return {
        "Total_Actual_Demand": total_demand,
        "Total_Prepared_Units": total_prepared,
        "Total_Simulated_Surplus": total_surplus,
        "Total_Simulated_Shortage": total_shortage,
        "Demand_Served_Pct": demand_served_pct,
    }


def run_waste_simulation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_predictions: Dict[str, np.ndarray],
    safety_buffer: float = 0.05,
    output_csv: str | Path = "results/waste_simulation.csv",
    figure_path: str | Path = "results/figures/waste_simulation.png",
) -> pd.DataFrame:
    """
    Compare the historical-demand baseline preparation policy against prediction-based
    preparation policies across the held-out test set.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (`week <= split_week`) used to fit the baseline preparation policy.
    test_df : pd.DataFrame
        Held-out test data (`week > split_week`).
    model_predictions : dict of str -> np.ndarray
        Test-set demand predictions from trained models.
    safety_buffer : float, default=0.05
        Proportional safety margin alpha applied to expected demand:
        `Q = round((1 + alpha) * forecast)`.
    output_csv : str or Path
        Destination path for `results/waste_simulation.csv`.
    figure_path : str or Path
        Destination path for `results/figures/waste_simulation.png`.

    Returns
    -------
    sim_df : pd.DataFrame
        Simulation summary table saved to `output_csv`.
    """
    actual_demand = test_df[TARGET_COL].to_numpy(dtype=float)

    # 1. Baseline preparation policy strictly from training-period historical averages
    baseline_prepared = compute_baseline_preparation(
        train_df=train_df,
        test_df=test_df,
        safety_buffer=safety_buffer,
    )
    baseline_stats = evaluate_preparation_policy(actual_demand, baseline_prepared)
    baseline_surplus = baseline_stats["Total_Simulated_Surplus"]

    rows: List[Dict[str, object]] = [
        {
            "Policy": "Baseline Policy (Historical Average)",
            "Model_Source": "Historical Average Baseline",
            "Safety_Buffer_Pct": round(safety_buffer * 100.0, 2),
            "Total_Actual_Demand": round(baseline_stats["Total_Actual_Demand"], 1),
            "Total_Prepared_Units": round(baseline_stats["Total_Prepared_Units"], 1),
            "Total_Simulated_Surplus": round(baseline_surplus, 1),
            "Total_Simulated_Shortage": round(baseline_stats["Total_Simulated_Shortage"], 1),
            "Demand_Served_Pct": round(baseline_stats["Demand_Served_Pct"], 2),
            "Relative_Surplus_Change_vs_Baseline_Pct": 0.0,
        }
    ]

    # 2. Prediction-based preparation policies for each ML model
    for model_name, preds in model_predictions.items():
        if "Baseline" in model_name:
            continue

        pred_prepared = np.round((1.0 + safety_buffer) * np.maximum(preds, 0.0))
        stats = evaluate_preparation_policy(actual_demand, pred_prepared)

        rel_change = (
            100.0 * (stats["Total_Simulated_Surplus"] - baseline_surplus) / baseline_surplus
            if baseline_surplus > 0
            else 0.0
        )

        rows.append(
            {
                "Policy": f"Prediction Policy ({model_name})",
                "Model_Source": model_name,
                "Safety_Buffer_Pct": round(safety_buffer * 100.0, 2),
                "Total_Actual_Demand": round(stats["Total_Actual_Demand"], 1),
                "Total_Prepared_Units": round(stats["Total_Prepared_Units"], 1),
                "Total_Simulated_Surplus": round(stats["Total_Simulated_Surplus"], 1),
                "Total_Simulated_Shortage": round(stats["Total_Simulated_Shortage"], 1),
                "Demand_Served_Pct": round(stats["Demand_Served_Pct"], 2),
                "Relative_Surplus_Change_vs_Baseline_Pct": round(rel_change, 2),
            }
        )

    sim_df = pd.DataFrame(rows)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(out_path, index=False)

    plot_waste_simulation(sim_df, output_path=figure_path)
    return sim_df


def plot_waste_simulation(
    sim_df: pd.DataFrame,
    output_path: str | Path = "results/figures/waste_simulation.png",
) -> Path:
    """Plot simulated surplus, shortage, and percentage of demand served across policies."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    policies = sim_df["Policy"].tolist()
    x = np.arange(len(policies))
    width = 0.36

    surplus = sim_df["Total_Simulated_Surplus"].to_numpy(dtype=float)
    shortage = sim_df["Total_Simulated_Shortage"].to_numpy(dtype=float)
    served_pct = sim_df["Demand_Served_Pct"].to_numpy(dtype=float)

    ax0 = axes[0]
    b1 = ax0.bar(
        x - width / 2,
        surplus,
        width,
        label="Simulated Surplus (Over-Preparation)",
        color="#F59E0B",
        edgecolor="#1F2937",
    )
    b2 = ax0.bar(
        x + width / 2,
        shortage,
        width,
        label="Simulated Shortage (Unmet Demand)",
        color="#EF4444",
        edgecolor="#1F2937",
    )
    ax0.set_title("Simulated Surplus vs. Shortage (Units)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(policies, rotation=20, ha="right", fontsize=8.5)
    ax0.set_ylabel("Simulated Meal Units")
    ax0.legend(fontsize=8.5)
    ax0.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax0.annotate(
            f"{h:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax1 = axes[1]
    b3 = ax1.bar(
        x,
        served_pct,
        width=0.5,
        color="#10B981",
        edgecolor="#1F2937",
    )
    ax1.set_title("Percentage of Total Demand Served (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, rotation=20, ha="right", fontsize=8.5)
    ax1.set_ylabel("Demand Served (%)")
    ax1.set_ylim(0, min(108.0, max(100.0, float(np.max(served_pct)) + 5.0)))
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, val in zip(b3, served_pct):
        ax1.annotate(
            f"{val:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fig.suptitle(
        "Simulated Meal Preparation Policy Comparison (Proxy Only — Not Measured Food Waste)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file

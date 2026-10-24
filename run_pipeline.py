"""
Command-line execution workflow for PlateBack Lite.

Executes the 7-step reproducible research pipeline:
1. Checks that all required dataset files exist in `data/`.
2. Validates expected columns and inspects actual data types.
3. Merges metadata, performs a chronological split, and engineers leak-free features.
4. Trains the 4 models (Historical Average Baseline, Linear Regression,
   Random Forest Regressor, Gradient Boosting Regressor).
5. Evaluates all models on the held-out test set (MAE, RMSE, R²).
6. Runs the food preparation surplus/shortage simulation.
7. Saves tables (`results/model_comparison.csv`, `results/waste_simulation.csv`)
   and figures (`results/figures/*.png`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

from src.data_preprocessing import (
    DatasetMissingError,
    TARGET_COL,
    check_dataset_files,
    chronological_train_test_split,
    engineer_leak_free_features,
    get_dataset_instructions,
    load_and_merge_data,
)
from src.evaluate import generate_all_evaluation_outputs, select_best_ml_model
from src.train_models import train_and_predict_all
from src.waste_simulation import SIMULATION_ASSUMPTIONS, run_waste_simulation


def run_workflow(
    data_dir: str | Path = "data",
    results_dir: str | Path = "results",
    test_weeks_ratio: float = 0.20,
    safety_buffer: float = 0.05,
    random_state: int = 42,
    rf_n_estimators: int = 100,
    gb_n_estimators: int = 150,
    max_train_samples_tree: int | None = 120000,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Execute the complete 7-step PlateBack Lite experiment workflow.
    """
    data_path = Path(data_dir)
    res_path = Path(results_dir)

    # Step 1: Check dataset files exist
    if verbose:
        print("[Step 1/7] Checking required dataset files in:", data_path.resolve())
    all_exist, missing_files = check_dataset_files(data_path)
    if not all_exist:
        raise DatasetMissingError(
            f"Missing required file(s): {', '.join(missing_files)}\n"
            + get_dataset_instructions(data_path)
        )

    # Step 2: Validate expected columns & inspect data types + merge
    if verbose:
        print("[Step 2/7] Inspecting schemas, validating columns, and merging tables...")
    merged_df = load_and_merge_data(data_dir=data_path, verbose=verbose)

    # Step 3: Chronological split and leak-free preprocessing
    if verbose:
        print("[Step 3/7] Performing chronological split and leak-free feature engineering...")
    train_raw, test_raw, split_week = chronological_train_test_split(
        merged_df, test_weeks_ratio=test_weeks_ratio
    )
    train_df, test_df = engineer_leak_free_features(train_raw, test_raw)

    if verbose:
        print(
            f"  Chronological split at week={split_week}: "
            f"Train weeks {int(train_df['week'].min())}..{int(train_df['week'].max())} "
            f"({len(train_df):,} rows) | "
            f"Test weeks {int(test_df['week'].min())}..{int(test_df['week'].max())} "
            f"({len(test_df):,} rows)"
        )

    # Step 4: Train the 4 models
    if verbose:
        print("[Step 4/7] Training baseline and machine learning models...")
    trained_models, predictions = train_and_predict_all(
        train_df=train_df,
        test_df=test_df,
        random_state=random_state,
        rf_n_estimators=rf_n_estimators,
        gb_n_estimators=gb_n_estimators,
        max_train_samples_tree=max_train_samples_tree,
        verbose=verbose,
    )

    # Step 5: Evaluate models and generate diagnostic plots
    if verbose:
        print("[Step 5/7] Evaluating models on held-out test set...")
    comparison_df, fig_paths = generate_all_evaluation_outputs(
        test_df=test_df,
        predictions=predictions,
        results_dir=res_path,
    )
    best_model = select_best_ml_model(comparison_df)

    if verbose:
        print("\n--- Model Comparison (Held-Out Test Set) ---")
        print(comparison_df.to_string(index=False))
        print(f"Best ML Model by RMSE: {best_model}")

    # Step 6: Run food waste (surplus/shortage) simulation
    if verbose:
        print("\n[Step 6/7] Running food preparation surplus and shortage simulation...")
        print("Simulation Assumptions:")
        for assumption in SIMULATION_ASSUMPTIONS:
            print(f"  {assumption}")

    waste_df = run_waste_simulation(
        train_df=train_raw,
        test_df=test_raw,
        model_predictions=predictions,
        safety_buffer=safety_buffer,
        output_csv=res_path / "waste_simulation.csv",
        figure_path=res_path / "figures" / "waste_simulation.png",
    )

    if verbose:
        print("\n--- Food Preparation Simulation Results ---")
        print(waste_df.to_string(index=False))

    # Step 7: Save predictions.csv, weekly summary & dataset metadata
    if verbose:
        print("\n[Step 7/7] Saving predictions and summary artifacts to:", res_path.resolve())

    pred_cols = [c for c in ["id", "week", "center_id", "meal_id", TARGET_COL] if c in test_df.columns]
    full_pred_df = test_df[pred_cols].copy()
    for model_name, preds in predictions.items():
        full_pred_df[model_name] = preds
    full_pred_df.to_csv(res_path / "predictions.csv", index=False)

    weekly_pred_df = pd.DataFrame(
        {"week": test_df["week"].to_numpy(), "Actual": test_df[TARGET_COL].to_numpy()}
    )
    for model_name, preds in predictions.items():
        weekly_pred_df[model_name] = preds
    weekly_summary = weekly_pred_df.groupby("week", as_index=False).sum()
    weekly_summary.to_csv(res_path / "weekly_test_predictions.csv", index=False)

    dataset_summary = {
        "total_rows": int(len(merged_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "min_week": int(merged_df["week"].min()),
        "split_week": int(split_week),
        "max_week": int(merged_df["week"].max()),
        "num_centers": int(merged_df["center_id"].nunique()),
        "num_meals": int(merged_df["meal_id"].nunique()),
        "num_categories": int(merged_df["category"].nunique()),
        "num_cuisines": int(merged_df["cuisine"].nunique()),
        "mean_orders": round(float(merged_df[TARGET_COL].mean()), 2),
        "median_orders": round(float(merged_df[TARGET_COL].median()), 2),
        "best_ml_model": best_model,
        "safety_buffer_pct": round(safety_buffer * 100.0, 2),
    }
    with open(res_path / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    return {
        "comparison_df": comparison_df,
        "waste_df": waste_df,
        "figure_paths": fig_paths,
        "dataset_summary": dataset_summary,
        "trained_models": trained_models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the PlateBack Lite food demand forecasting and waste simulation workflow."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing train.csv, meal_info.csv, and fulfilment_center_info.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory where CSV tables and figures will be saved",
    )
    parser.add_argument(
        "--test-weeks-ratio",
        type=float,
        default=0.20,
        help="Fraction of chronological weeks held out for testing (default: 0.20)",
    )
    parser.add_argument(
        "--safety-buffer",
        type=float,
        default=0.05,
        help="Proportional preparation buffer alpha (default: 0.05 for +5%% buffer)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible training (default: 42)",
    )
    args = parser.parse_args()

    try:
        run_workflow(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            test_weeks_ratio=args.test_weeks_ratio,
            safety_buffer=args.safety_buffer,
            random_state=args.random_state,
            verbose=True,
        )
        print("\nWorkflow completed successfully.")
        return 0
    except DatasetMissingError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

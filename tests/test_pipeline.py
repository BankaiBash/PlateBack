"""
Automated unit and integration tests for PlateBack Lite.

Uses pytest's `tmp_path` fixture so that no synthetic data or fake experimental
results are ever written to the workspace's `data/` or `results/` directories.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from run_pipeline import run_workflow
from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    DatasetMissingError,
    NUMERIC_FEATURES,
    TARGET_COL,
    check_dataset_files,
    chronological_train_test_split,
    engineer_leak_free_features,
    inspect_and_validate_schemas,
    load_and_merge_data,
)
from src.waste_simulation import compute_baseline_preparation


def _create_temp_dataset(data_dir: Path, n_weeks: int = 12) -> None:
    """Create a small, realistic relational fixture in a temporary directory."""
    rng = np.random.default_rng(123)

    meal_df = pd.DataFrame(
        {
            "meal_id": [101, 102, 103],
            "category": ["Beverages", "Rice Bowl", "Sandwich"],
            "cuisine": ["Thai", "Indian", "Italian"],
        }
    )
    center_df = pd.DataFrame(
        {
            "center_id": [10, 20],
            "city_code": [500, 600],
            "region_code": [56, 77],
            "center_type": ["TYPE_A", "TYPE_B"],
            "op_area": [3.5, 4.2],
        }
    )

    rows = []
    rec_id = 1000
    for w in range(1, n_weeks + 1):
        for c in [10, 20]:
            for m in [101, 102, 103]:
                base_p = 200.0 + (m - 100) * 20.0
                promo = int((w + m) % 3 == 0)
                featured = int((w + c) % 4 == 0)
                checkout_p = base_p - (25.0 if promo else 0.0)
                orders = max(
                    1,
                    int(
                        120
                        + (c == 20) * 40
                        + promo * 55
                        + featured * 35
                        - 0.2 * checkout_p
                        + rng.normal(0, 8)
                    ),
                )
                rows.append(
                    {
                        "id": rec_id,
                        "week": w,
                        "center_id": c,
                        "meal_id": m,
                        "checkout_price": checkout_p,
                        "base_price": base_p,
                        "emailer_for_promotion": promo,
                        "homepage_featured": featured,
                        "num_orders": orders,
                    }
                )
                rec_id += 1

    train_df = pd.DataFrame(rows)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_dir / "train.csv", index=False)
    meal_df.to_csv(data_dir / "meal_info.csv", index=False)
    center_df.to_csv(data_dir / "fulfilment_center_info.csv", index=False)


def test_missing_dataset_detection(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir(exist_ok=True)

    all_exist, missing = check_dataset_files(empty_dir)
    assert not all_exist
    assert set(missing) == {"train.csv", "meal_info.csv", "fulfilment_center_info.csv"}

    try:
        inspect_and_validate_schemas(empty_dir, verbose=False)
        raise AssertionError("Expected DatasetMissingError was not raised.")
    except DatasetMissingError:
        pass


def test_chronological_split_and_zero_leakage(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _create_temp_dataset(data_dir, n_weeks=10)

    merged = load_and_merge_data(data_dir, verbose=False)
    train_raw, test_raw, split_week = chronological_train_test_split(
        merged, test_weeks_ratio=0.20
    )

    assert split_week == 8
    assert train_raw["week"].max() == 8
    assert test_raw["week"].min() == 9

    # Engineer features normally
    train_f1, test_f1 = engineer_leak_free_features(train_raw, test_raw)

    # Corrupt test-period target with extreme values (+1,000,000) and verify
    # that neither train features nor test features change at all
    test_corrupted = test_raw.copy()
    test_corrupted[TARGET_COL] = test_corrupted[TARGET_COL] + 1_000_000
    train_f2, test_f2 = engineer_leak_free_features(train_raw, test_corrupted)

    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        pd.testing.assert_series_equal(train_f1[col], train_f2[col])
        pd.testing.assert_series_equal(test_f1[col], test_f2[col])

    # Verify causal property within train_raw: changing week 8 target must not alter
    # historical features for weeks 1..8 (including week 8 itself)
    train_corrupted_w8 = train_raw.copy()
    train_corrupted_w8.loc[train_corrupted_w8["week"] == 8, TARGET_COL] += 500_000
    train_f3, _ = engineer_leak_free_features(train_corrupted_w8, test_raw)

    w1_8_orig = train_f1[train_f1["week"] <= 8]
    w1_8_mod = train_f3[train_f3["week"] <= 8]
    for col in [
        "hist_pair_mean",
        "hist_pair_roll4_mean",
        "hist_pair_std",
        "hist_meal_mean",
        "hist_center_mean",
    ]:
        pd.testing.assert_series_equal(w1_8_orig[col], w1_8_mod[col])


def test_no_contemporaneous_target_leakage_including_week1(tmp_path: Path) -> None:
    """
    Regression test ensuring that mutating target values in week t (specifically
    including the very first training week, week == 1, as well as later weeks)
    never alters any engineered feature for observations in week <= t.
    """
    data_dir = tmp_path / "data"
    _create_temp_dataset(data_dir, n_weeks=8)
    merged = load_and_merge_data(data_dir, verbose=False)
    train_raw, test_raw, _ = chronological_train_test_split(merged, test_weeks_ratio=0.25)

    base_train_feat, _ = engineer_leak_free_features(train_raw, test_raw)

    for target_week in [1, 2, int(train_raw["week"].max())]:
        mutated_train = train_raw.copy()
        mutated_train.loc[mutated_train["week"] == target_week, TARGET_COL] += 750_000
        mut_train_feat, _ = engineer_leak_free_features(mutated_train, test_raw)

        # Every feature in weeks <= target_week (including target_week itself!) must be identical
        orig_sub = base_train_feat[base_train_feat["week"] <= target_week]
        mut_sub = mut_train_feat[mut_train_feat["week"] <= target_week]
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            pd.testing.assert_series_equal(orig_sub[col], mut_sub[col])


def test_baseline_policy_ignores_test_actuals(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _create_temp_dataset(data_dir, n_weeks=10)
    merged = load_and_merge_data(data_dir, verbose=False)
    train_raw, test_raw, _ = chronological_train_test_split(merged, test_weeks_ratio=0.2)

    q1 = compute_baseline_preparation(train_raw, test_raw, safety_buffer=0.05)

    test_modified = test_raw.copy()
    test_modified[TARGET_COL] = 999_999
    q2 = compute_baseline_preparation(train_raw, test_modified, safety_buffer=0.05)

    np.testing.assert_array_equal(q1, q2)


def test_end_to_end_workflow(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    res_dir = tmp_path / "results"
    _create_temp_dataset(data_dir, n_weeks=15)

    outputs = run_workflow(
        data_dir=data_dir,
        results_dir=res_dir,
        test_weeks_ratio=0.20,
        safety_buffer=0.05,
        random_state=42,
        rf_n_estimators=20,
        gb_n_estimators=25,
        verbose=False,
    )

    comp_csv = res_dir / "model_comparison.csv"
    waste_csv = res_dir / "waste_simulation.csv"
    assert comp_csv.is_file()
    assert waste_csv.is_file()
    assert (res_dir / "figures" / "actual_vs_predicted.png").is_file()
    assert (res_dir / "figures" / "model_comparison.png").is_file()
    assert (res_dir / "figures" / "residuals.png").is_file()
    assert (res_dir / "figures" / "waste_simulation.png").is_file()

    comp_df = outputs["comparison_df"]
    assert list(comp_df.columns) == ["Model", "MAE", "RMSE", "R2"]
    assert len(comp_df) == 4

    waste_df = outputs["waste_df"]
    assert len(waste_df) == 4
    expected_waste_cols = {
        "Policy",
        "Total_Simulated_Surplus",
        "Total_Simulated_Shortage",
        "Demand_Served_Pct",
        "Relative_Surplus_Change_vs_Baseline_Pct",
    }
    assert expected_waste_cols.issubset(set(waste_df.columns))

    # Verify predictions.csv exists and exactly reproduces model_comparison.csv and waste_simulation.csv
    preds_csv = res_dir / "predictions.csv"
    assert preds_csv.is_file()
    preds_df = pd.read_csv(preds_csv)
    y_true = preds_df[TARGET_COL].to_numpy(dtype=float)

    for _, row in comp_df.iterrows():
        m_name = str(row["Model"])
        y_pred = preds_df[m_name].to_numpy(dtype=float)
        mae = round(float(np.mean(np.abs(y_true - y_pred))), 4)
        rmse = round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = round(1.0 - ss_res / ss_tot, 4)
        assert row["MAE"] == mae
        assert row["RMSE"] == rmse
        assert row["R2"] == r2

    for _, row in waste_df.iterrows():
        m_name = str(row["Model_Source"])
        y_pred = preds_df[m_name].to_numpy(dtype=float)
        q = np.round(1.05 * np.maximum(y_pred, 0.0))
        surplus = round(float(np.sum(np.maximum(0.0, q - y_true))), 1)
        shortage = round(float(np.sum(np.maximum(0.0, y_true - q))), 1)
        prepared = round(float(np.sum(q)), 1)
        assert row["Total_Prepared_Units"] == prepared
        assert row["Total_Simulated_Surplus"] == surplus
        assert row["Total_Simulated_Shortage"] == shortage
        # Conservation identity: Prepared - Surplus == Actual - Shortage
        assert round(prepared - surplus, 1) == round(float(np.sum(y_true)) - shortage, 1)


class TestPlateBackLitePipeline(unittest.TestCase):
    """Standard library unittest wrapper so tests run without external test runners."""

    def test_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            test_missing_dataset_detection(Path(tmp))

    def test_chronological_and_zero_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            test_chronological_split_and_zero_leakage(Path(tmp))

    def test_no_contemporaneous_leakage_week1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            test_no_contemporaneous_target_leakage_including_week1(Path(tmp))

    def test_baseline_policy_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            test_baseline_policy_ignores_test_actuals(Path(tmp))

    def test_full_end_to_end_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            test_end_to_end_workflow(Path(tmp))


if __name__ == "__main__":
    unittest.main(verbosity=2)


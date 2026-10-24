"""
Data inspection, validation, merging, chronological splitting, and leak-free
feature engineering for PlateBack Lite (Genpact Food Demand Forecasting dataset).
"""

from __future__ import annotations

from pathlib import Path
import shutil
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_FILES: Dict[str, List[str]] = {
    "train.csv": [
        "week",
        "center_id",
        "meal_id",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured",
        "num_orders",
    ],
    "meal_info.csv": [
        "meal_id",
        "category",
        "cuisine",
    ],
    "fulfilment_center_info.csv": [
        "center_id",
        "city_code",
        "region_code",
        "center_type",
        "op_area",
    ],
}

TARGET_COL = "num_orders"

NUMERIC_FEATURES: List[str] = [
    "week",
    "checkout_price",
    "base_price",
    "price_diff",
    "discount_ratio",
    "is_discounted",
    "emailer_for_promotion",
    "homepage_featured",
    "promo_count",
    "op_area",
    "week_sin",
    "week_cos",
    "hist_pair_mean",
    "hist_pair_roll4_mean",
    "hist_pair_std",
    "hist_meal_mean",
    "hist_center_mean",
]

CATEGORICAL_FEATURES: List[str] = [
    "category",
    "cuisine",
    "center_type",
    "city_code",
    "region_code",
]


class DatasetMissingError(FileNotFoundError):
    """Raised when one or more required dataset CSV files are missing."""


def get_dataset_instructions(data_dir: str | Path = "data") -> str:
    """Return clear instructions for obtaining and placing the Genpact dataset."""
    data_path = Path(data_dir).resolve()
    return (
        "\n======================================================================\n"
        "DATASET FILES MISSING - ACTION REQUIRED\n"
        "======================================================================\n"
        "PlateBack Lite uses the public 'Genpact Food Demand Forecasting' dataset.\n"
        "Per project guidelines, fake dataset contents and download URLs are not\n"
        "fabricated. Please obtain the dataset using the steps below:\n\n"
        "1. Search for the public dataset titled:\n"
        "   'Food Demand Forecasting' (originally released for the Analytics Vidhya\n"
        "   Genpact Machine Learning Hackathon, also mirrored on Kaggle).\n"
        "2. Download and extract the training archive containing:\n"
        "   - train.csv\n"
        "   - meal_info.csv\n"
        "   - fulfilment_center_info.csv\n"
        f"3. Place all three CSV files inside:\n"
        f"   {data_path}\n"
        "======================================================================\n"
    )


def ensure_data_files_in_dir(data_dir: str | Path = "data") -> None:
    """
    If any required CSV file is missing from `data_dir` when `data_dir` is the
    default project `data/` folder, check the project root `.` and any `.zip`
    archives in `.` or `data_dir`, copying/extracting found files into `data_dir`
    without deleting or overwriting the user's original files.
    """
    data_path = Path(data_dir)
    if data_path.resolve() != Path("data").resolve():
        return

    data_path.mkdir(parents=True, exist_ok=True)
    root_path = Path(".")

    for filename in REQUIRED_FILES:
        target = data_path / filename
        if target.is_file():
            continue
        root_candidate = root_path / filename
        if root_candidate.is_file():
            shutil.copy2(root_candidate, target)

    # Also check zip archives if any file is still missing
    still_missing = [f for f in REQUIRED_FILES if not (data_path / f).is_file()]
    if still_missing:
        for zpath in list(root_path.glob("*.zip")) + list(data_path.glob("*.zip")):
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    names_map = {Path(member).name: member for member in zf.namelist()}
                    for fname in still_missing:
                        if fname in names_map and not (data_path / fname).is_file():
                            with zf.open(names_map[fname]) as src, open(
                                data_path / fname, "wb"
                            ) as dst:
                                shutil.copyfileobj(src, dst)
            except zipfile.BadZipFile:
                continue


def check_dataset_files(data_dir: str | Path = "data") -> Tuple[bool, List[str]]:
    """
    Check whether all required CSV files exist in `data_dir`.
    """
    ensure_data_files_in_dir(data_dir)
    data_path = Path(data_dir)
    missing = [
        filename
        for filename in REQUIRED_FILES
        if not (data_path / filename).is_file()
    ]
    return len(missing) == 0, missing


def inspect_and_validate_schemas(
    data_dir: str | Path = "data", verbose: bool = True
) -> Dict[str, Dict[str, object]]:
    """
    Load the three dataset CSV files, inspect actual columns and data types,
    and validate that all required columns exist.
    """
    all_exist, missing = check_dataset_files(data_dir)
    if not all_exist:
        raise DatasetMissingError(
            f"Missing dataset file(s) in '{data_dir}': {', '.join(missing)}.\n"
            + get_dataset_instructions(data_dir)
        )

    data_path = Path(data_dir)
    inspection_report: Dict[str, Dict[str, object]] = {}

    for filename, expected_cols in REQUIRED_FILES.items():
        file_path = data_path / filename
        df = pd.read_csv(file_path)

        actual_cols = list(df.columns)
        missing_cols = [c for c in expected_cols if c not in actual_cols]
        if missing_cols:
            raise ValueError(
                f"File '{filename}' is missing required columns: {missing_cols}. "
                f"Found columns: {actual_cols}"
            )
        if df.empty:
            raise ValueError(f"File '{filename}' is empty (0 rows).")

        dtypes_map = {col: str(dtype) for col, dtype in df.dtypes.items()}
        null_counts = {col: int(cnt) for col, cnt in df.isna().sum().items()}

        inspection_report[filename] = {
            "dataframe": df,
            "num_rows": int(len(df)),
            "num_cols": int(df.shape[1]),
            "columns": actual_cols,
            "dtypes": dtypes_map,
            "null_counts": null_counts,
        }

        if verbose:
            print(f"\n--- Schema Inspection: {filename} ---")
            print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
            for col in actual_cols:
                print(
                    f"  - {col:<24} dtype={dtypes_map[col]:<10} "
                    f"nulls={null_counts[col]}"
                )

    return inspection_report


def load_and_merge_data(
    data_dir: str | Path = "data", verbose: bool = True
) -> pd.DataFrame:
    """
    Validate schemas, load `train.csv`, `meal_info.csv`, and
    `fulfilment_center_info.csv`, merge them, and sort chronologically.
    """
    report = inspect_and_validate_schemas(data_dir=data_dir, verbose=verbose)

    train_df: pd.DataFrame = report["train.csv"]["dataframe"]  # type: ignore[assignment]
    meal_df: pd.DataFrame = report["meal_info.csv"]["dataframe"]  # type: ignore[assignment]
    center_df: pd.DataFrame = report["fulfilment_center_info.csv"]["dataframe"]  # type: ignore[assignment]

    merged = train_df.merge(meal_df, on="meal_id", how="left")
    merged = merged.merge(center_df, on="center_id", how="left")

    merged = merged.sort_values(
        by=["week", "center_id", "meal_id"], ascending=[True, True, True]
    ).reset_index(drop=True)

    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in merged.columns:
            merged[cat_col] = merged[cat_col].astype(str)

    if (merged[TARGET_COL] < 0).any():
        raise ValueError(f"Target column '{TARGET_COL}' contains negative values.")

    if verbose:
        min_w, max_w = int(merged["week"].min()), int(merged["week"].max())
        print(
            f"\nMerged dataset ready: {len(merged):,} rows across weeks {min_w}..{max_w} "
            f"({merged['center_id'].nunique()} centers, {merged['meal_id'].nunique()} meals)."
        )

    return merged


def chronological_train_test_split(
    df: pd.DataFrame,
    test_weeks_ratio: float = 0.20,
    split_week: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Perform a reproducible chronological train/test split based on `week`.
    """
    unique_weeks = np.sort(df["week"].unique())
    if len(unique_weeks) < 2:
        raise ValueError(
            "Dataset must contain at least 2 distinct weeks for a chronological split."
        )

    if split_week is None:
        n_test_weeks = max(1, int(round(len(unique_weeks) * test_weeks_ratio)))
        split_idx = len(unique_weeks) - n_test_weeks - 1
        split_idx = max(0, min(split_idx, len(unique_weeks) - 2))
        split_week = int(unique_weeks[split_idx])

    train_df = df[df["week"] <= split_week].copy().reset_index(drop=True)
    test_df = df[df["week"] > split_week].copy().reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Invalid split_week={split_week}: resulted in empty train or test set."
        )

    assert train_df["week"].max() < test_df["week"].min(), (
        "Temporal leakage detected: training weeks overlap with test weeks!"
    )

    return train_df, test_df, split_week


def _add_deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-level price, promotion, and cyclical calendar features."""
    out = df.copy()
    out["price_diff"] = out["base_price"] - out["checkout_price"]
    safe_base = np.maximum(out["base_price"].to_numpy(dtype=float), 1e-6)
    out["discount_ratio"] = out["price_diff"] / safe_base
    out["is_discounted"] = (out["checkout_price"] < out["base_price"]).astype(int)
    out["promo_count"] = (
        out["emailer_for_promotion"].astype(int) + out["homepage_featured"].astype(int)
    )
    week_mod = out["week"].to_numpy(dtype=float) % 52.0
    out["week_sin"] = np.sin(2.0 * np.pi * week_mod / 52.0)
    out["week_cos"] = np.cos(2.0 * np.pi * week_mod / 52.0)
    return out


def engineer_leak_free_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Engineer demand features strictly without target leakage using fast vectorized
    operations.

    Leakage prevention rules enforced:
    1. Chronological boundary check: `train_df['week'].max() < test_df['week'].min()`.
    2. Within `train_df`, historical features (`hist_pair_mean`, `hist_pair_roll4_mean`,
       `hist_pair_std`, `hist_meal_mean`, `hist_center_mean`) for week `t` are computed
       ONLY from observations in weeks `< t`.
    3. For `test_df` (all weeks `> split_week`), historical demand features are computed
       ONLY from `train_df` (weeks `<= split_week`). Actual test-period demand is
       never used to compute any feature.
    """
    if train_df["week"].max() >= test_df["week"].min():
        raise ValueError(
            "Leakage violation: train_df max week must be strictly less than test_df min week."
        )

    train_feat = _add_deterministic_features(train_df)
    test_feat = _add_deterministic_features(test_df)

    train_feat = train_feat.sort_values(
        by=["week", "center_id", "meal_id"]
    ).reset_index(drop=True)

    # --- 1. Strictly prior week-level meal, center, and global averages for train ---
    # For the initial training week (where prior_global_cnt == 0 and no past observations
    # exist), we use a strict past-only constant fallback of 0.0 so that week 1 targets
    # are never used to construct week 1 features (zero contemporaneous target leakage).
    weekly_global = (
        train_feat.groupby("week", as_index=False)[TARGET_COL]
        .agg(w_sum="sum", w_cnt="count")
        .sort_values("week")
    )
    weekly_global["prior_global_sum"] = (
        weekly_global["w_sum"].cumsum() - weekly_global["w_sum"]
    )
    weekly_global["prior_global_cnt"] = (
        weekly_global["w_cnt"].cumsum() - weekly_global["w_cnt"]
    )
    weekly_global["prior_global_mean"] = np.where(
        weekly_global["prior_global_cnt"] > 0,
        weekly_global["prior_global_sum"] / weekly_global["prior_global_cnt"],
        0.0,
    )
    train_feat = train_feat.merge(
        weekly_global[["week", "prior_global_mean"]], on="week", how="left"
    )

    weekly_meal = (
        train_feat.groupby(["meal_id", "week"], as_index=False)[TARGET_COL]
        .agg(m_sum="sum", m_cnt="count")
        .sort_values(["meal_id", "week"])
    )
    weekly_meal["prior_m_sum"] = (
        weekly_meal.groupby("meal_id")["m_sum"].cumsum() - weekly_meal["m_sum"]
    )
    weekly_meal["prior_m_cnt"] = (
        weekly_meal.groupby("meal_id")["m_cnt"].cumsum() - weekly_meal["m_cnt"]
    )
    weekly_meal["hist_meal_mean"] = np.where(
        weekly_meal["prior_m_cnt"] > 0,
        weekly_meal["prior_m_sum"] / weekly_meal["prior_m_cnt"],
        np.nan,
    )
    train_feat = train_feat.merge(
        weekly_meal[["meal_id", "week", "hist_meal_mean"]],
        on=["meal_id", "week"],
        how="left",
    )
    train_feat["hist_meal_mean"] = train_feat["hist_meal_mean"].fillna(
        train_feat["prior_global_mean"]
    )

    weekly_center = (
        train_feat.groupby(["center_id", "week"], as_index=False)[TARGET_COL]
        .agg(c_sum="sum", c_cnt="count")
        .sort_values(["center_id", "week"])
    )
    weekly_center["prior_c_sum"] = (
        weekly_center.groupby("center_id")["c_sum"].cumsum() - weekly_center["c_sum"]
    )
    weekly_center["prior_c_cnt"] = (
        weekly_center.groupby("center_id")["c_cnt"].cumsum() - weekly_center["c_cnt"]
    )
    weekly_center["hist_center_mean"] = np.where(
        weekly_center["prior_c_cnt"] > 0,
        weekly_center["prior_c_sum"] / weekly_center["prior_c_cnt"],
        np.nan,
    )
    train_feat = train_feat.merge(
        weekly_center[["center_id", "week", "hist_center_mean"]],
        on=["center_id", "week"],
        how="left",
    )
    train_feat["hist_center_mean"] = train_feat["hist_center_mean"].fillna(
        train_feat["prior_global_mean"]
    )

    # --- 2. Vectorized strictly prior (center_id, meal_id) stats in train ---
    # Because (center_id, meal_id, week) is unique per week in the Genpact dataset,
    # sorting by (center_id, meal_id, week) lets us compute shifted expanding/rolling stats directly.
    train_feat["_orig_idx"] = np.arange(len(train_feat))
    train_feat = train_feat.sort_values(
        by=["center_id", "meal_id", "week"]
    ).reset_index(drop=True)

    grp_keys = [train_feat["center_id"], train_feat["meal_id"]]
    y_vals = train_feat[TARGET_COL].astype(float)

    # Strictly prior count, sum, and sum of squares per (center_id, meal_id)
    prior_cnt = train_feat.groupby(grp_keys).cumcount().to_numpy(dtype=float)
    cum_sum = y_vals.groupby(grp_keys).cumsum().to_numpy(dtype=float)
    prior_sum = cum_sum - y_vals.to_numpy(dtype=float)

    y_sq = (y_vals ** 2).rename("_y_sq")
    cum_sq_sum = y_sq.groupby(grp_keys).cumsum().to_numpy(dtype=float)
    prior_sq_sum = cum_sq_sum - y_sq.to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        pair_mean = np.where(prior_cnt > 0, prior_sum / prior_cnt, np.nan)
        # Sample variance: (sum(x^2) - n * mean^2) / (n - 1)
        pair_var = np.where(
            prior_cnt > 1,
            np.maximum(0.0, (prior_sq_sum - prior_cnt * (pair_mean ** 2)) / (prior_cnt - 1.0)),
            0.0,
        )
        pair_std = np.sqrt(pair_var)

    # Strictly prior 4-week rolling mean using shifted lags 1..4 within each pair
    pair_grp_obj = train_feat.groupby(["center_id", "meal_id"])[TARGET_COL]
    l1 = pair_grp_obj.shift(1).to_numpy(dtype=float)
    l2 = pair_grp_obj.shift(2).to_numpy(dtype=float)
    l3 = pair_grp_obj.shift(3).to_numpy(dtype=float)
    l4 = pair_grp_obj.shift(4).to_numpy(dtype=float)
    lags_stack = np.vstack([l1, l2, l3, l4])
    valid_lags = ~np.isnan(lags_stack)
    lag_counts = valid_lags.sum(axis=0)
    lag_sums = np.where(valid_lags, lags_stack, 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pair_roll4 = np.where(lag_counts > 0, lag_sums / lag_counts, np.nan)

    train_feat["hist_pair_mean"] = pair_mean
    train_feat["hist_pair_roll4_mean"] = pair_roll4
    train_feat["hist_pair_std"] = pair_std

    fallback_pair = 0.5 * (train_feat["hist_meal_mean"] + train_feat["hist_center_mean"])
    train_feat["hist_pair_mean"] = train_feat["hist_pair_mean"].fillna(fallback_pair)
    train_feat["hist_pair_roll4_mean"] = train_feat["hist_pair_roll4_mean"].fillna(
        train_feat["hist_pair_mean"]
    )
    train_feat["hist_pair_std"] = train_feat["hist_pair_std"].fillna(0.0)

    # Restore chronological sort order
    train_feat = (
        train_feat.sort_values("_orig_idx")
        .drop(columns=["_orig_idx", "prior_global_mean"])
        .reset_index(drop=True)
    )

    # --- 3. Test set historical features derived EXCLUSIVELY from `train_df` ---
    train_global_mean = float(train_feat[TARGET_COL].mean())

    meal_means_train = (
        train_feat.groupby("meal_id")[TARGET_COL].mean().rename("hist_meal_mean")
    )
    center_means_train = (
        train_feat.groupby("center_id")[TARGET_COL].mean().rename("hist_center_mean")
    )
    pair_stats_train = (
        train_feat.groupby(["center_id", "meal_id"])[TARGET_COL]
        .agg(
            hist_pair_mean="mean",
            hist_pair_std="std",
        )
        .reset_index()
    )
    pair_stats_train["hist_pair_std"] = pair_stats_train["hist_pair_std"].fillna(0.0)

    # Vectorized last up-to-4 training observations per (center_id, meal_id)
    last4_rows = (
        train_feat.sort_values("week")
        .groupby(["center_id", "meal_id"], as_index=False)
        .tail(4)
    )
    last4_train = (
        last4_rows.groupby(["center_id", "meal_id"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "hist_pair_roll4_mean"})
    )

    test_feat = test_feat.merge(meal_means_train, on="meal_id", how="left")
    test_feat = test_feat.merge(center_means_train, on="center_id", how="left")
    test_feat = test_feat.merge(
        pair_stats_train, on=["center_id", "meal_id"], how="left"
    )
    test_feat = test_feat.merge(
        last4_train, on=["center_id", "meal_id"], how="left"
    )

    test_feat["hist_meal_mean"] = test_feat["hist_meal_mean"].fillna(train_global_mean)
    test_feat["hist_center_mean"] = test_feat["hist_center_mean"].fillna(
        train_global_mean
    )
    test_fallback_pair = 0.5 * (
        test_feat["hist_meal_mean"] + test_feat["hist_center_mean"]
    )
    test_feat["hist_pair_mean"] = test_feat["hist_pair_mean"].fillna(test_fallback_pair)
    test_feat["hist_pair_roll4_mean"] = test_feat["hist_pair_roll4_mean"].fillna(
        test_feat["hist_pair_mean"]
    )
    test_feat["hist_pair_std"] = test_feat["hist_pair_std"].fillna(0.0)

    return train_feat, test_feat


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Create a scikit-learn ColumnTransformer that preprocesses numeric and
    categorical features inside a Pipeline without data leakage.
    """
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = (
        categorical_features if categorical_features is not None else CATEGORICAL_FEATURES
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor

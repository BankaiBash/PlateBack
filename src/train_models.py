"""
Model definitions and training routines for PlateBack Lite.

Implements:
1. Historical-Average Baseline (per (center_id, meal_id) from training period only)
2. Linear Regression (via scikit-learn Pipeline + ColumnTransformer)
3. Random Forest Regressor (via scikit-learn Pipeline + ColumnTransformer)
4. Gradient Boosting Regressor (via scikit-learn Pipeline + ColumnTransformer)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    build_preprocessor,
)


class HistoricalAverageBaseline(BaseEstimator, RegressorMixin):
    """
    Leak-free historical-average baseline regressor.

    Fits historical mean demand per `(center_id, meal_id)` pair exclusively on the
    training split. Unseen pairs fall back hierarchically to the average of their
    training `meal_id` and `center_id` means, and finally to the global training mean.
    """

    def __init__(self) -> None:
        self.pair_means_: Dict[Tuple[int, int], float] = {}
        self.meal_means_: Dict[int, float] = {}
        self.center_means_: Dict[int, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "HistoricalAverageBaseline":
        df = X[["center_id", "meal_id"]].copy()
        df["_target"] = np.asarray(y, dtype=float)

        self.global_mean_ = float(df["_target"].mean())
        self.meal_means_ = (
            df.groupby("meal_id")["_target"].mean().to_dict()
        )
        self.center_means_ = (
            df.groupby("center_id")["_target"].mean().to_dict()
        )
        self.pair_means_ = (
            df.groupby(["center_id", "meal_id"])["_target"].mean().to_dict()
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.empty(len(X), dtype=float)
        centers = X["center_id"].to_numpy()
        meals = X["meal_id"].to_numpy()

        for i, (c_id, m_id) in enumerate(zip(centers, meals)):
            key = (c_id, m_id)
            if key in self.pair_means_:
                preds[i] = self.pair_means_[key]
            else:
                m_val = self.meal_means_.get(m_id, self.global_mean_)
                c_val = self.center_means_.get(c_id, self.global_mean_)
                preds[i] = 0.5 * (m_val + c_val)

        return np.maximum(preds, 0.0)


def build_models(
    random_state: int = 42,
    rf_n_estimators: int = 100,
    gb_n_estimators: int = 150,
) -> Dict[str, BaseEstimator]:
    """
    Construct the four required models using scikit-learn Pipelines for proper
    categorical and numerical preprocessing.
    """
    feature_cols_num = list(NUMERIC_FEATURES)
    feature_cols_cat = list(CATEGORICAL_FEATURES)

    models: Dict[str, BaseEstimator] = {
        "Historical Average Baseline": HistoricalAverageBaseline(),
        "Linear Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(feature_cols_num, feature_cols_cat),
                ),
                ("regressor", LinearRegression()),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(feature_cols_num, feature_cols_cat),
                ),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=rf_n_estimators,
                        max_depth=16,
                        min_samples_leaf=5,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting Regressor": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(feature_cols_num, feature_cols_cat),
                ),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        n_estimators=gb_n_estimators,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    return models


def train_and_predict_all(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
    rf_n_estimators: int = 100,
    gb_n_estimators: int = 150,
    max_train_samples_tree: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, BaseEstimator], Dict[str, np.ndarray]]:
    """
    Train all four models on `train_df` and generate predictions on `test_df`.

    Parameters
    ----------
    train_df : pd.DataFrame
        Feature-engineered training split (`week <= split_week`).
    test_df : pd.DataFrame
        Feature-engineered test split (`week > split_week`).
    random_state : int, default=42
        Random seed for reproducibility.
    rf_n_estimators : int, default=100
        Number of trees for Random Forest.
    gb_n_estimators : int, default=150
        Number of boosting stages for Gradient Boosting.
    max_train_samples_tree : int, optional
        If specified and `len(train_df)` exceeds this threshold, takes the most recent
        `max_train_samples_tree` training rows chronologically for tree ensembles to
        keep training fast on large datasets while preserving chronological order.
    verbose : bool, default=True
        Whether to log training progress.

    Returns
    -------
    trained_models : dict mapping model name to fitted estimator
    predictions : dict mapping model name to 1D numpy array of non-negative test predictions
    """
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])

    models = build_models(
        random_state=random_state,
        rf_n_estimators=rf_n_estimators,
        gb_n_estimators=gb_n_estimators,
    )

    trained_models: Dict[str, BaseEstimator] = {}
    predictions: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        if verbose:
            print(f"Training {name}...")

        is_tree = "Random Forest" in name or "Gradient Boosting" in name
        if (
            is_tree
            and max_train_samples_tree is not None
            and len(X_train) > max_train_samples_tree
        ):
            # Take the most recent training rows chronologically
            X_fit = X_train.iloc[-max_train_samples_tree:]
            y_fit = y_train[-max_train_samples_tree:]
        else:
            X_fit = X_train
            y_fit = y_train

        model.fit(X_fit, y_fit)
        raw_preds = model.predict(X_test)
        clipped_preds = np.maximum(np.asarray(raw_preds, dtype=float), 0.0)

        trained_models[name] = model
        predictions[name] = clipped_preds

    return trained_models, predictions

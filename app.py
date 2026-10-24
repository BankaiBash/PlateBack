"""
Streamlit dashboard for PlateBack Lite:
Predicting Food Demand and Simulating Food Waste Reduction Using Machine Learning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from run_pipeline import run_workflow
from src.data_preprocessing import (
    check_dataset_files,
    get_dataset_instructions,
    inspect_and_validate_schemas,
)
from src.waste_simulation import SIMULATION_ASSUMPTIONS


st.set_page_config(
    page_title="PlateBack Lite — Food Demand & Waste Simulation",
    layout="wide",
)

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"


def render_header() -> None:
    st.title(
        "PlateBack Lite: Predicting Food Demand and Simulating Food Waste Reduction"
    )
    st.markdown(
        "**Academic ML Research System** — Reproducible meal demand forecasting "
        "and policy-based surplus/shortage simulation on the Genpact Food Demand dataset."
    )
    st.warning(
        "**Important Methodological Disclaimer (Simulated Figures Only):** "
        "The public Genpact Food Demand Forecasting dataset records fulfilled meal orders "
        "(`num_orders`) and does **not** contain physical kitchen preparation logs or "
        "measured leftover waste. All surplus and shortage quantities reported in this "
        "dashboard are **simulated operational proxies** under explicit meal-preparation "
        "policy assumptions and must not be interpreted as measured physical food waste."
    )


def render_dataset_overview(data_exists: bool, missing_files: list[str]) -> None:
    st.header("1. Dataset Overview")

    if not data_exists:
        st.error(
            f"Missing required dataset CSV file(s) in `{DATA_DIR.resolve()}`: "
            f"**{', '.join(missing_files)}**"
        )
        st.code(get_dataset_instructions(DATA_DIR), language="text")
        return

    summary_path = RESULTS_DIR / "dataset_summary.json"
    if summary_path.is_file():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Merged Records", f"{summary['total_rows']:,}")
        col2.metric(
            "Train Split (Chronological)",
            f"{summary['train_rows']:,} rows",
            f"Weeks {summary['min_week']}–{summary['split_week']}",
        )
        col3.metric(
            "Test Split (Held-Out)",
            f"{summary['test_rows']:,} rows",
            f"Weeks {summary['split_week'] + 1}–{summary['max_week']}",
        )
        col4.metric(
            "Fulfilment Centres / Meals",
            f"{summary['num_centers']} centres / {summary['num_meals']} meals",
        )

    with st.expander("Inspect Raw CSV Schemas & Data Types", expanded=False):
        report = inspect_and_validate_schemas(DATA_DIR, verbose=False)
        for fname, info in report.items():
            st.subheader(f"`{fname}` ({info['num_rows']:,} rows × {info['num_cols']} columns)")
            schema_df = pd.DataFrame(
                {
                    "Column": info["columns"],
                    "Data Type": [info["dtypes"][c] for c in info["columns"]],
                    "Missing Values": [info["null_counts"][c] for c in info["columns"]],
                }
            )
            st.dataframe(schema_df, use_container_width=True, hide_index=True)


def render_model_evaluation() -> None:
    st.header("2. Model Comparison & Evaluation Metrics")

    comp_path = RESULTS_DIR / "model_comparison.csv"
    if not comp_path.is_file():
        st.info(
            "No evaluation results found in `results/model_comparison.csv` yet. "
            "Run `python run_pipeline.py` from the command line or click **Run Experiments** in the sidebar."
        )
        return

    comp_df = pd.read_csv(comp_path)
    st.subheader("Held-Out Test Set Metrics (MAE, RMSE, R²)")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    best_row = comp_df.loc[comp_df["RMSE"].idxmin()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lowest RMSE Model", str(best_row["Model"]))
    c2.metric("Best MAE", f"{best_row['MAE']:.2f}")
    c3.metric("Best RMSE", f"{best_row['RMSE']:.2f}")
    c4.metric("Best R²", f"{best_row['R2']:.4f}")

    comp_fig = FIGURES_DIR / "model_comparison.png"
    if comp_fig.is_file():
        st.image(str(comp_fig), caption="Model Performance Comparison on Held-Out Test Set")


def render_demand_charts() -> None:
    st.header("3. Actual vs. Predicted Demand & Residual Diagnostics")

    weekly_path = RESULTS_DIR / "weekly_test_predictions.csv"
    if weekly_path.is_file():
        weekly_df = pd.read_csv(weekly_path).set_index("week")
        st.subheader("Weekly Aggregated Test-Period Demand Trajectory")
        st.line_chart(weekly_df)

    col_a, col_b = st.columns(2)
    avp_fig = FIGURES_DIR / "actual_vs_predicted.png"
    res_fig = FIGURES_DIR / "residual_analysis.png"
    if not res_fig.is_file():
        res_fig = FIGURES_DIR / "residuals.png"

    if avp_fig.is_file():
        with col_a:
            st.image(str(avp_fig), caption="Actual vs. Predicted Demand")
    if res_fig.is_file():
        with col_b:
            st.image(str(res_fig), caption="Prediction Residual Analysis")


def render_waste_simulation() -> None:
    st.header("4. Food Preparation Surplus & Shortage Simulation")

    st.info(
        "**Simulation Notice:** Surplus and shortage values below are simulated "
        "from historical-average and ML-predicted preparation policies evaluated "
        "against held-out test orders. They represent hypothetical over-preparation "
        "and unmet demand—not measured kitchen food waste."
    )

    with st.expander("Explicit Simulation Assumptions", expanded=True):
        for item in SIMULATION_ASSUMPTIONS:
            st.markdown(f"- {item}")

    sim_path = RESULTS_DIR / "waste_simulation.csv"
    if not sim_path.is_file():
        st.info(
            "No simulation output found at `results/waste_simulation.csv` yet. "
            "Run `python run_pipeline.py` to generate simulation tables."
        )
        return

    sim_df = pd.read_csv(sim_path)
    st.subheader("Policy Comparison Table (`results/waste_simulation.csv`)")
    st.dataframe(sim_df, use_container_width=True, hide_index=True)

    sim_fig = FIGURES_DIR / "waste_simulation.png"
    if sim_fig.is_file():
        st.image(str(sim_fig), caption="Simulated Surplus, Shortage, and Demand Served")


def main() -> None:
    render_header()

    data_exists, missing_files = check_dataset_files(DATA_DIR)

    with st.sidebar:
        st.header("Experiment Controls")
        st.markdown(
            "Core experiments can be executed via CLI (`python run_pipeline.py`) "
            "or triggered directly below when the dataset is present."
        )
        test_ratio = st.slider(
            "Chronological Test Weeks Ratio",
            min_value=0.10,
            max_value=0.35,
            value=0.20,
            step=0.05,
        )
        safety_buffer_pct = st.slider(
            "Preparation Safety Buffer (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=1.0,
        )
        run_btn = st.button(
            "Run / Update Experiments",
            disabled=not data_exists,
            type="primary",
        )

        if run_btn and data_exists:
            with st.spinner("Running preprocessing, training, evaluation, and simulation..."):
                run_workflow(
                    data_dir=DATA_DIR,
                    results_dir=RESULTS_DIR,
                    test_weeks_ratio=float(test_ratio),
                    safety_buffer=float(safety_buffer_pct) / 100.0,
                    verbose=False,
                )
            st.success("Experiments completed and saved to `results/`.")

    render_dataset_overview(data_exists, missing_files)
    render_model_evaluation()
    render_demand_charts()
    render_waste_simulation()


if __name__ == "__main__":
    main()

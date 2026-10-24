# Experimental Results Tables — PlateBack Lite

> **Note:** Populated with verified outputs from `results/dataset_summary.json`, `results/model_comparison.csv`, and `results/waste_simulation.csv`. See also `paper/results_summary.md` for detailed analysis.

---

## Table 1: Dataset & Chronological Split Summary
*(Source: `results/dataset_summary.json`)*

| Attribute | Value |
| :--- | :--- |
| Total Merged Observations | `456,548` |
| Training Observations (`week <= 116`) | `361,348` |
| Held-Out Test Observations (`week > 116`) | `95,200` |
| Training Week Range | `Weeks 1–116` |
| Test Week Range | `Weeks 117–145` |
| Number of Fulfilment Centres | `77` |
| Number of Distinct Meals | `51` (`14` categories, `4` cuisines) |
| Mean / Median Weekly Orders (`num_orders`) | `261.87 / 136.0` |

---

## Table 2: Held-Out Test Set Forecasting Performance
*(Source: `results/model_comparison.csv`)*

| Model | MAE $\downarrow$ | RMSE $\downarrow$ | $R^2$ $\uparrow$ |
| :--- | :---: | :---: | :---: |
| Historical Average Baseline | `109.7701` | `248.9597` | `0.5353` |
| Linear Regression | `104.9161` | `220.4739` | `0.6356` |
| Random Forest Regressor | `82.5332` | `178.1692` | `0.7620` |
| Gradient Boosting Regressor | **`80.8407`** | **`172.8451`** | **`0.7760`** |

---

## Table 3: Simulated Meal Preparation Policy Comparison
*(Source: `results/waste_simulation.csv`)*

> **Disclaimer:** Surplus and shortage quantities below are **simulated operational proxies** computed from preparation policy rules ($Q = \text{round}((1 + \alpha)\hat{D})$) against held-out test orders (`num_orders`). They do **not** represent measured physical kitchen food waste.

| Preparation Policy | Safety Buffer (%) | Total Actual Demand | Total Prepared Units | Total Simulated Surplus | Total Simulated Shortage | Demand Served (%) | Relative Surplus Change vs. Baseline (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Policy (Historical Average) | `5.0` | `23,618,862.0` | `26,077,208.0` | `6,693,322.0` | `4,234,976.0` | `82.07%` | `0.00%` |
| Prediction Policy (Linear Regression) | `5.0` | `23,618,862.0` | `25,256,017.0` | `5,971,543.0` | `4,334,388.0` | `81.65%` | `-10.78%` |
| Prediction Policy (Random Forest Regressor) | `5.0` | `23,618,862.0` | `25,039,075.0` | `4,753,258.0` | **`3,333,045.0`** | **`85.89%`** | `-28.99%` |
| Prediction Policy (Gradient Boosting Regressor) | `5.0` | `23,618,862.0` | `24,450,993.0` | **`4,350,038.0`** | `3,517,907.0` | `85.11%` | **`-35.01%`** |

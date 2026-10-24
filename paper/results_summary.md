# PlateBack Lite — Final Verified Experimental Results Summary

> **Verification Status:** All figures and numbers reported in this document have been cross-verified against the final regenerated output files (`results/dataset_summary.json`, `results/model_comparison.csv`, `results/predictions.csv`, and `results/waste_simulation.csv`) produced by `python run_pipeline.py` after removing the Week 1 target fallback in `src/data_preprocessing.py`. All 5 automated tests in `tests/test_pipeline.py` pass (`0` failures), and recomputing evaluation metrics and simulation quantities directly from `results/predictions.csv` matches the summary tables with **zero discrepancy**.

---

## 1. Dataset Inspection & Chronological Split Summary

### 1.1 Raw Dataset Files (`data/`)
| File Name | Rows | Columns | Missing Values | Verified Column Names & Data Types |
| :--- | :---: | :---: | :---: | :--- |
| `train.csv` | `456,548` | `9` | `0` | `id` (`int64`), `week` (`int64`), `center_id` (`int64`), `meal_id` (`int64`), `checkout_price` (`float64`), `base_price` (`float64`), `emailer_for_promotion` (`int64`), `homepage_featured` (`int64`), `num_orders` (`int64`) |
| `meal_info.csv` | `51` | `3` | `0` | `meal_id` (`int64`), `category` (`str`, 14 categories), `cuisine` (`str`, 4 cuisines) |
| `fulfilment_center_info.csv` | `77` | `5` | `0` | `center_id` (`int64`), `city_code` (`int64`), `region_code` (`int64`), `center_type` (`str`), `op_area` (`float64`) |

### 1.2 Merged Dataset & Chronological Split (`results/dataset_summary.json`)
| Attribute | Verified Value |
| :--- | :--- |
| **Total Merged Records** | `456,548` (0 duplicate `(center_id, meal_id, week)` keys) |
| **Time Horizon** | Weeks `1` to `145` (`145` distinct weeks) |
| **Chronological Split Boundary (`split_week`)** | Week `116` |
| **Training Set (`week <= 116`)** | `361,348` rows (Weeks `1–116`, `80.0%` of weeks) |
| **Held-Out Test Set (`week > 116`)** | `95,200` rows (Weeks `117–145`, `20.0%` of weeks) |
| **Distinct Fulfilment Centres / Meals** | `77` centres / `51` meals (`14` categories, `4` cuisines) |
| **Target (`num_orders`) Mean / Std** | `261.87` / `395.92` |
| **Target (`num_orders`) Median / IQR** | `136.0` (25%: `54.0`, 75%: `324.0`) |
| **Target (`num_orders`) Min / Max** | `13` / `24,299` |

---

## 2. Held-Out Test Set Forecasting Performance
*(Source: `results/model_comparison.csv` & verified against `results/predictions.csv` across all $N = 95{,}200$ test rows in Weeks 117–145)*

| Model | MAE $\downarrow$ | RMSE $\downarrow$ | $R^2$ $\uparrow$ | MAE Change vs. Baseline | RMSE Change vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical Average Baseline** | `109.7701` | `248.9597` | `0.5353` | `0.00%` | `0.00%` |
| **Linear Regression** | `104.9161` | `220.4739` | `0.6356` | `-4.42%` | `-11.44%` |
| **Random Forest Regressor** | `82.5332` | `178.1692` | `0.7620` | `-24.81%` | `-28.43%` |
| **Gradient Boosting Regressor** | **`80.8407`** | **`172.8451`** | **`0.7760`** | **`-26.35%`** | **`-30.57%`** |

---

## 3. Food Preparation Surplus & Shortage Simulation Results
*(Source: `results/waste_simulation.csv` & verified against `results/predictions.csv` across Weeks 117–145 with Safety Buffer $\alpha = 5.0\%$)*

> **Methodological Disclaimer:** The Kaggle Genpact dataset records fulfilled customer meal orders (`num_orders`) and does not record physical kitchen preparation logs or leftover disposal. All surplus and shortage quantities below are **simulated operational proxies** computed under the preparation rule $Q_{c,m,t} = \text{round}(1.05 \times \max(\hat{D}_{c,m,t}, 0))$. They must not be reported as empirically measured physical food waste.

| Policy | Model Source | Safety Buffer (%) | Total Actual Demand | Total Prepared Units | Total Simulated Surplus | Total Simulated Shortage | Total Demand Served (Units) | Demand Served (%) | Relative Surplus Change vs. Baseline (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Policy (Historical Average)** | Historical Average Baseline | `5.0` | `23,618,862.0` | `26,077,208.0` | `6,693,322.0` | `4,234,976.0` | `19,383,886.0` | `82.07%` | `0.00%` |
| **Prediction Policy (Linear Regression)** | Linear Regression | `5.0` | `23,618,862.0` | `25,256,017.0` | `5,971,543.0` | `4,334,388.0` | `19,284,474.0` | `81.65%` | `-10.78%` |
| **Prediction Policy (Random Forest Regressor)** | Random Forest Regressor | `5.0` | `23,618,862.0` | `25,039,075.0` | `4,753,258.0` | **`3,333,045.0`** | **`20,285,817.0`** | **`85.89%`** | `-28.99%` |
| **Prediction Policy (Gradient Boosting Regressor)** | Gradient Boosting Regressor | `5.0` | `23,618,862.0` | `24,450,993.0` | **`4,350,038.0`** | `3,517,907.0` | `20,100,955.0` | `85.11%` | **`-35.01%`** |

### Internal Consistency & Conservation Identity Check
For every policy, the exact unit conservation identity holds:
$$\text{Total\_Prepared\_Units} - \text{Total\_Simulated\_Surplus} = \text{Total\_Actual\_Demand} - \text{Total\_Simulated\_Shortage} = \text{Total\_Demand\_Served}$$
- **Baseline:** $26{,}077{,}208 - 6{,}693{,}322 = 23{,}618{,}862 - 4{,}234{,}976 = 19{,}383{,}886$
- **Linear Regression:** $25{,}256{,}017 - 5{,}971{,}543 = 23{,}618{,}862 - 4{,}334{,}388 = 19{,}284{,}474$
- **Random Forest:** $25{,}039{,}075 - 4{,}753{,}258 = 23{,}618{,}862 - 3{,}333{,}045 = 20{,}285{,}817$
- **Gradient Boosting:** $24{,}450{,}993 - 4{,}350{,}038 = 23{,}618{,}862 - 3{,}517{,}907 = 20{,}100{,}955$

---

## 4. Audit Summary of Leakage Safeguards & Methodological Limitations

1. **Zero Contemporaneous or Future Leakage in Training and Test Splits:**
   - All training observations (`week <= 116`) strictly precede all test observations (`week >= 117`).
   - Within `train_df` (`Weeks 1–116`), all historical demand features for week $t$ are computed strictly from weeks $< t$. For the initial training week (`week == 1`, where `prior_global_cnt == 0`), `prior_global_mean` uses a strict past-only constant fallback of `0.0` (verified by `test_no_contemporaneous_leakage_week1` in `tests/test_pipeline.py`).
   - Test-set historical features (`hist_pair_mean`, `hist_pair_roll4_mean`, `hist_pair_std`, `hist_meal_mean`, `hist_center_mean`) and the `HistoricalAverageBaseline` estimator are fitted exclusively on `train_df` (`week <= 116`).
   - Preprocessing transformers (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) are fitted strictly on training data inside `scikit-learn` `Pipeline` objects.
2. **Static Historical Features Across Multi-Week Test Horizon:**
   - Because realized test-period demand (Weeks `117–145`) is withheld from feature updates to guarantee zero test leakage during the simulation, historical demand features remain fixed at their end-of-training (Week `116`) values across the 29-week test window.
3. **Chronological Recent-Window Subsampling for Tree Ensembles:**
   - To keep training lightweight on a standard CPU, Random Forest and Gradient Boosting fit on the most recent `120,000` chronological training rows (`max_train_samples_tree=120000`), while their historical features reflect the full training history from Week `1`.
4. **Simulated Proxy vs. Physical Waste & Censored Demand:**
   - `num_orders` measures fulfilled customer orders (which may be right-censored if historical stockouts occurred) and does not measure kitchen preparation batches or physical leftover disposal.

---

## 5. Verified Output Artifacts Inventory

| File Path | Description |
| :--- | :--- |
| `paper/PlateBack_Lite_Research_Paper.md` | Full 14-section academic research manuscript |
| `paper/results_summary.md` | Verified experimental results and audit summary |
| `paper/research_outline.md` | Structured research outline |
| `results/model_comparison.csv` | Held-out test MAE, RMSE, and $R^2$ for all 4 models |
| `results/predictions.csv` | Full-precision held-out test actuals and predictions (`95,200` rows) |
| `results/waste_simulation.csv` | Simulated surplus, shortage, demand served %, and relative surplus change |
| `results/weekly_test_predictions.csv` | Weekly aggregated actual vs. predicted demand across test weeks `117–145` |
| `results/dataset_summary.json` | Dataset dimensions, split week, and target summary metadata |
| `results/figures/actual_vs_predicted.png` | Weekly test demand trajectory + Actual vs. Predicted scatter plot |
| `results/figures/model_comparison.png` | Bar charts comparing MAE, RMSE, and $R^2$ across all 4 models |
| `results/figures/residual_analysis.png` | Residuals vs. Predicted scatter and residual distribution histogram |
| `results/figures/waste_simulation.png` | Bar chart comparing simulated surplus, shortage, and % demand served |

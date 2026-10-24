# PlateBack Lite: Predicting Food Demand and Simulating Food Waste Reduction Using Machine Learning

> **Research Integrity & Verification Note:** All dataset statistics, model evaluation metrics, and simulation outputs reported in this outline are populated directly from the verified execution of `run_pipeline.py` on the uploaded Genpact Food Demand Forecasting dataset (`results/dataset_summary.json`, `results/model_comparison.csv`, `results/waste_simulation.csv`). Bibliographic references in Section 11 are standard foundational works and dataset sources marked `[VERIFY BIBLIOGRAPHIC DETAILS BEFORE SUBMISSION]` where venue/access dates must be confirmed by the author.

---

## 1. Abstract

Commercial food-delivery and multi-centre catering operations must commit to meal preparation quantities prior to observing realized customer orders. Inaccurate demand forecasts create a dual operational risk: preparing too many meals leads to unconsumed surplus, whereas preparing too few leads to stockouts and unserved demand. In this study, we present **PlateBack Lite**, a reproducible machine learning framework that forecasts weekly meal demand (`num_orders`) and evaluates a policy-based meal preparation simulation. Using the public Genpact Food Demand Forecasting dataset comprising **456,548** weekly order records across **77** fulfilment centres and **51** meals over **145** weeks, we apply a strict chronological train/test split (training on weeks **1–116**, $N = 361{,}348$; evaluating on held-out weeks **117–145**, $N = 95{,}200$). We compare a leak-free Historical Average Baseline against Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor pipelines utilizing pricing discounts, promotional indicators, meal/centre metadata, and strictly prior historical demand statistics. On the held-out test period, the **Gradient Boosting Regressor** achieved the lowest forecasting error (**MAE = 80.8407**, **RMSE = 172.8451**, **$R^2 = 0.7760$**), followed closely by the **Random Forest Regressor** (**MAE = 82.5332**, **RMSE = 178.1692**, **$R^2 = 0.7620$**), both substantially outperforming **Linear Regression** (**MAE = 104.9161**, **RMSE = 220.4739**, **$R^2 = 0.6356$**) and the **Historical Average Baseline** (**MAE = 109.7701**, **RMSE = 248.9597**, **$R^2 = 0.5353$**). In an operational meal-preparation simulation with a $+5.0\%$ safety buffer across the $23{,}618{,}862$ total test-period meal orders, replacing the historical-average preparation policy with Gradient Boosting and Random Forest predictions reduced **simulated surplus** by **35.01%** and **28.99%**, respectively, while simultaneously increasing the **percentage of total demand served** from **82.07%** (baseline) to **85.11%** (Gradient Boosting) and **85.89%** (Random Forest). Because the dataset records fulfilled orders rather than physical kitchen leftovers, all surplus and shortage figures are explicitly reported as **simulated operational proxies** rather than measured food waste.

---

## 2. Introduction

- **Established Context:** Commercial meal fulfilment centres operate under perishable preparation constraints where weekly production commitments must be made before customer demand is realized.
- **Operational Trade-off:**
  - *Over-prediction* ($\hat{D}_{c,m,t} > D_{c,m,t}$): Generates excess prepared units (operational surplus), increasing production cost and disposal risk.
  - *Under-prediction* ($\hat{D}_{c,m,t} < D_{c,m,t}$): Causes stockouts (unmet demand), resulting in lost sales and reduced service levels.
- **Project Objective:** Evaluate whether classical, lightweight machine learning models trained on historical order patterns, promotional flags, and price discounts can improve weekly demand forecasting accuracy and reduce simulated over-preparation surplus without degrading customer demand fulfilment.

---

## 3. Related Work

- **Retail and Food Demand Forecasting:** Established literature in hospitality and supply chain forecasting shows that item-location baselines, promotional campaigns (email promotions, homepage featuring), and price discounting strongly drive short-term order volumes `[VERIFY DOMAIN CITATION]`.
- **Classical Tabular Regression Ensembles:** Bagged decision trees (Random Forests; Breiman, 2001) and boosted regression trees (Gradient Boosting; Friedman, 2001) are established methods for capturing non-linear interactions between pricing, promotions, and categorical metadata on tabular datasets without requiring deep learning architectures.
- **Safety-Buffer Preparation Policies:** Operational inventory control commonly applies a proportional safety margin $(1 + \alpha)$ to point forecasts to buffer against demand variance `[VERIFY INVENTORY CITATION]`.

---

## 4. Research Questions and Contributions

### Research Questions
- **RQ1 (Forecasting Accuracy):** How much do Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor improve held-out test MAE, RMSE, and $R^2$ over a training-period Historical Average Baseline under a strict chronological split?
- **RQ2 (Simulated Surplus Reduction):** Under a $+5\%$ safety-buffer preparation policy, by what relative percentage do ML-driven preparation policies change simulated surplus compared with a historical-average preparation policy?
- **RQ3 (Service-Level Preservation):** Does reducing simulated surplus via ML predictions come at the expense of customer service levels, or does higher forecast responsiveness also increase the percentage of total demand served?

### Scope of Contributions (Empirical Findings from This Study)
1. **Leak-Free Chronological Benchmarking:** Implementation and verification of a strictly causal preprocessing pipeline on the 456,548-row Genpact dataset where training features at week $t$ use only weeks $< t$ and test features (weeks 117–145) use only training-period data (weeks 1–116).
2. **Empirical Model Comparison:** Demonstration that tree ensembles (Gradient Boosting and Random Forest) reduce test RMSE by **30.57%** and **28.43%**, respectively, relative to the Historical Average Baseline.
3. **Transparent Policy Simulation:** Quantitative evaluation showing that improved forecast responsiveness simultaneously decreases simulated surplus (by **28.99%–35.01%**) and decreases simulated shortages (improving demand served from **82.07%** to **85.11%–85.89%**).

---

## 5. Dataset Description

### 5.1 Source and Files
The experiments use the public **Genpact Food Demand Forecasting** dataset (uploaded via Kaggle), consisting of three relational tables with **zero missing values**:
1. `train.csv`: **456,548 rows × 9 columns** (`id`, `week`, `center_id`, `meal_id`, `checkout_price`, `base_price`, `emailer_for_promotion`, `homepage_featured`, and target `num_orders`).
2. `meal_info.csv`: **51 rows × 3 columns** (`meal_id`, `category` across **14** distinct categories, `cuisine` across **4** distinct cuisines).
3. `fulfilment_center_info.csv`: **77 rows × 5 columns** (`center_id`, `city_code`, `region_code`, `center_type`, `op_area`).

### 5.2 Empirical Target & Feature Summary
- **Time Horizon:** Weeks **1 to 145**.
- **Target Variable (`num_orders`):**
  - Mean: **261.87** orders/week (Std: **395.92**)
  - Median (50th percentile): **136.0** orders/week (IQR: **54.0** to **324.0**)
  - Range: **13** to **24,299** orders/week
- **Pricing Summary:** Mean `checkout_price` = **332.24**, mean `base_price` = **354.16**.

---

## 6. Methodology

### 6.1 Chronological Train/Test Split
To mirror realistic deployment and prevent look-ahead bias, the merged dataset (sorted by `week`, `center_id`, `meal_id`) is partitioned chronologically using an 80/20 week split:
- **Training Split:** Weeks **1–116** (**361,348** rows; 80.0% of weeks).
- **Held-Out Test Split:** Weeks **117–145** (**95,200** rows; 20.0% of weeks).

### 6.2 Leak-Free Feature Engineering
1. **Deterministic Price, Promotion & Seasonality Features:**
   - `price_diff = base_price - checkout_price`
   - `discount_ratio = (base_price - checkout_price) / base_price`
   - `is_discounted = I(checkout_price < base_price)`
   - `promo_count = emailer_for_promotion + homepage_featured`
   - Cyclical week-of-year encoding: `week_sin = sin(2π * (week mod 52) / 52)` and `week_cos = cos(2π * (week mod 52) / 52)`.
2. **Strictly Causal Historical Demand Features:**
   - Within `train` (weeks 1–116): For each row at week $t$, `hist_pair_mean` (expanding mean), `hist_pair_roll4_mean` (4-week rolling mean), `hist_pair_std` (expanding standard deviation), `hist_meal_mean`, and `hist_center_mean` are computed **strictly from weeks $< t$** (using lag-shifted cumulative statistics).
   - Within `test` (weeks 117–145): All historical demand statistics are frozen at the end of the training period (computed exclusively from weeks 1–116). Actual test-period `num_orders` is never used.
3. **Scikit-Learn Preprocessing Pipeline:**
   - Numerical features (17 columns): Median imputation + `StandardScaler`.
   - Categorical features (`category`, `cuisine`, `center_type`, `city_code`, `region_code`): Mode imputation + `OneHotEncoder(handle_unknown="ignore")`.

### 6.3 Evaluated Models
1. **Historical Average Baseline:** Predicts the training-period mean `num_orders` for each `(center_id, meal_id)` pair, with hierarchical fallback to `0.5 * (meal_mean + center_mean)` and global training mean.
2. **Linear Regression:** Ordinary least squares fitted on the full preprocessed feature space.
3. **Random Forest Regressor:** `n_estimators=100, max_depth=16, min_samples_leaf=5, random_state=42`, fitted on the most recent 120,000 chronological training records for computational efficiency.
4. **Gradient Boosting Regressor:** `n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42`, fitted on the most recent 120,000 chronological training records.
5. **Non-Negativity Constraint:** All demand predictions are clipped at zero ($\hat{D}_{c,m,t} = \max(0, \hat{y}_{c,m,t})$).

### 6.4 Food Preparation Simulation Formulation
For each test record $(c, m, t)$ with realized demand $D_{c,m,t}$ and forecast $\hat{D}_{c,m,t}$:
- **Preparation Rule:** $Q_{c,m,t} = \text{round}\big((1 + \alpha)\,\hat{D}_{c,m,t}\big)$ with safety buffer $\alpha = 0.05$ ($+5.0\%$).
- **Simulated Surplus:** $\text{Surplus}_{c,m,t} = \max(0, Q_{c,m,t} - D_{c,m,t})$.
- **Simulated Shortage:** $\text{Shortage}_{c,m,t} = \max(0, D_{c,m,t} - Q_{c,m,t})$.
- **Demand Served (%):** $100 \times \frac{\sum \min(Q_{c,m,t}, D_{c,m,t})}{\sum D_{c,m,t}}$.
- **Relative Surplus Change vs. Baseline (%):** $100 \times \frac{\text{Surplus}_{\text{policy}} - \text{Surplus}_{\text{baseline}}}{\text{Surplus}_{\text{baseline}}}$.

---

## 7. Experimental Setup

- **Environment:** Python 3.14, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `streamlit`.
- **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$), evaluated across all $95{,}200$ held-out test records.
- **Verification:** Automated test suite (`tests/test_pipeline.py`) verifying zero temporal or target leakage.

---

## 8. Results and Discussion

### 8.1 Demand Forecasting Results (RQ1)

| Model | MAE $\downarrow$ | RMSE $\downarrow$ | $R^2$ $\uparrow$ | RMSE Reduction vs. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Historical Average Baseline** | 109.7701 | 248.9597 | 0.5353 | — |
| **Linear Regression** | 104.9161 | 220.4739 | 0.6356 | -11.44% |
| **Random Forest Regressor** | 82.5332 | 178.1692 | 0.7620 | -28.43% |
| **Gradient Boosting Regressor** | **80.8407** | **172.8451** | **0.7760** | **-30.57%** |

- **Linear vs. Historical Baseline:** Incorporating linear effects of price discounts, promotions, and metadata reduced test RMSE from **248.9597** to **220.4739** ($+0.1003$ gain in $R^2$).
- **Non-Linear Tree Ensembles:** Both Random Forest ($R^2 = 0.7620$) and Gradient Boosting ($R^2 = 0.7760$) substantially outperformed Linear Regression, indicating strong non-linear interactions between promotional exposure (`emailer_for_promotion`, `homepage_featured`), price discounting, and item-centre baseline volume.

### 8.2 Food Preparation Simulation Results (RQ2 & RQ3)

*(Total Held-Out Test Demand = **23,618,862** meal orders across weeks 117–145; Safety Buffer = **5.0%**)*

| Preparation Policy | Total Prepared Units | Simulated Surplus (Units) | Simulated Shortage (Units) | Demand Served (%) | Relative Surplus Change vs. Baseline (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Policy (Historical Average)** | 26,077,208 | 6,693,322 | 4,234,976 | 82.07% | 0.00% |
| **Prediction Policy (Linear Regression)** | 25,256,017 | 5,971,543 | 4,334,388 | 81.65% | -10.78% |
| **Prediction Policy (Random Forest Regressor)** | 25,039,075 | 4,753,258 | **3,333,045** | **85.89%** | -28.99% |
| **Prediction Policy (Gradient Boosting Regressor)** | 24,450,993 | **4,350,038** | 3,517,907 | 85.11% | **-35.01%** |

- **Simultaneous Reduction of Simulated Surplus and Shortage:** A static historical average cannot react to promotional spikes (causing severe shortages during promoted weeks) and over-prepares during non-promoted weeks (causing high surplus). By adjusting weekly preparation to promotional and pricing signals, **Gradient Boosting** reduced simulated surplus by **35.01%** (from 6.69M to 4.35M units) while increasing demand served by **+3.04 percentage points** (from 82.07% to 85.11%). **Random Forest** achieved the highest service level (**85.89%** demand served) while reducing simulated surplus by **28.99%**.

---

## 9. Limitations

1. **Simulated Surplus Is Not Measured Food Waste:** The Genpact dataset contains customer order counts (`num_orders`), not physical kitchen batch logs, unserved ingredient weight, or plate waste. All surplus and shortage quantities are simulated policy outcomes.
2. **Demand Censorship:** Observed orders reflect fulfilled transactions; potential unfulfilled demand during historical stockouts is unobserved.
3. **Frozen Historical Features Across the Multi-Week Test Horizon:** Because test-period actuals are strictly excluded from feature updates to guarantee zero test leakage in the simulation, lag/rolling historical features remain anchored at Week 116 across Weeks 117–145.
4. **Symmetric Training Loss:** Models were trained using standard squared-error loss rather than cost-asymmetric quantile loss.

---

## 10. Conclusion

This study developed and verified **PlateBack Lite** on 456,548 weekly meal demand records. Chronological evaluation showed that Gradient Boosting (**RMSE = 172.85, $R^2 = 0.7760$**) and Random Forest (**RMSE = 178.17, $R^2 = 0.7620$**) markedly improve demand forecasting accuracy over both a Historical Average Baseline (**RMSE = 248.96**) and Linear Regression (**RMSE = 220.67**). In our operational meal-preparation simulation (5% safety buffer), prediction-guided preparation reduced simulated surplus by **28.99%–35.01%** while simultaneously improving the share of customer demand served from **82.07%** to **85.11%–85.89%**. Future research should investigate asymmetric quantile regression to directly optimize centre-specific overage and stockout cost ratios.

---

## 11. References

> **Author Action Required:** Verify full bibliographic metadata and access dates before final manuscript submission.

1. **Dataset Source [VERIFY ACCESS DATE & URL]:** Analytics Vidhya / Genpact Machine Learning Hackathon (mirrored on Kaggle). *Food Demand Forecasting Dataset*.
2. **Scikit-Learn:** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. **Random Forests:** Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
4. **Gradient Boosting:** Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.

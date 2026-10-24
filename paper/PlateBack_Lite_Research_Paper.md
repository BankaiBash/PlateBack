# 1. Title

**PlateBack Lite: Predicting Food Demand and Simulating Food Waste Reduction Using Machine Learning**

---

# 2. Abstract

Multi-centre meal-delivery and commercial catering operations must determine meal preparation quantities ahead of realized weekly customer demand. Forecast inaccuracies impose asymmetric operational costs: over-prediction generates excess prepared meals, whereas under-prediction causes stockouts and unserved customer demand. This paper presents **PlateBack Lite**, a reproducible machine learning pipeline that forecasts weekly meal demand (`num_orders`) across fulfilment centres and evaluates the operational consequences of prediction-guided meal preparation through a policy simulation. We evaluate our framework on the public Genpact Food Demand Forecasting dataset, which comprises **456,548** weekly order records spanning **145** weeks across **77** fulfilment centres and **51** distinct meals (`14` categories and `4` cuisines). To prevent look-ahead bias and target leakage, we enforce a strict chronological split—training on Weeks **1–116** ($361{,}348$ observations) and evaluating on held-out Weeks **117–145** ($95{,}200$ observations)—and construct causal historical demand features that rely strictly on past observations prior to each prediction week (with a zero-history constant fallback of $0.0$ in Week 1 to eliminate contemporaneous target leakage). We compare a training-period **Historical Average Baseline** against **Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor** pipelines incorporating pricing discounts, promotional exposure flags, categorical metadata, cyclical seasonality, and causal historical demand summaries. On the held-out test set, the **Gradient Boosting Regressor** achieved the best statistical forecasting performance across all point-prediction metrics (**$\text{MAE} = 80.8407$**, **$\text{RMSE} = 172.8451$**, **$R^2 = 0.7760$**), followed by the **Random Forest Regressor** (**$\text{MAE} = 82.5332$**, **$\text{RMSE} = 178.1692$**, **$R^2 = 0.7620$**) and **Linear Regression** (**$\text{MAE} = 104.9161$**, **$\text{RMSE} = 220.4739$**, **$R^2 = 0.6356$**), all improving over the **Historical Average Baseline** (**$\text{MAE} = 109.7701$**, **$\text{RMSE} = 248.9597$**, **$R^2 = 0.5353$**). We then simulate a weekly meal-preparation policy with a $+5.0\%$ safety buffer across the $23{,}618{,}862$ total test-period meal orders, revealing a clear distinction between the optimal model for simulated surplus minimization and the optimal model for service-level maximization. Relative to the historical-average preparation policy ($6{,}693{,}322$ simulated surplus units; $4{,}234{,}976$ simulated shortage units; $82.07\%$ demand served), **Gradient Boosting** achieved the **lowest simulated surplus** ($4{,}350{,}038$ units, a **$-35.01\%$** relative change vs. baseline, while serving **$85.11\%$** of demand with $3{,}517{,}907$ shortage units), whereas **Random Forest** achieved the **lowest simulated shortage** ($3{,}333{,}045$ units) and **highest percentage of demand served** (**$85.89\%$**, while reducing simulated surplus by **$-28.99\%$** to $4{,}753{,}258$ units). Because the underlying dataset records fulfilled order counts rather than physical kitchen leftovers, all surplus and shortage metrics represent simulated operational proxies under explicit policy assumptions rather than empirically measured or real-world food waste reduction.

---

# 3. Keywords

Food demand forecasting; machine learning; gradient boosting; random forest; perishable inventory simulation; meal preparation policy; chronological validation; leakage prevention.

---

# 4. Introduction

Commercial food preparation and meal-delivery fulfilment centres operate under strict perishability and lead-time constraints. Culinary teams and fulfilment managers must procure ingredients and prepare meal batches before observing actual customer order volumes. When weekly preparation quantities are set using static rules of thumb or unadjusted historical averages, sudden demand fluctuations driven by promotional campaigns or price discounts produce substantial operational inefficiencies. Specifically, preparing more meal units than customers order creates excess prepared inventory (operational surplus), whereas preparing fewer units than demanded results in stockouts (unmet demand), lost revenue, and diminished customer trust.

While machine learning (ML) methods have been widely studied for retail and hospitality demand forecasting (Rodrigues et al., 2024; Hübner et al., 2024), two methodological challenges frequently complicate empirical evaluations in student and applied research projects. First, random train/test splitting or global target-encoding across time series introduces temporal data leakage, artificially inflating test-set accuracy by allowing future demand signals to inform past predictions. Second, evaluating forecasting models solely through symmetric statistical error metrics—such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE)—does not directly reveal how forecast errors translate into operational trade-offs between over-preparation surplus and customer service levels under a buffered preparation policy.

In this study, we design, audit, and evaluate **PlateBack Lite**, a lightweight and reproducible machine learning system built on the public Genpact Food Demand Forecasting dataset ($456{,}548$ records across $77$ fulfilment centres and $51$ meals over $145$ weeks). Our work addresses both methodological rigour and operational interpretability:
1. We implement a strictly chronological train/test split (training on Weeks $1\text{–}116$; testing on Weeks $117\text{–}145$) paired with a causal feature engineering pipeline that eliminates both future look-ahead leakage and Week 1 contemporaneous target leakage.
2. We benchmark four lightweight models suitable for standard CPU execution—a **Historical Average Baseline**, **Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor**—using unified `scikit-learn` preprocessing pipelines.
3. We couple the held-out test forecasts with an explicit, auditable meal-preparation simulation incorporating a $+5.0\%$ safety buffer, quantifying how each forecasting model affects **simulated surplus**, **simulated shortage**, and **percentage of customer demand served**.
4. We maintain a strict conceptual and reporting distinction between statistical prediction accuracy on order counts (`num_orders`) and simulated preparation outcomes, explicitly documenting why simulated surplus on order data is an operational proxy rather than measured physical kitchen waste.

---

# 5. Related Work

### 5.1 Machine Learning for Food and Catering Demand Forecasting
Short-horizon demand forecasting in food service and catering operations requires modelling both baseline item popularity and exogenous demand shifters such as pricing and marketing promotions. Rodrigues et al. (2024) evaluated machine learning models for short-term demand forecasting in food catering services, showing that data-driven demand predictions can inform meal production planning to mitigate overproduction. Similarly, Hübner et al. (2024) examined machine-learning-based demand forecasting in a bakery case study, linking improved demand forecasts to reductions in overbaked surplus and associated life-cycle environmental impacts. On the Genpact Food Demand Forecasting benchmark specifically, Panda and Mohanty (2023) analyzed time-series and regression approaches for supply chain demand modelling, highlighting the importance of centre-level and meal-level attributes alongside promotional indicators.

### 5.2 Classical Regression Ensembles on Tabular Demand Data
For structured tabular datasets with heterogeneous numerical and categorical features, tree-based ensemble methods remain strong, computationally tractable baselines. Random Forests (Breiman, 2001) reduce variance by averaging predictions across bootstrap-aggregated decision trees trained on random feature subsets, capturing non-linear interactions without extensive hyperparameter tuning. Gradient Boosting Machines (Friedman, 2001) sequentially fit shallow regression trees to the negative gradient of the loss function, reducing both bias and variance. Unlike deep neural networks, classical linear and tree-based models implemented in `scikit-learn` (Pedregosa et al., 2011) run reproducibly on standard desktop hardware and allow straightforward inspection of preprocessing pipelines.

### 5.3 Perishable Preparation Policies and Safety Buffers
Translating point demand forecasts into production quantities is a classical problem in perishable inventory management (Arrow et al., 1951). When demand is uncertain and unserved orders cannot be backlogged across weeks, kitchens typically apply a proportional safety buffer $\alpha > 0$ above the point forecast $\hat{D}$ to protect service levels against upward demand variance. In such settings, forecast errors manifest asymmetrically: static historical averages fail to anticipate promotional surges (causing large stockouts during promoted weeks) while over-preparing during unpromoted weeks when prices return to baseline.

---

# 6. Research Questions and Contributions

This study investigates three empirical research questions:

- **RQ1 (Forecasting Accuracy):** Under a strict chronological train/test split free of target leakage, to what extent do supervised regression models (**Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor**) utilizing pricing, promotional, metadata, and causal historical features reduce held-out test error (MAE, RMSE, and $R^2$) compared with a training-period **Historical Average Baseline**?
- **RQ2 (Simulated Surplus Reduction):** When weekly meal preparation quantities are set using a $+5.0\%$ safety buffer ($Q_{c,m,t} = \text{round}(1.05 \times \hat{D}_{c,m,t})$) without access to test-period actual demand, how do prediction-guided policies change total **simulated surplus** relative to a historical-average preparation policy?
- **RQ3 (Service-Level Trade-off):** Does reducing simulated surplus via machine learning predictions degrade customer demand fulfilment (increase simulated shortages), or does dynamic responsiveness to price and promotion signals improve both surplus and the percentage of demand served?

### Summary of Contributions
1. **Verified Leak-Free Feature Engineering:** We implement and unit-test a vectorized causal feature pipeline ensuring that training features at week $t$ depend solely on weeks $< t$ (using a strict $0.0$ constant fallback at Week 1) and test features depend solely on Weeks $1\text{–}116$.
2. **Empirical Model Benchmarking:** Across $95{,}200$ held-out test records, we establish that Gradient Boosting ($\text{RMSE} = 172.8451, R^2 = 0.7760$) and Random Forest ($\text{RMSE} = 178.1692, R^2 = 0.7620$) reduce test RMSE by **30.57%** and **28.43%** relative to the Historical Average Baseline ($\text{RMSE} = 248.9597, R^2 = 0.5353$).
3. **Operational Policy Simulation:** We demonstrate that while Linear Regression reduces simulated surplus by **10.78%** with a slight reduction in demand served ($81.65\%$ vs. $82.07\%$ baseline), non-linear tree ensembles (**Random Forest** and **Gradient Boosting**) reduce simulated surplus by **28.99%** and **35.01%** while simultaneously *increasing* total demand served to **85.89%** and **85.11%**.

---

# 7. Dataset Description

### 7.1 Relational Schema and Completeness
We use the public **Genpact Food Demand Forecasting** dataset (accessed via Kaggle), which contains three relational CSV files with **zero missing values** across all columns:

1. **`train.csv` ($456{,}548$ rows $\times 9$ columns):**
   - `id` (`int64`): Unique transaction record identifier.
   - `week` (`int64`): Sequential week index ranging from $1$ to $145$.
   - `center_id` (`int64`): Fulfilment centre identifier ($77$ unique centres).
   - `meal_id` (`int64`): Meal identifier ($51$ unique meals).
   - `checkout_price` (`float64`): Final price paid by the customer (including discounts/taxes).
   - `base_price` (`float64`): Baseline menu price of the meal.
   - `emailer_for_promotion` (`int64`): Binary indicator ($0$ or $1$) for email promotional campaigns.
   - `homepage_featured` (`int64`): Binary indicator ($0$ or $1$) for homepage placement.
   - `num_orders` (`int64`): **Target variable** representing total weekly orders fulfilled for `(center_id, meal_id, week)`.
2. **`meal_info.csv` ($51$ rows $\times 3$ columns):**
   - Maps each `meal_id` (`int64`) to its food `category` (`str`, $14$ unique categories such as Beverages, Rice Bowl, Sandwich) and `cuisine` (`str`, $4$ unique cuisines: Thai, Indian, Italian, Continental).
3. **`fulfilment_center_info.csv` ($77$ rows $\times 5$ columns):**
   - Maps each `center_id` (`int64`) to `city_code` (`int64`), `region_code` (`int64`), `center_type` (`str`: `TYPE_A`, `TYPE_B`, `TYPE_C`), and operational area `op_area` (`float64`, in $\text{km}^2$).

### 7.2 Exploratory Statistics of Merged Data
Merging `train.csv` with `meal_info.csv` (on `meal_id`) and `fulfilment_center_info.csv` (on `center_id`) yields $456{,}548$ complete observations with zero duplicate `(center_id, meal_id, week)` keys. The target variable `num_orders` is right-skewed:
- **Mean:** $261.87$ orders/week ($\text{SD} = 395.92$)
- **Quartiles:** 25th percentile $= 54.0$, Median (50th percentile) $= 136.0$, 75th percentile $= 324.0$
- **Range:** Minimum $= 13$ orders/week; Maximum $= 24{,}299$ orders/week
- **Pricing:** Mean `checkout_price` $= 332.24$ ($\text{SD} = 152.94$); Mean `base_price` $= 354.16$ ($\text{SD} = 160.72$).

---

# 8. Methodology

### 8.1 Chronological Train/Test Partitioning
To evaluate forecasting generalization on future unseen weeks without temporal leakage, we sort the merged dataset chronologically by $(\text{week}, \text{center\_id}, \text{meal\_id})$ and reserve the final $20\%$ of unique weeks ($29$ weeks) as the held-out test period:
- **Training Period ($\mathcal{D}_{\text{train}}$):** Weeks $1 \le t \le 116$ ($361{,}348$ observations; $79.15\%$ of rows).
- **Held-Out Test Period ($\mathcal{D}_{\text{test}}$):** Weeks $117 \le t \le 145$ ($95{,}200$ observations; $20.85\%$ of rows).

This enforces the strict chronological boundary $\max_{i \in \mathcal{D}_{\text{train}}} \text{week}_i = 116 < \min_{j \in \mathcal{D}_{\text{test}}} \text{week}_j = 117$.

### 8.2 Leak-Free Feature Engineering
For each observation $(c, m, t)$, we construct $17$ numerical features and $5$ categorical features:

1. **Deterministic Price, Promotion, and Seasonality Features (Known at Prediction Time):**
   - Absolute price discount: $\text{price\_diff}_{c,m,t} = \text{base\_price}_{c,m,t} - \text{checkout\_price}_{c,m,t}$.
   - Relative discount ratio: $\text{discount\_ratio}_{c,m,t} = \frac{\text{price\_diff}_{c,m,t}}{\max(\text{base\_price}_{c,m,t}, 10^{-6})}$.
   - Discount indicator: $\text{is\_discounted}_{c,m,t} = \mathbb{I}(\text{checkout\_price}_{c,m,t} < \text{base\_price}_{c,m,t})$.
   - Combined promotion count: $\text{promo\_count}_{c,m,t} = \text{emailer\_for\_promotion}_{c,m,t} + \text{homepage\_featured}_{c,m,t}$.
   - Cyclical annual seasonality ($52$-week period):
     $$\text{week\_sin}_t = \sin\!\left(\frac{2\pi (t \bmod 52)}{52}\right), \qquad \text{week\_cos}_t = \cos\!\left(\frac{2\pi (t \bmod 52)}{52}\right)$$
   - Fulfilment centre operational area: $\text{op\_area}_c$ and raw time index $\text{week}_t$.

2. **Strictly Causal Historical Demand Features (Zero Contemporaneous or Future Leakage):**
   - **Within the Training Set ($1 \le t \le 116$):** For any week $t$, all historical demand features are computed exclusively from observations in strictly prior weeks $t' < t$:
     - `prior_global_mean`: Cumulative sum of `num_orders` across all training records in weeks $1 \dots t-1$ divided by the cumulative count in weeks $1 \dots t-1$. For the initial training week ($t = 1$), where zero prior observations exist ($\text{prior\_global\_cnt} = 0$), we assign a strict past-only constant fallback of $0.0$ so that Week 1 target values are never used to construct Week 1 features.
     - `hist_meal_mean` and `hist_center_mean`: Cumulative mean of `num_orders` for meal $m$ (or centre $c$) across weeks $1 \dots t-1$, falling back to `prior_global_mean` when a meal or centre has not yet appeared prior to week $t$.
     - `hist_pair_mean`, `hist_pair_roll4_mean`, and `hist_pair_std`: Within each `(center_id, meal_id)` series sorted by week, the expanding mean, 4-observation rolling mean (from shifted lags $1 \dots 4$), and sample standard deviation over strictly prior weeks $< t$. When a `(center_id, meal_id)` pair appears for the first time in training, `hist_pair_mean` falls back to $0.5 \times (\text{hist\_meal\_mean} + \text{hist\_center\_mean})$, `hist_pair_roll4_mean` falls back to `hist_pair_mean`, and `hist_pair_std` defaults to $0.0$.
   - **Within the Held-Out Test Set ($117 \le t \le 145$):** To ensure that neither feature engineering nor the preparation simulation uses actual test-period demand, all historical features for test observations are computed **exclusively from $\mathcal{D}_{\text{train}}$ (Weeks $1\text{–}116$)**. Specifically, `hist_pair_mean` and `hist_pair_std` are the full training-period mean and standard deviation per `(center_id, meal_id)`, and `hist_pair_roll4_mean` is the mean of the final up-to-4 training observations per pair prior to Week $117$.

3. **Preprocessing Pipeline (`ColumnTransformer`):**
   - **Numerical features ($17$ columns):** Median imputation (`SimpleImputer(strategy="median")`) followed by zero-mean, unit-variance standardization (`StandardScaler()`).
   - **Categorical features ($5$ columns: `category`, `cuisine`, `center_type`, `city_code`, `region_code`):** Most-frequent imputation (`SimpleImputer(strategy="most_frequent")`) followed by one-hot encoding (`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`).

### 8.3 Forecasting Models
We implement four models with non-negative prediction clipping ($\hat{D}_{c,m,t} = \max(0, \hat{y}_{c,m,t})$):
1. **Historical Average Baseline (`HistoricalAverageBaseline`):** A custom `scikit-learn` regressor that memorizes the training-period mean demand $\bar{D}^{\text{train}}_{c,m}$ for each `(center_id, meal_id)` pair over Weeks $1\text{–}116$, with hierarchical fallback to $0.5(\bar{D}^{\text{train}}_m + \bar{D}^{\text{train}}_c)$ and the global training mean for unseen pairs.
2. **Linear Regression (`LinearRegression`):** Ordinary least squares regression wrapped in a `scikit-learn` `Pipeline` with the `ColumnTransformer`, fitted on all $361{,}348$ training records.
3. **Random Forest Regressor (`RandomForestRegressor`):** An ensemble of $100$ decision trees (`n_estimators=100, max_depth=16, min_samples_leaf=5, random_state=42, n_jobs=-1`).
4. **Gradient Boosting Regressor (`GradientBoostingRegressor`):** A boosted ensemble of $150$ regression trees (`n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42`).

To maintain lightweight execution on a local workstation CPU, the two tree-ensemble models fit on the most recent $120{,}000$ chronological training records (`max_train_samples_tree=120000`, corresponding to the most recent training weeks leading up to Week $116$), while their input historical features retain full cumulative information from Week $1$.

### 8.4 Food Preparation Surplus and Shortage Simulation
To evaluate how demand forecasts perform when guiding operational meal preparation, we simulate a weekly preparation rule across all $N_{\text{test}} = 95{,}200$ held-out test records $(c, m, t) \in \mathcal{D}_{\text{test}}$. Let $D_{c,m,t} = \text{num\_orders}_{c,m,t}$ denote realized test demand and $\hat{D}_{c,m,t} \ge 0$ denote the point forecast produced by a given model using only pre-test demand history.

Under a proportional safety buffer $\alpha = 0.05$ ($+5.0\%$), the prepared quantity $Q_{c,m,t}$ is:
$$Q_{c,m,t} = \text{round}\big((1 + \alpha)\,\hat{D}_{c,m,t}\big) = \text{round}\big(1.05 \times \hat{D}_{c,m,t}\big)$$

Assuming prepared meals are perishable within the weekly planning window and cannot be carried over as fresh meals to week $t + 1$, each observation yields:
- **Simulated Surplus (Over-Preparation):**
  $$S^{+}_{c,m,t} = \max(0, \; Q_{c,m,t} - D_{c,m,t})$$
- **Simulated Shortage (Unmet Demand):**
  $$S^{-}_{c,m,t} = \max(0, \; D_{c,m,t} - Q_{c,m,t})$$
- **Served Demand:**
  $$\text{Served}_{c,m,t} = \min(Q_{c,m,t}, \; D_{c,m,t}) = Q_{c,m,t} - S^{+}_{c,m,t} = D_{c,m,t} - S^{-}_{c,m,t}$$

Aggregating across all $(c, m, t) \in \mathcal{D}_{\text{test}}$, we compute:
$$\text{Total Simulated Surplus} = \sum_{c,m,t} S^{+}_{c,m,t}, \qquad \text{Total Simulated Shortage} = \sum_{c,m,t} S^{-}_{c,m,t}$$
$$\text{Demand Served (\%)} = 100 \times \frac{\sum_{c,m,t} \text{Served}_{c,m,t}}{\sum_{c,m,t} D_{c,m,t}}$$
$$\text{Relative Surplus Change vs. Baseline (\%)} = 100 \times \frac{\text{Total Simulated Surplus}_{\text{policy}} - \text{Total Simulated Surplus}_{\text{baseline}}}{\text{Total Simulated Surplus}_{\text{baseline}}}$$

---

# 9. Experimental Setup

- **Software Environment:** Python `3.14.7`, `numpy 2.5.3`, `pandas 3.0.6`, `scikit-learn 1.9.1`, `matplotlib 3.11.2`, and `streamlit 1.65.0`.
- **Evaluation Metrics:** All models are evaluated on the exact same held-out test set ($\mathcal{D}_{\text{test}}, N = 95{,}200$) using:
  $$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i|, \qquad \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}, \qquad R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$
- **Automated Verification Suite (`tests/test_pipeline.py`):** Five unit and integration tests verify (1) missing-dataset error handling, (2) strict chronological splitting and zero test-target leakage, (3) zero contemporaneous target leakage in Week 1 and subsequent training weeks, (4) baseline policy invariance to test actuals, and (5) exact mathematical agreement between `results/predictions.csv`, `results/model_comparison.csv`, and `results/waste_simulation.csv`.

---

# 10. Results

### 10.1 Held-Out Test Set Forecasting Accuracy (RQ1)
Table 1 reports the forecasting performance of all four models on the held-out test period (Weeks $117\text{–}145$, $N = 95{,}200$), corresponding directly to `results/model_comparison.csv`. Visual comparisons of the metrics, weekly demand trajectories, and residual distributions are saved in `results/figures/model_comparison.png`, `results/figures/actual_vs_predicted.png`, and `results/figures/residual_analysis.png`.

**Table 1: Held-Out Test Set Forecasting Evaluation (Weeks 117–145, $N = 95{,}200$)**

| Model | MAE $\downarrow$ | RMSE $\downarrow$ | $R^2$ $\uparrow$ | MAE Change vs. Baseline | RMSE Change vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical Average Baseline** | `109.7701` | `248.9597` | `0.5353` | `0.00%` | `0.00%` |
| **Linear Regression** | `104.9161` | `220.4739` | `0.6356` | `-4.42%` | `-11.44%` |
| **Random Forest Regressor** | `82.5332` | `178.1692` | `0.7620` | `-24.81%` | `-28.43%` |
| **Gradient Boosting Regressor** | **`80.8407`** | **`172.8451`** | **`0.7760`** | **`-26.35%`** | **`-30.57%`** |

Key empirical findings from Table 1:
1. The **Historical Average Baseline** explains slightly more than half of the variance in weekly test demand ($R^2 = 0.5353, \text{MAE} = 109.7701, \text{RMSE} = 248.9597$), reflecting persistent baseline differences in popularity across `(center_id, meal_id)` combinations.
2. **Linear Regression** reduces MAE by $4.42\%$ (to $104.9161$) and RMSE by $11.44\%$ (to $220.4739$), lifting $R^2$ to $0.6356$. This confirms that incorporating linear terms for price discounts, promotional campaigns, and rolling historical averages improves upon static historical means.
3. The non-linear tree ensembles achieve substantial error reductions: **Random Forest Regressor** lowers MAE by $24.81\%$ ($\text{MAE} = 82.5332, \text{RMSE} = 178.1692, R^2 = 0.7620$), and **Gradient Boosting Regressor** achieves the best statistical forecasting performance across all three metrics, lowering MAE by **26.35%** (to **$80.8407$**), lowering RMSE by **30.57%** (to **$172.8451$**), and achieving **$R^2 = 0.7760$**.

![Figure 1: Held-Out Test Set Performance Comparison](results/figures/model_comparison.png)
*Figure 1 (`results/figures/model_comparison.png`): Comparison of Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$) across the four evaluated models on the held-out chronological test set (Weeks 117–145, $N = 95{,}200$).*

![Figure 2: Actual vs. Predicted Meal Demand](results/figures/actual_vs_predicted.png)
*Figure 2 (`results/figures/actual_vs_predicted.png`): Left: Weekly aggregated actual vs. predicted meal demand across test weeks 117–145. Right: Sample-level actual vs. predicted order scatter plot for the Gradient Boosting Regressor against the $y = x$ identity line.*

![Figure 3: Prediction Residual Analysis](results/figures/residual_analysis.png)
*Figure 3 (`results/figures/residual_analysis.png`): Left: Residuals ($y - \hat{y}$) vs. predicted demand for the Gradient Boosting Regressor. Right: Residual error distribution comparing the Gradient Boosting Regressor against the Historical Average Baseline.*

### 10.2 Food Preparation Simulation Results (RQ2 & RQ3)
Table 2 presents the operational simulation outputs across Weeks $117\text{–}145$ (`results/waste_simulation.csv`), illustrated in `results/figures/waste_simulation.png`. Total realized customer demand across the $95{,}200$ test records is **$23{,}618{,}862.0$** meal orders, and every policy applies the identical $+5.0\%$ safety buffer ($\alpha = 0.05$).

**Table 2: Simulated Meal Preparation Policy Comparison ($\alpha = 5.0\%$ Safety Buffer, Weeks 117–145)**

| Preparation Policy | Model Source | Safety Buffer (%) | Total Actual Demand | Total Prepared Units | Total Simulated Surplus | Total Simulated Shortage | Demand Served (%) | Relative Surplus Change vs. Baseline (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Policy (Historical Average)** | Historical Average Baseline | `5.0%` | `23,618,862.0` | `26,077,208.0` | `6,693,322.0` | `4,234,976.0` | `82.07%` | `0.00%` |
| **Prediction Policy (Linear Regression)** | Linear Regression | `5.0%` | `23,618,862.0` | `25,256,017.0` | `5,971,543.0` | `4,334,388.0` | `81.65%` | `-10.78%` |
| **Prediction Policy (Random Forest Regressor)** | Random Forest Regressor | `5.0%` | `23,618,862.0` | `25,039,075.0` | `4,753,258.0` | **`3,333,045.0`** | **`85.89%`** | `-28.99%` |
| **Prediction Policy (Gradient Boosting Regressor)** | Gradient Boosting Regressor | `5.0%` | `23,618,862.0` | `24,450,993.0` | **`4,350,038.0`** | `3,517,907.0` | `85.11%` | **`-35.01%`** |

Key simulation findings from Table 2:
1. **Baseline Policy Inefficiency:** Preparing $1.05 \times$ the training-period historical average generates $26{,}077{,}208$ prepared meal units, resulting in **$6{,}693{,}322$ simulated surplus units** and **$4{,}234{,}976$ simulated shortage units** ($82.07\%$ of total demand served).
2. **Linear Regression Policy Trade-off:** Guiding preparation with Linear Regression reduces total prepared units to $25{,}256{,}017$, lowering simulated surplus by **10.78%** (to $5{,}971{,}543$ units), but slightly increases simulated shortages from $4{,}234{,}976$ to $4{,}334{,}388$ units (reducing demand served by $0.42$ percentage points to $81.65\%$).
3. **Pareto Improvement by Tree Ensembles:** Both **Random Forest** and **Gradient Boosting** improve upon the Baseline Policy on *both* dimensions simultaneously:
   - **Random Forest Policy** reduces simulated surplus by **28.99%** (saving $1{,}940{,}064$ excess prepared units) while reducing simulated shortages by **21.30%** (from $4{,}234{,}976$ to $3{,}333{,}045$ units), achieving the highest customer service level of **85.89%** demand served ($+3.82$ percentage points above baseline).
   - **Gradient Boosting Policy (Best for Minimizing Simulated Surplus):** Achieves the largest reduction in simulated surplus—cutting surplus by **35.01%** (to **$4{,}350{,}038$** units, saving $2{,}343{,}284$ excess prepared units relative to baseline)—while also reducing simulated shortages by **16.93%** (to $3{,}517{,}907$ units) and serving **85.11%** of total demand ($+3.04$ percentage points above baseline).
   - **Random Forest Policy (Best for Minimizing Simulated Shortage and Maximizing Demand Served):** Achieves the lowest simulated shortage (**$3{,}333{,}045$** units, a **21.30%** reduction in unmet demand vs. baseline) and the highest service level (**85.89%** of demand served, $+3.82$ percentage points above baseline), while simultaneously reducing simulated surplus by **28.99%** (to $4{,}753{,}258$ units).

![Figure 4: Simulated Meal Preparation Policy Comparison](results/figures/waste_simulation.png)
*Figure 4 (`results/figures/waste_simulation.png`): Left: Total simulated surplus (over-preparation) and simulated shortage (unmet demand) across the four preparation policies ($\alpha = 5.0\%$ buffer). Right: Percentage of total test-period demand served by each policy. Note: Figures represent simulated operational proxies on order data, not measured physical food waste.*

### 10.3 Summary of Best Model by Evaluation Criterion

To clearly distinguish statistical point-forecast accuracy from each simulated operational metric, Table 3 summarizes the top-performing model under each criterion on the held-out test set.

**Table 3: Best-Performing Model by Statistical and Simulated Operational Criterion**

| Criterion Category | Evaluation Metric | Best Model / Policy | Verified Value | Comparison to Historical Baseline |
| :--- | :--- | :--- | :---: | :---: |
| **Statistical Forecasting** | Lowest MAE $\downarrow$ | **Gradient Boosting Regressor** | `80.8407` | `-26.35%` (vs. `109.7701`) |
| **Statistical Forecasting** | Lowest RMSE $\downarrow$ | **Gradient Boosting Regressor** | `172.8451` | `-30.57%` (vs. `248.9597`) |
| **Statistical Forecasting** | Highest $R^2$ $\uparrow$ | **Gradient Boosting Regressor** | `0.7760` | `+0.2407` (vs. `0.5353`) |
| **Operational Simulation ($\alpha = 5\%$)** | Lowest Simulated Surplus $\downarrow$ | **Prediction Policy (Gradient Boosting Regressor)** | `4,350,038.0` units | `-35.01%` (vs. `6,693,322.0`) |
| **Operational Simulation ($\alpha = 5\%$)** | Lowest Simulated Shortage $\downarrow$ | **Prediction Policy (Random Forest Regressor)** | `3,333,045.0` units | `-21.30%` (vs. `4,234,976.0`) |
| **Operational Simulation ($\alpha = 5\%$)** | Highest Demand Served (%) $\uparrow$ | **Prediction Policy (Random Forest Regressor)** | `85.89%` | `+3.82 pp` (vs. `82.07%`) |

---

# 11. Discussion

### 11.1 Why Non-Linear Models Simultaneously Reduce Simulated Surplus and Shortage
At first glance, reducing total prepared units (from $26.08\text{M}$ under the baseline to $24.45\text{M}$ under Gradient Boosting) while *increasing* the percentage of customer demand served (from $82.07\%$ to $85.11\%$) might appear counterintuitive. However, comparing the weekly trajectories in `results/figures/actual_vs_predicted.png` reveals the mechanism:
- Meal demand in the Genpact dataset exhibits pronounced spikes when a meal is featured on the homepage (`homepage_featured = 1`), promoted via email (`emailer_for_promotion = 1`), and offered at a discounted checkout price (`checkout_price < base_price`).
- The **Historical Average Baseline** spreads those promotional demand spikes across all weeks in the training period. Consequently, during normal unpromoted weeks in the test period, the historical average systematically *over-predicts* demand (creating large simulated surplus), whereas during promoted weeks it severely *under-predicts* demand (creating large simulated shortages).
- **Linear Regression** captures additive main effects of promotions and discounts ($R^2 = 0.6356$), but because promotional elasticity scales multiplicatively with a meal's baseline popularity and centre size, an additive linear model under-predicts high-volume promotional surges, explaining why its demand served ($81.65\%$) lags slightly behind the baseline ($82.07\%$) despite lower overall RMSE.
- **Random Forest** and **Gradient Boosting** capture the multiplicative interaction between baseline `(center_id, meal_id)` demand (`hist_pair_mean`, `hist_pair_roll4_mean`) and promotional/discount indicators. They allocate more prepared units specifically to promoted item-weeks (reducing stockouts) and scale back preparation during unpromoted item-weeks (reducing over-preparation surplus).

### 11.2 Distinction Between Prediction Accuracy and Simulated Operational Outcomes
Our findings illustrate why evaluating forecasting models alongside an operational policy simulation provides richer insight than point-prediction metrics alone:
- Although **Gradient Boosting** achieves a slightly lower RMSE ($172.8451$) and MAE ($80.8407$) than **Random Forest** ($\text{RMSE} = 178.1692, \text{MAE} = 82.5332$), **Random Forest** prepares slightly more aggregate units ($25.04\text{M}$ vs. $24.45\text{M}$) and achieves a slightly higher demand-served percentage ($85.89\%$ vs. $85.11\%$), whereas **Gradient Boosting** achieves a larger reduction in simulated surplus ($-35.01\%$ vs. $-28.99\%$).
- Depending on whether a fulfilment centre places a higher marginal penalty on unserved customer orders or on excess prepared inventory, operations managers can select between the Random Forest policy (higher service level) and the Gradient Boosting policy (lower simulated surplus) or tune the safety buffer $\alpha$.

---

# 12. Limitations and Threats to Validity

To ensure honest and transparent academic reporting, several methodological and dataset limitations must be emphasized:

1. **Simulated Surplus Is an Operational Proxy, Not Measured Physical Food Waste:**
   The Genpact Food Demand Forecasting dataset records fulfilled weekly customer order quantities (`num_orders`). It does **not** contain physical kitchen production logs, batch-cooking sizes, unplated bulk ingredients, shelf-life reuse across consecutive days, or post-consumer plate waste. Therefore, $\text{Surplus}_{c,m,t} = \max(0, Q_{c,m,t} - D_{c,m,t})$ represents **simulated over-preparation under the assumed preparation policy**, and must not be interpreted as empirically measured physical food waste reduction.
2. **Right-Censored Historical Demand:**
   Because `num_orders` records completed customer orders rather than unconstrained demand intent, any historical stockouts during Weeks $1\text{–}145$ would right-censor the observed target values.
3. **Static Historical Features Across the 29-Week Test Horizon:**
   To enforce a strict pre-test information boundary so that actual test-period demand (`Weeks 117–145`) never leaks into test predictions or preparation quantities, the historical demand features (`hist_pair_mean`, `hist_pair_roll4_mean`, `hist_pair_std`, `hist_meal_mean`, `hist_center_mean`) are frozen at the end of the training period (Week $116$) across all $29$ test weeks. In a rolling one-step-ahead production deployment where Week $t-1$ realized orders become known before preparing for Week $t$, updating `hist_pair_roll4_mean` weekly would likely yield further accuracy gains.
4. **Chronological Subsampling for Tree Ensembles:**
   While `HistoricalAverageBaseline` and `LinearRegression` fit on all $361{,}348$ training rows, `RandomForestRegressor` and `GradientBoostingRegressor` fit on the most recent $120{,}000$ chronological training rows (`max_train_samples_tree=120000`) to keep single-machine CPU training under five minutes. Although the historical features passed into those trees summarize all $116$ training weeks, training tree splits on a subset of rows is a computational trade-off.
5. **Absence of Statistical Significance Testing and Hyperparameter Optimization:**
   Each model was trained with fixed, documented hyperparameters (`random_state=42`) on a single chronological split. We do not claim formal statistical significance across multiple rolling origins or claim algorithmic novelty over state-of-the-art boosting libraries.

---

# 13. Conclusion and Future Work

This study presented **PlateBack Lite**, a reproducible machine learning pipeline for forecasting weekly meal demand and simulating meal-preparation policies on the 456,548-row Genpact Food Demand Forecasting dataset. Under a strictly leak-free chronological evaluation protocol (training on Weeks $1\text{–}116$; testing on Weeks $117\text{–}145$), **Gradient Boosting Regressor** ($\text{MAE} = 80.8407, \text{RMSE} = 172.8451, R^2 = 0.7760$) and **Random Forest Regressor** ($\text{MAE} = 82.5332, \text{RMSE} = 178.1692, R^2 = 0.7620$) substantially outperformed both **Linear Regression** ($\text{MAE} = 104.9161, \text{RMSE} = 220.4739, R^2 = 0.6356$) and the **Historical Average Baseline** ($\text{MAE} = 109.7701, \text{RMSE} = 248.9597, R^2 = 0.5353$). When evaluated in a weekly meal-preparation simulation with a $+5.0\%$ safety buffer, the Gradient Boosting and Random Forest policies reduced **simulated surplus** by **35.01%** and **28.99%**, respectively, while increasing the **percentage of customer demand served** from **82.07%** (baseline) to **85.11%** and **85.89%**.

Future extensions of this work include: (1) training quantile regression models under asymmetric pinball loss so that the safety buffer is learned directly from the conditional demand distribution rather than applied as a uniform $+5.0\%$ multiplier; (2) evaluating rolling walk-forward retraining across the test horizon; and (3) validating preparation policies on kitchen datasets that pair order logs with physical batch-production and leftover weight measurements.

---

# 14. References

1. Arrow, K. J., Harris, T., & Marschak, J. (1951). Optimal inventory policy. *Econometrica*, 19(3), 250–272. https://doi.org/10.2307/1906813
2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
3. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
4. Hübner, N., Caspers, J., Coroamă, V. C., & Finkbeiner, M. (2024). Machine-learning-based demand forecasting against food waste: Life cycle environmental impacts and benefits of a bakery case study. *Journal of Industrial Ecology*, 28(5), 1118–1131. https://doi.org/10.1111/jiec.13528
5. Panda, S. K., & Mohanty, S. N. (2023). Time series forecasting and modeling of food demand supply chain based on regressors analysis. *IEEE Access*, 11, 42679–42700. https://doi.org/10.1109/ACCESS.2023.3271159
6. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
7. Rodrigues, M., Migueis, V., Freitas, S., & Machado, T. (2024). Machine learning models for short-term demand forecasting in food catering services: A solution to reduce food waste. *Journal of Cleaner Production*, 435, 140265. https://doi.org/10.1016/j.jclepro.2023.140265

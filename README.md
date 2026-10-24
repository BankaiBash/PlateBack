# PlateBack Lite: Predicting Food Demand and Simulating Food Waste Reduction Using Machine Learning

**PlateBack Lite** is a reproducible machine learning research codebase designed to forecast weekly meal demand (`num_orders`) across fulfilment centres and evaluate the simulated effect of prediction-guided meal preparation policies on **simulated surplus** (over-preparation) and **simulated shortage** (unmet demand).

---

## Methodological Disclaimer

The public **Genpact Food Demand Forecasting** dataset records fulfilled customer order quantities (`num_orders`) and does **not** record physical kitchen preparation logs or measured food disposal. Consequently, all surplus and shortage quantities reported by this project are **simulated operational proxies** under explicit meal-preparation policy assumptions—they must **not** be claimed as empirically measured physical food waste.

---

## Project Structure

```text
.
├── data/                        # Place train.csv, meal_info.csv, fulfilment_center_info.csv here
├── paper/
│   ├── research_outline.md      # Academic paper outline (abstract through references)
│   └── results_template.md      # Tables template to populate after running experiments
├── results/                     # Generated outputs (CSV tables and summary metadata)
│   └── figures/                 # Generated diagnostic and comparison plots (.png)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Schema inspection, merging, chronological split, leak-free features
│   ├── train_models.py          # Baseline, Linear Regression, Random Forest, Gradient Boosting
│   ├── evaluate.py              # MAE, RMSE, R² evaluation and matplotlib visualizations
│   └── waste_simulation.py      # Leak-free meal preparation surplus/shortage simulation
├── tests/
│   └── test_pipeline.py         # Automated pytest suite verifying pipeline & zero data leakage
├── app.py                       # Streamlit interactive dashboard
├── run_pipeline.py              # Command-line 7-step end-to-end research workflow
├── requirements.txt             # Python dependencies
└── .gitignore
```

---

## 1. Obtaining and Placing the Dataset

This repository does not bundle the raw CSV files or fabricate experimental outputs. To run the experiments on the real dataset:

1. Obtain the public **Genpact Food Demand Forecasting** dataset (originally published for the Analytics Vidhya Genpact Machine Learning Hackathon and publicly mirrored on Kaggle under *"Food Demand Forecasting"*).
2. Extract the following three required CSV files:
   - `train.csv` (expected columns: `id`, `week`, `center_id`, `meal_id`, `checkout_price`, `base_price`, `emailer_for_promotion`, `homepage_featured`, `num_orders`)
   - `meal_info.csv` (expected columns: `meal_id`, `category`, `cuisine`)
   - `fulfilment_center_info.csv` (expected columns: `center_id`, `city_code`, `region_code`, `center_type`, `op_area`)
3. Copy all three CSV files into the `data/` directory:
   ```text
   data/train.csv
   data/meal_info.csv
   data/fulfilment_center_info.csv
   ```

---

## 2. Installation

Install the required dependencies (Python 3.10+):

```bash
pip install -r requirements.txt
```

---

## 3. Running the Complete Workflow (Command Line)

Run the end-to-end 7-step pipeline from the command line without opening the dashboard:

```bash
python run_pipeline.py
```

Optional CLI flags:
```bash
python run_pipeline.py --data-dir data --results-dir results --test-weeks-ratio 0.20 --safety-buffer 0.05 --random-state 42
```

When executed, `run_pipeline.py`:
1. Verifies that `train.csv`, `meal_info.csv`, and `fulfilment_center_info.csv` exist in `data/`.
2. Inspects and prints the actual columns, data types, and missing value counts for each file.
3. Merges metadata, performs a chronological train/test split by `week`, and engineers strictly leak-free features.
4. Trains the 4 models:
   - **Historical Average Baseline**
   - **Linear Regression**
   - **Random Forest Regressor**
   - **Gradient Boosting Regressor**
5. Evaluates all models on the held-out test set using **MAE**, **RMSE**, and **$R^2$**, saving `results/model_comparison.csv`.
6. Runs the food preparation surplus/shortage simulation, saving `results/waste_simulation.csv`.
7. Generates and saves publication figures in `results/figures/`:
   - `results/figures/actual_vs_predicted.png`
   - `results/figures/model_comparison.png`
   - `results/figures/residuals.png`
   - `results/figures/waste_simulation.png`

---

## 4. Running the Streamlit Dashboard

To explore the dataset overview, model comparison metrics, actual vs. predicted charts, and food waste simulation results interactively:

```bash
streamlit run app.py
```

---

## 5. Running Automated Tests

To verify schema validation, chronological splitting, strict leakage prevention, model training, evaluation metrics, and simulation logic using temporary synthetic test fixtures (without writing fake results to `results/`):

```bash
python -m unittest discover -s tests -v
```
*(or `pytest -v` if `pytest` is installed)*

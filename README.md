# Lending Club Credit Risk and Risk-Adjusted Lending Returns

## Executive summary

This project turns Lending Club loan data into a credit-risk decision workflow. It compares feature pipelines for linear, LightGBM, Random Forest, and XGBoost models, then connects default probabilities to expected cash flows, IRR, and a risk-adjusted Sharpe-ratio threshold decision.

The portfolio value is the bridge between **predictive modeling** and a **lending business decision**: a probability estimate is only useful when it changes which loans an investor would accept at an acceptable risk-adjusted return.

## Business question

> Given a borrower's application and credit history, which loans should an investor accept, and how should the acceptance threshold be chosen when default risk affects cash flows?

## Workflow

1. **EDA**: inspect FICO, income, utilization, inquiries, delinquency, and balance distributions.
2. **Feature engineering**: create missingness flags, parse percentages and dates, and prepare separate representations for linear and tree-based models.
3. **Model comparison**: train LightGBM, Random Forest, and XGBoost variants with reproducible seeds.
4. **Decision layer**: convert predicted default probabilities into loan cash flows, IRR, and a validation-set threshold search for Sharpe ratio.
5. **Explainability**: generate mean absolute SHAP feature importance for the tree-based model.

## Modeling choices

| Model family | Preprocessing rationale |
|---|---|
| Linear models | Mean imputation plus missingness indicators and one-hot encoding |
| LightGBM | Native missing-value handling and categorical features |
| Random Forest / XGBoost | Label-encoded categorical features and shared numeric/date transformations |
| Decision rule | Threshold selected on validation Sharpe, then frozen for test evaluation |

The current model script uses a fixed random seed, a 20,000-row working sample, and a 60/20/20 train/validation/test split for the exploratory comparison. For production-quality inference, the next version should use a chronological split and out-of-time vintage validation.

## Repository map

```text
notebooks/
├── 01_EDA.ipynb
├── 02_Feature_Engineering.ipynb
└── 03_Model_Comparison.ipynb
archive/lendingclub_1st/
├── 01_initial_data_definition.ipynb
├── 02_default_labeling.ipynb
├── 03_baseline_model.ipynb
├── 04_feature_selection.ipynb
└── 05_information_set_review.ipynb
src/
├── data/                          # dataset construction and feature lists
├── models/                        # LightGBM, Random Forest, XGBoost
├── pipelines/                     # end-to-end orchestration
└── utils/                         # cash flows, IRR, Sharpe, risk-free rates
reports/figures/                   # exploratory and model figures
data/raw/                          # local raw Lending Club data (not committed)
data/processed/                   # local feature matrices (not committed)
```

The `archive/lendingclub_1st/` folder is the earliest Lending Club study in this account's oldest repository, preserved here so the portfolio shows the full analytical progression rather than only the final pipeline.

## Reproduce

1. Install the dependencies with `pip install -r requirements.txt`.
2. Obtain the Lending Club source data and place it under `data/raw/`.
3. For the risk-adjusted return models, export `FRED_API_KEY` (see `.env.example`).
4. Run the feature-building scripts under `src/data/`.
5. Generate the feature list for the selected preprocessing family; the default is LightGBM:
   - `python src/data/generate_features_list.py --model lightgbm`
   - `python src/data/generate_features_list.py --model tree`
   - `python src/data/generate_features_list.py --model linear`
6. Run the model scripts under `src/models/`, then inspect the generated threshold and SHAP plots.

To run the lightweight regression checks for the model-specific feature-list generator:

```bash
python -m unittest discover -s tests -v
```

Raw and processed CSV files are intentionally excluded from GitHub: the raw file is large and the repository should not redistribute the source dataset. The `.gitignore` also excludes local secrets and Python cache files.

## Limitations and next steps

- The exploratory 60/20/20 random split can leak vintage information; replace it with an out-of-time split for a defensible credit estimate.
- Threshold search needs explicit transaction/servicing costs, loss-given-default assumptions, and confidence intervals.
- The repository would be stronger with a pinned environment file, non-empty end-to-end notebooks, and a tracked model-card style results table.
- The curated notebooks now provide a readable `01 → 02 → 03` path; the underlying scripts remain the source of truth for execution.
- `generate_features_list.py` uses an explicit `--model` argument, so switching between LightGBM, linear, and tree features no longer requires commenting and uncommenting source lines.

## Portfolio signal

This project shows end-to-end analytical judgment: data quality checks, model-specific preprocessing, explainability, cash-flow modeling, and a risk-adjusted business decision rather than a model metric alone.

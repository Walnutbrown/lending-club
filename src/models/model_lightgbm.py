"""Single-seed LightGBM credit-risk model with a cash-flow decision layer."""

import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.calculate_sharpe import calculate_irr, calculate_sharpe
from utils.fetch_risk_free_rate import apply_risk_free_rate, load_risk_free_series
from utils.make_cashflow import get_cash_flow


def _attach_cf_irr_and_sharpe(df, threshold):
    """Apply a default-probability threshold and calculate portfolio Sharpe."""
    scored = df.copy()
    scored["cash_flow"] = scored.apply(
        lambda row: get_cash_flow(row) if row["pred_prob"] <= threshold else np.nan,
        axis=1,
    )
    scored["irr"] = scored["cash_flow"].apply(
        lambda cash_flow: calculate_irr(cash_flow)
        if isinstance(cash_flow, list) and cash_flow
        else np.nan
    )
    scored["irr"] = scored["irr"].fillna(scored["risk_free_rate"])
    return calculate_sharpe(scored["irr"].to_numpy(), scored["risk_free_rate"].to_numpy())


def main():
    data_path = PROJECT_DIR / "data" / "processed" / "lendingclub_features_for_lightgbm.csv"
    feature_path = PROJECT_DIR / "data" / "processed" / "features_final_list_lightgbm.csv"
    df = pd.read_csv(data_path)
    print(f"🔍 원본 데이터 크기: {df.shape}")

    for col in ["issue_d", "last_pymnt_d"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    rate_3y, rate_5y = load_risk_free_series()
    df = apply_risk_free_rate(df, rate_3y, rate_5y)

    features = pd.read_csv(feature_path)["feature"].dropna().tolist()
    features = [feature for feature in features if feature in df.columns and feature != "default"]
    categorical_cols = df[features].select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    seed = 42
    df_temp = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_end, val_end = int(len(df_temp) * 0.6), int(len(df_temp) * 0.8)
    train = df_temp.iloc[:train_end].copy()
    val = df_temp.iloc[train_end:val_end].copy()
    test = df_temp.iloc[val_end:].copy()

    # Keep the exploratory run reproducible and light enough for a portfolio demo.
    if len(train) > 20_000:
        train, _ = train_test_split(
            train,
            train_size=20_000,
            stratify=train["default"],
            random_state=seed,
        )
        train = train.reset_index(drop=True)
    print(f"🔍 학습셋 크기: {train.shape}")

    train_data = lgb.Dataset(train[features], label=train["default"], categorical_feature=categorical_cols)
    val_data = lgb.Dataset(val[features], label=val["default"], categorical_feature=categorical_cols)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": seed,
    }
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    val["pred_prob"] = model.predict(val[features])
    test["pred_prob"] = model.predict(test[features])

    # SHAP is calculated from the validation set for recruiter-facing explainability.
    try:
        import shap

        shap_values = model.predict(val[features], pred_contrib=True)[:, :-1]
        shap.summary_plot(shap_values, val[features], plot_type="dot", max_display=10, show=False)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("SHAP가 설치되지 않아 설명가능성 그래프를 건너뜁니다.")

    threshold_grid = np.linspace(0.05, 0.95, 200)
    val_sharpes = [_attach_cf_irr_and_sharpe(val, threshold) for threshold in threshold_grid]
    finite = np.isfinite(val_sharpes)
    if not finite.any():
        raise RuntimeError("유효한 validation Sharpe가 없어 threshold를 선택할 수 없습니다.")
    best_idx = int(np.nanargmax(val_sharpes))
    best_threshold = float(threshold_grid[best_idx])
    best_val_sharpe = float(val_sharpes[best_idx])
    test_sharpe = _attach_cf_irr_and_sharpe(test, best_threshold)
    baseline_sharpe = _attach_cf_irr_and_sharpe(test, 1.0)
    print(f"Validation best threshold={best_threshold:.2f}; Sharpe={best_val_sharpe:.4f}")
    print(f"Test Sharpe={test_sharpe:.4f}; threshold=1 baseline={baseline_sharpe:.4f}")

    fpr, tpr, _ = roc_curve(test["default"], test["pred_prob"])
    auc_score = roc_auc_score(test["default"], test["pred_prob"])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("ROC Curve — Test Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_grid, val_sharpes, label="Validation Sharpe")
    plt.axvline(best_threshold, color="red", linestyle="--", label=f"Best = {best_threshold:.2f}")
    plt.xlabel("Default-probability threshold")
    plt.ylabel("Sharpe ratio")
    plt.title("Validation Sharpe by Decision Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

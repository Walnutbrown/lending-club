import sys
import os
import lightgbm as lgb

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))  # lendingclub_2nd
src_dir = os.path.join(project_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _attach_cf_irr_and_sharpe(df, threshold):
    df['cash_flow'] = df.apply(
        lambda row: create_cash_flow(row) if row['pred_prob'] <= threshold else np.nan,
        axis=1
    )

    df['irr'] = df.apply(
        lambda row: calculate_irr(row['cash_flow']) if isinstance(row['cash_flow'], list) and len(row['cash_flow']) > 0 else row['risk_free_rate'],
        axis=1
    )
    df['irr'] = df['irr'].fillna(df['risk_free_rate'])
    
    # 🔍 디버깅: Sharpe 계산 전 IRR, risk-free, excess 통계 확인
    excess = df['irr'].values - df['risk_free_rate'].values
    std = np.nanstd(excess, ddof=1)
    if std == 0:
        return 0.0
    return np.nanmean(excess) / std

from utils.make_cashflow import create_cash_flow
from utils.fetch_risk_free_rate import load_risk_free_series, apply_risk_free_rate
from utils.calculate_sharpe import calculate_irr, calculate_sharpe


def main():
    # 1. 데이터 로딩
    df = pd.read_csv('../../data/processed/lendingclub_features_for_lightgbm.csv')
    print(f"🔍 원본 데이터 크기: {df.shape}")


    # 날짜 형식 변환
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors = 'coerce')
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], errors = 'coerce')

    # 2. Risk‑free rate 붙이기 ── Sharpe 계산용
    rate_3y, rate_5y = load_risk_free_series()
    df = apply_risk_free_rate(df, rate_3y, rate_5y)

    # 3. 전처리 및 변수 호출
    features = pd.read_csv('../../data/processed/features_final_list_lightgbm.csv')
    features = features['feature'].squeeze().tolist()
    if 'default' in features:
        features.remove('default')

    # object 타입을 category로 변환
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    cat_features = [c for c in categorical_cols if c in features]
   
    seed = 42
    # 5-1. 무작위 셔플
    df_temp = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df_temp)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train = df_temp.iloc[:train_end]
    val = df_temp.iloc[train_end:val_end]
    test = df_temp.iloc[val_end:]

    # Downsample only the training set
    from sklearn.model_selection import train_test_split
    _, train = train_test_split(
        train,
        train_size=20000,
        stratify=train['default'],
        random_state=42
    )
    train = train.reset_index(drop=True)
    print(f"🔍 학습셋 다운샘플링 후 크기: {train.shape}")
    
    X_train = train[features]
    y_train = train['default']
    X_val = val[features]
    y_val = val['default']

    X_test = test[features]
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': seed,
    }

    from lightgbm import early_stopping

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        callbacks=[early_stopping(stopping_rounds=30)]
    )

    val['pred_prob']  = model.predict(X_val)
    # SHAP explainability using LightGBM's built-in SHAP calculation
    import shap
    shap_values = model.predict(X_val, pred_contrib=True)
    shap_values_no_bias = shap_values[:, :-1]

    # SHAP summary: 점 그래프 (beeswarm)
    shap.summary_plot(shap_values_no_bias, X_val, plot_type="dot", max_display=10)

    # ── ① threshold grid search on val ──
    threshold_grid = np.linspace(0.05, 0.95, 200)   
    val_sharpes = []
    for th in threshold_grid:
        val_copy = val.copy()
        s = _attach_cf_irr_and_sharpe(val_copy, th)
        if np.isnan(s):
            print(f"⚠️ Threshold {th:.4f}: Sharpe = NaN")
            print(f"   → 유효 cash_flow 개수 = {(~val_copy['cash_flow'].isna()).sum()}")
            print(f"   → 유효 IRR 개수 = {(~val_copy['irr'].isna()).sum()}")
            excess = val_copy['irr'].values - val_copy['risk_free_rate'].values
            print(f"   → excess 고유값들: {np.unique(excess)}")
            print(f"   → excess 표준편차: {np.nanstd(excess, ddof=1)}")
        val_sharpes.append(s)

    best_idx        = int(np.nanargmax(val_sharpes))
    best_threshold  = threshold_grid[best_idx]
    best_val_sharpe = val_sharpes[best_idx]
    print(f"Seed {seed}: best threshold={best_threshold:.2f}  val‑Sharpe={best_val_sharpe:.4f}")

    # 🔍 IRR과 Risk-Free 분석
    val_copy = val.copy()
    _ = _attach_cf_irr_and_sharpe(val_copy, best_threshold)

    print("==== IRR & Risk-Free 분석 ====")
    print("IRR 분포:", val_copy['irr'].describe())
    print("Risk-Free 분포:", val_copy['risk_free_rate'].describe())
    print("Excess Return 평균:", np.mean(val_copy['irr'] - val_copy['risk_free_rate']))

    plt.figure(figsize=(10, 6))
    plt.plot(threshold_grid, val_sharpes, label="Sharpe Ratio")
    plt.axvline(best_threshold, color="red", linestyle="--", label=f"Best Threshold = {best_threshold:.2f}")
    plt.title("Sharpe Ratio by Threshold (Validation Set)")
    plt.xlabel("Threshold")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ── AUC Visualization ──
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(test['default'], test['pred_prob'])
    auc_score = roc_auc_score(test['default'], test['pred_prob'])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve — Test Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ── ② apply best threshold to test & compute Sharpe ──
    test_sharpe = _attach_cf_irr_and_sharpe(test, best_threshold)
    print(f"📊 Test Set Sharpe Ratio at best threshold ({best_threshold:.2f}): {test_sharpe:.4f}")

    # Sharpe at threshold = 1.0 for comparison
    sharpe_at_1 = _attach_cf_irr_and_sharpe(test.copy(), 1.0)

    # Bar plot comparing both
    plt.figure(figsize=(6, 5))
    plt.bar(['Best Threshold', 'Threshold = 1'], [test_sharpe, sharpe_at_1], color=['skyblue', 'lightcoral'])
    plt.ylabel("Sharpe Ratio")
    plt.title("Test Set Sharpe Ratio Comparison")
    plt.ylim(0, max(test_sharpe, sharpe_at_1) * 1.2)
    plt.grid(axis='y')
    plt.show()

if __name__ == "__main__":
    main()
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
    return calculate_sharpe(df['irr'].values, df['risk_free_rate'].values)

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
   
    # 4. 결과 저장용 리스트
    sharpe_ratios = []
    val_sharpe_ratios = []

    # 5. 100번 반복
    for seed in range(100):
        # 5-1. 무작위 셔플
        df_temp = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        n = len(df_temp)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        train = df_temp.iloc[:train_end]
        val = df_temp.iloc[train_end:val_end]
        test = df_temp.iloc[val_end:]
        
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
        # 🔥 EDA check for potential issues before threshold search
        print("==== EDA check ====")
        print("loan_amnt NaN 비율:", val['loan_amnt'].isna().mean())
        print("loan_amnt <= 0 비율:", (val['loan_amnt'] <= 0).mean())
        print("term NaN 비율:", val['term'].isna().mean())
        print("default NaN 비율:", val['default'].isna().mean())
        print("last_pymnt_num NaN 비율:", val['last_pymnt_num'].isna().mean())
        print("recoveries NaN 비율:", val['recoveries'].isna().mean())
        print("collection_recovery_fee NaN 비율:", val['collection_recovery_fee'].isna().mean())
        print("====================")
        test['pred_prob'] = model.predict(X_test)

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

        # ── ② apply best threshold to test & compute Sharpe ──
        test_sharpe = _attach_cf_irr_and_sharpe(test, best_threshold)

        sharpe_ratios.append(test_sharpe)
        val_sharpe_ratios.append(best_val_sharpe)

    # 6. 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(val_sharpe_ratios, bins=30, edgecolor='black')
    plt.title('Distribution of Sharpe Ratios (100 repetitions) — validation set')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(sharpe_ratios, bins=30, edgecolor='black')
    plt.title('Distribution of Sharpe Ratios (100 repetitions) — validation set')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # 7. 결과 저장
    result_df = pd.DataFrame({'Val Sharpe': val_sharpe_ratios,
                              'Test Sharpe': sharpe_ratios})
    result_df.to_csv('../../reports/sharpe_distribution_lightgbm.csv', index=False)
    print("🎯 Sharpe ratio 분포 저장 완료!")

if __name__ == "__main__":
    main()

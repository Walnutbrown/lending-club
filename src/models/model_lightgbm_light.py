import sys
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.make_cashflow import get_cash_flow
from utils.fetch_risk_free_rate import load_risk_free_series, apply_risk_free_rate
from utils.calculate_sharpe import get_irr, calculate_sharpe, precompute_cashflow_and_irr, compute_sharpe_for_threshold

def _attach_cf_irr_and_sharpe_cached(df, threshold):
    df = df.copy()
    df['cash_flow'] = df.apply(
        lambda row: get_cash_flow(row) if row['pred_prob'] <= threshold else np.nan,
        axis=1
    )
    df['irr'] = df['cash_flow'].apply(
        lambda cf: get_irr(cf) if isinstance(cf, list) and len(cf) > 0 else np.nan
    )
    df['irr'] = df['irr'].fillna(df['risk_free_rate'])
    return calculate_sharpe(df['irr'].values, df['risk_free_rate'].values)



def main():
    # 1. 데이터 로딩
    data_path = PROJECT_DIR / "data" / "processed" / "lendingclub_features_for_lightgbm.csv"
    feature_path = PROJECT_DIR / "data" / "processed" / "features_final_list_lightgbm.csv"
    df = pd.read_csv(data_path)
    print(f"🔍 원본 데이터 크기: {df.shape}")

    # 날짜 형식 변환
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors = 'coerce')
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], errors = 'coerce')

    # 2. Risk‑free rate 붙이기 ── Sharpe 계산용
    rate_3y, rate_5y = load_risk_free_series()
    df = apply_risk_free_rate(df, rate_3y, rate_5y)

    # 3. 전처리 및 변수 호출
    features = pd.read_csv(feature_path)['feature'].dropna().tolist()
    features = [feature for feature in features if feature in df.columns]
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
    sharpe_at_1 = []

    df_indices = np.arange(len(df))

    # 5. 100번 반복
    for seed in range(100):
        # 5-1. 무작위 셔플
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(df_indices)

        n = len(df)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        train_idx = shuffled_indices[:train_end]
        val_idx = shuffled_indices[train_end:val_end]
        test_idx = shuffled_indices[val_end:]

        train = df.iloc[train_idx].copy()
        val = df.iloc[val_idx].copy()
        test = df.iloc[test_idx].copy()
        print(f"🔍 Seed {seed}: train={len(train)}, val={len(val)}, test={len(test)}")

        # 5-2. LightGBM 모델 학습
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
        test['pred_prob'] = model.predict(X_test)
        val  = precompute_cashflow_and_irr(val)
        test = precompute_cashflow_and_irr(test)

        # ── ① threshold grid search on val ──
        threshold_grid = np.linspace(0.05, 0.95, 100)
        val_sharpes = [compute_sharpe_for_threshold(val, th) for th in threshold_grid]
        sharpe1 = compute_sharpe_for_threshold(test, 1)
        sharpe_at_1.append(sharpe1)


        best_idx        = int(np.nanargmax(val_sharpes))
        best_threshold  = threshold_grid[best_idx]
        best_val_sharpe = val_sharpes[best_idx]
        print(f"Seed {seed}: best threshold={best_threshold:.2f}  val‑Sharpe={best_val_sharpe:.4f}")

        # ── ② apply best threshold to test & compute Sharpe ──
        test_sharpe = _attach_cf_irr_and_sharpe_cached(test, best_threshold)

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
    plt.axvline(np.mean(sharpe_at_1), color='red', linestyle='dashed', linewidth=1, label='Sharpe at threshold = 1')
    plt.title('Distribution of Sharpe Ratios (100 repetitions) — validation set')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # 7. 결과 저장
    result_df = pd.DataFrame({'Val Sharpe': val_sharpe_ratios,
                              'Test Sharpe': sharpe_ratios})
    output_path = PROJECT_DIR / "reports" / "sharpe_distribution_lightgbm.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print("🎯 Sharpe ratio 분포 저장 완료!")

if __name__ == "__main__":
    main()

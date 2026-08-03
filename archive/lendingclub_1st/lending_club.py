#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
import shap
import optuna
from sklearn.metrics import log_loss
import numpy_financial as nf
warnings.filterwarnings('ignore')



# In[2]:


df = pd.read_csv('lendingclub.csv')
df['recoveries'] = df['recoveries'].fillna(0)
df.head()

# --- 포함 여부 = O 로 확정된 100개 변수 리스트 ---
include_vars = [
    'acc_now_delinq', 'acc_open_past_24mths', 'addr_state', 'all_util',
    'annual_inc', 'annual_inc_joint', 'application_type', 'avg_cur_bal',
    'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths',
    'collections_12_mths_ex_med', 'delinq_2yrs',
    'delinq_amnt', 'desc', 'dti', 'dti_joint', 'earliest_cr_line',
    'emp_length', 'fico_range_high', 'fico_range_low', 'home_ownership',
    'il_util', 'inq_fi', 'inq_last_12m', 'inq_last_6mths', 'max_bal_bc',
    'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
    'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_last_delinq',
    'mths_since_last_major_derog', 'mths_since_last_record',
    'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
    'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
    'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
    'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
    'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
    'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
    'num_tl_op_past_12m', 'open_acc', 'open_acc_6m', 'open_act_il',
    'open_il_12m', 'open_il_24m', 'open_rv_12m', 'open_rv_24m',
    'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec', 'pub_rec_bankruptcies',
    'purpose', 'revol_bal', 'revol_util', 'tax_liens',
    'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_acc',
    'total_bal_ex_mort', 'total_bal_il', 'total_bc_limit', 'total_cu_tl',
    'total_il_high_credit_limit', 'total_rev_hi_lim', 'verification_status',
    'verified_status_joint', 'revol_bal_joint', 'sec_app_fico_range_low',
    'sec_app_fico_range_high', 'sec_app_earliest_cr_line',
    'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc',
    'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts',
    'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med',
    'sec_app_mths_since_last_major_derog'
]


# In[3]:


df.info()
df = df.drop(columns = ['id'])


# In[4]:


# object 타입 컬럼만 추출
object_cols = df.select_dtypes(include='object').columns

# 숫자로 변환 가능한 문자열이 있는지 확인하고, 샘플 출력
for col in object_cols:
    sample = df[col].dropna().astype(str).head(100)  # 결측 제외 후 100개만 확인
    numeric_like = sample.apply(lambda x: x.replace('%','').replace(',','').replace('$','').strip().replace('+','').replace('<','')).str.replace(r'\.','', regex=True).str.isdigit()


# In[5]:


# 부도 1 부도아님 0
df['default'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)

# % 제거 후 실수 처리
df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float) / 100
df['revol_util'] = df['revol_util'].str.replace('%', '').astype(float) / 100                  


# In[6]:


# emp_length 전처리: 범주형 -> 숫자형
def parse_emp_length(x):
    if pd.isnull(x):
        return None
    
    x = str(x)
    
    if '<' in x:
        return 0
    if '10+' in x:
        return 10
    digits = ''.join(filter(str.isdigit, x))
    return int(digits) if digits else None

df['emp_length'] = df['emp_length'].apply(parse_emp_length)
                                                 
# term 전처리
df['term'] = df['term'].str.extract(r'(\d+)').astype(float) / 12

# earliest_cr_line 전처리: 연도 추출
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors = 'coerce').dt.year
df['sec_app_earliest_cr_line'] = pd.to_datetime(df['sec_app_earliest_cr_line'], errors='coerce').dt.year

# LightGBM은 결측치를 자동 처리하므로 별도 nanflag, 0 대체 없이 그대로 둠
pass

# --- 최종 feature 리스트 재구성 (딕셔너리 O표시 + 결측치 파생 포함) ---
# 1) 딕셔너리에서 O로 표시된 base feature
base_features = [col for col in include_vars if col in df.columns]

# 2) base feature에서 파생된 nanflag / interaction 컬럼 자동 포함
features = base_features

# 종속변수 제외
included_features = [col for col in features if col != 'default']

# 포함 여부 확인
if not isinstance(included_features, list):
    raise TypeError("included_features는 리스트여야 합니다.")

# 희귀 카테고리 압축 함수 정의
def compress_rare(series, top_n=10, name='Other'):
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), other=name)

# 희귀 카테고리 압축: 고유값 10개 초과 object/category 컬럼 전체 적용
for col in df.select_dtypes(include=['object', 'category']).columns:
    if df[col].nunique() > 10:
        df[col] = compress_rare(df[col], top_n=10)

# --- 범주형 변수 인코딩 및 목록 정의 ---
# LightGBM은 범주형을 int 코드로 받는다. object/category dtype → 코드 변환
categorical_cols = [
    col for col in features
    if str(df[col].dtype) in ['object', 'category']
]
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

# In[8]:


# 전체 데이터에서 X, y 분리 (features 리스트 기반으로만 선택)
X_total = df[features].copy()
 
y_total = df['default']

## 클래스 비율 확인
print("✅ 클래스 비율 (1 = 부도):")
print(y_total.value_counts(normalize=True))

## 다수 클래스 다운샘플링
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# 전체 데이터에서 분할 (60% 학습, 20% 검증, 20% 테스트)
X_full = df[features].copy()
y_full = df['default']

X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
    X_full, y_full, test_size=0.4, stratify=y_full, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_holdout, y_holdout, test_size=0.5, stratify=y_holdout, random_state=42
)

# 학습셋만 다운샘플링 (75:25 기준)
df_train = pd.concat([X_train_full, y_train_full], axis=1)
df_majority = df_train[df_train['default'] == 0]
df_minority = df_train[df_train['default'] == 1]

target_ratio = 0.25
desired_majority_size = int(len(df_minority) / target_ratio - len(df_minority))

df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=desired_majority_size,
    random_state=42
)

df_downsampled = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)

X_sel = df_downsampled.drop(columns=['default'])
y_sel = df_downsampled['default']

# 2. LightGBM 모델 하이퍼파라미터 최적화
def objective(trial):
    param = {
        'n_estimators': 500,
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'min_child_samples': trial.suggest_int("min_child_samples", 10, 100),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 5.0),
        'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 5.0),
        'random_state': 42,
        'n_jobs': -1
    }

    model = LGBMClassifier(verbose=-1, **param)
    trial_features = included_features[:15]  # fallback in case global trial_features not yet defined
    # Use precomputed top_15_cols defined outside the objective function
    model.fit(
        X_sel[trial_features], y_sel,
        eval_set=[(X_val[trial_features], y_val)],
        eval_metric='logloss',
        callbacks=[early_stopping(stopping_rounds=30, verbose=False)]
    )
    preds = model.predict_proba(X_val[trial_features])
    return log_loss(y_val, preds)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("✅ 최적 파라미터:")
print(study.best_params)

best_params = study.best_params
# Update best_params with fixed parameters
best_params.update({
    'n_estimators': 500,
    'random_state': 42,
    'n_jobs': -1
})

#3. ✅ 최적 하이퍼파라미터 적용한 모델로 최종 top_30_cols 다시 추출
model = LGBMClassifier(verbose=-1, **best_params)
model.fit(X_sel[included_features], y_sel)

importances = model.booster_.feature_importance(importance_type='gain')
feature_importance_df = pd.DataFrame({'feature': X_sel.columns, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
top_15_cols = feature_importance_df.head(30)['feature'].tolist()
print("📌 LGBM Feature Importance (순수 중요도 기준 상위 30):")
for i, (f, v) in enumerate(zip(feature_importance_df.head(30)['feature'], feature_importance_df.head(30)['importance'])):
    print(f"{f:35} → {v}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
top_30_df = feature_importance_df.head(30)
plt.barh(top_30_df['feature'][::-1], top_30_df['importance'][::-1])
plt.xlabel('Importance (Gain)')
plt.title('Top 30 Feature Importances (Gain)')
plt.tight_layout()
plt.show()

# model training block for final top_30_cols
model.fit(
    X_sel[top_15_cols], y_sel,
    eval_set=[(X_val[top_15_cols], y_val)],
    eval_metric='logloss',
    callbacks=[early_stopping(stopping_rounds=30, verbose=False)]
)

# Compute and plot AUC for validation and test sets
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# AUC for validation set
y_pred_proba_val = model.predict_proba(X_val[top_15_cols])[:, 1]
auc_val = roc_auc_score(y_val, y_pred_proba_val)
print(f"📊 Validation AUC: {auc_val:.4f}")

# AUC for test set
y_pred_proba_test = model.predict_proba(X_test[top_15_cols])[:, 1]
auc_test = roc_auc_score(y_test, y_pred_proba_test)
print(f"📊 Test AUC: {auc_test:.4f}")

# ROC Curve
fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_proba_val)
fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_val, tpr_val, label=f'Validation AUC = {auc_val:.2f}')
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test:.2f}', linestyle='--')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# 4. validation set에서 threshold 찾기
X_val_selected = X_val[top_15_cols]
y_pred_proba_val = model.predict_proba(X_val_selected)[:, 1]

collection_recovery_fee_val = df.loc[X_val_selected.index, 'collection_recovery_fee'].values.astype('float32')
# 실제 수익률 계산: (회수액 + 총상환액 - 대출금 - 추심비) / 대출금
loan_amnts_val = df.loc[X_val_selected.index, 'loan_amnt'].values.astype('float32')
total_pymnt_val = df.loc[X_val_selected.index, 'total_pymnt'].values.astype('float32')
recoveries_val = df.loc[X_val_selected.index, 'recoveries'].values.astype('float32')

# 대출 기간 (연 단위)
loan_term_val = df.loc[X_val_selected.index, 'term'].values.astype('float32')

# 기본 수익률 계산
raw_returns_val = ((total_pymnt_val + recoveries_val - collection_recovery_fee_val - loan_amnts_val) / loan_amnts_val).astype('float32')

# 승인 + default 케이스 마스크
val_defaults = y_val.values
val_selected = y_pred_proba_val < 1.0  # threshold보다 낮은 모든 케이스 = 전부 승인 가정
mask_val_default = (val_selected) & (val_defaults == 1)

# Default IRR 계산
risk_free_rate = 0 
ann_returns_val = np.full_like(raw_returns_val, risk_free_rate, dtype='float32')
default_indices = np.where((y_val.values == 1) & val_selected)[0]
for i in default_indices:
    cf0 = -loan_amnts_val[i]
    cf1 = total_pymnt_val[i] + recoveries_val[i] - collection_recovery_fee_val[i]
    irr_rate = nf.irr([cf0, cf1])
    ann_returns_val[i] = irr_rate if not np.isnan(irr_rate) else raw_returns_val[i]

portfolio_returns = ann_returns_val

thresholds = np.linspace(0.0, 0.5, 100)
sharpe_ratios = []

risk_free_rate = 0.03

for t in thresholds:
    selected = y_pred_proba_val < t

    selected_returns = np.zeros_like(selected, dtype='float32')

    # default 여부
    defaults_selected = y_val.values
    loan_amnts_selected = loan_amnts_val
    recoveries_selected = recoveries_val
    collection_fee_selected = collection_recovery_fee_val
    int_rates_selected = df.loc[X_val_selected.index, 'int_rate'].values.astype('float32')
    loan_term_selected = loan_term_val

    # (1) 승인 + default
    mask_1 = (selected) & (defaults_selected == 1)
    raw_return_1 = (
        (recoveries_selected[mask_1] - collection_fee_selected[mask_1] - loan_amnts_selected[mask_1])
        / loan_amnts_selected[mask_1]
    )
    with np.errstate(invalid='ignore'):
        ann_return_1 = (1 + raw_return_1) ** (1 / loan_term_selected[mask_1]) - 1
    selected_returns[mask_1] = ann_return_1

    # (2) 승인 + default 아님
    mask_2 = (selected) & (defaults_selected == 0)
    selected_returns[mask_2] = int_rates_selected[mask_2]

    # (3) 미승인
    selected_returns[~selected] = risk_free_rate

    if selected_returns.std() == 0:
        sharpe_ratios.append(np.nan)
    else:
        sharpe_ratios.append((selected_returns.mean() - risk_free_rate) / selected_returns.std())

best_threshold = thresholds[np.nanargmax(sharpe_ratios)]
print("📈 최적 Sharpe Ratio 기준 Threshold:", round(best_threshold, 3))
best_sharpe_ratio = np.nanmax(sharpe_ratios)
print("📊 Validation Sharpe Ratio (최대값):", round(best_sharpe_ratio, 4))

# Sharpe Ratio vs Threshold 그래프 시각화
plt.figure(figsize=(8, 5))
plt.plot(thresholds, sharpe_ratios, marker='o')
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold = {round(best_threshold, 3)}')
plt.title('Sharpe Ratio vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. 최종 test
X_test_selected = X_test[top_15_cols]
y_pred_proba_test = model.predict_proba(X_test_selected)[:, 1]
test_selected = y_pred_proba_test < best_threshold
recoveries_test = df.loc[X_test_selected.index, 'recoveries'].fillna(0).values.astype('float32')
loan_amnts_test = df.loc[X_test_selected.index, 'loan_amnt'].values.astype('float32')
test_defaults = y_test.values
# Test set additional variables for IRR calculation
collection_recovery_fee_test = df.loc[X_test_selected.index, 'collection_recovery_fee'].values.astype('float32')
loan_term_test = df.loc[X_test_selected.index, 'term'].values.astype('float32')
int_rates_test = df.loc[X_test_selected.index, 'int_rate'].values.astype('float32')
# Risk-free rate for return calculations
risk_free_rate = 0

# 테스트셋 수익률 계산 방식 수정
# 1. 대출이 나갔는데 default → 회수율 기반 수익률
# 2. 대출이 나갔고 default 안 났음 → 이자 수익률 (int_rate)
# 3. 대출이 안 나갔음 → risk-free 수익률

# installment (월별 상환액) 벡터 추가
installment_test = df.loc[X_test_selected.index, 'installment'].values.astype('float32')

# IRR 계산: default, 정상 상환, 미승인 케이스 반영
test_returns = np.full_like(int_rates_test, risk_free_rate, dtype='float32')

# (1) Default IRR: 월별 현금흐름 + 최종 회수
mask_def = (test_selected) & (test_defaults == 1)
for idx in np.where(mask_def)[0]:
    n_periods = int(loan_term_test[idx] * 12)
    cf = [-loan_amnts_test[idx]] + [installment_test[idx]] * n_periods
    cf[-1] += (recoveries_test[idx] - collection_recovery_fee_test[idx])
    irr = nf.irr(cf)
    test_returns[idx] = irr if not np.isnan(irr) else (recoveries_test[idx] - collection_recovery_fee_test[idx] - loan_amnts_test[idx]) / loan_amnts_test[idx]

# (2) 정상 상환 IRR
mask_norm = (test_selected) & (test_defaults == 0)
for idx in np.where(mask_norm)[0]:
    n_periods = int(loan_term_test[idx] * 12)
    cf = [-loan_amnts_test[idx]] + [installment_test[idx]] * n_periods
    irr = nf.irr(cf)
    test_returns[idx] = irr if not np.isnan(irr) else int_rates_test[idx]

# (3) 미승인 케이스 → risk-free 수익률
test_returns[~test_selected] = risk_free_rate
portfolio_returns = ann_returns_val

# Sharpe 계산 (excluding NaN values)
valid_returns = test_returns[~np.isnan(test_returns)]

if valid_returns.std() > 0:
    excess_return = valid_returns.mean() - risk_free_rate
    test_sharpe = excess_return / valid_returns.std()
else:
    test_sharpe = np.nan

# 📊 테스트셋 Sharpe Ratio 출력
print("📊 테스트셋 Sharpe Ratio:", round(test_sharpe, 4))

# 🔍 SHAP 설명은 중요도 상위 30개 변수 기준으로 수행됩니다.
explainer = shap.TreeExplainer(model.booster_)
shap_values = explainer.shap_values(X_test_selected)
shap.summary_plot(shap_values, X_test_selected)
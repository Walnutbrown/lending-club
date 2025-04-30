# Feature Engineering
## 📚 Feature Engineering for Linear Models

- **데이터 소스**: `01_data/01-2_interim/lendingclub_clean.csv`
- **주요 전처리 내용**:
  - **결측치 처리**:
    - 결측률이 **5% 이상**인 변수에 대해 `__nanflag`(결측 여부 플래그) 생성
    - 결측값은 **변수별 평균(mean)** 으로 대체
    - 숫자형 변수는 추가로 `__nan_x_value` 생성 (평균값 × nanflag)
    - 결측률이 5% 이하인 경우에는 단순히 평균값으로만 대체
  - **문자열 처리**:
    - 결측값을 `"missing"` 문자열로 대체
    - **One-Hot Encoding**으로 더미변수화 진행
- **출력 파일**:
  - `01_data/01-3_processed/lendingclub_features_for_linear.csv`

---

## 📚 Feature Engineering for Tree-Based Models

### LightGBM 전처리

- **결측치 처리**:
  - 문자열 변수: 결측값은 `"missing"` 문자열로 대체
  - 숫자형 변수: 결측값은 그대로 두고, 모델이 내부적으로 처리하도록 설계
  - 추가 파생변수(`__nanflag`, `__nan_x_value`)는 생성하지 않음
- **퍼센트 데이터 처리**:
  - `%` 기호가 포함된 컬럼을 자동 탐색하여 `%` 제거 후 숫자형 변환
  - 퍼센트 의미를 갖는 컬럼들도 숫자형으로 강제 변환
- **날짜형 데이터 처리**:
  - `issue_d`, `last_pymnt_d`, `earliest_cr_line`, `sec_app_earliest_cr_line` 컬럼을 datetime 타입으로 변환
- **카테고리형 처리**:
  - One-Hot Encoding을 사용하지 않고, **원래 카테고리형 상태 그대로 유지**
  - LightGBM 모델 내부에서 카테고리형 변수로 인식

### Random Forest / XGBoost 전처리

- **결측치 처리**:
  - 문자열 변수: 결측값은 `"missing"` 문자열로 대체
  - 숫자형 변수: 결측값은 그대로 두고 모델이 내부적으로 처리
- **퍼센트 데이터 처리 및 날짜형 데이터 변환**:
  - LightGBM 전처리와 동일
- **카테고리형 처리**:
  - **Label Encoding**을 통해 카테고리형 변수를 수치형으로 변환하여 학습

- **출력 파일**:
  - `01_data/01-3_processed/lendingclub_features_for_tree.csv`

---

## 📊 EDA Insights

- **FICO Range**: 대출이 승인된 데이터만 존재하기 때문에, 신용 점수(FICO)는 일정 수준 이상으로 편향되어 있음. 극단적으로 낮은 신용 점수는 없음.
- **inq_last_6mths (6개월 내 신용 조회 수)**: 대부분 0~1회이며, 조회가 많은 경우 대출 승인이 어려운 경향을 보임.
- **delinq_2yrs, pub_rec**: 연체 이력 및 공공 기록이 거의 없음. 신용 이력에 문제가 없는 사람 위주.
- **annual_inc (연소득)**: 일부 상위 소득자를 제외하면 대부분이 일정 수준에 몰려 있음. 상위 소득자 비율은 낮음.
- **revol_util (리볼빙 사용률)**: 대체로 낮은 리볼빙 사용률을 보이며, 리볼빙 빚을 많이 쓰는 사람은 승인이 어려웠을 가능성.
- **open_acc, total_acc (계좌 수)**: 보통 10~30개 계좌 보유. 일정 수준 이상의 계좌 수가 신용 점수에 긍정적 영향을 미친다고 추측 가능.
- **revol_bal, total_rev_hi_lim, tot_cur_bal (잔액 관련 변수)**: 잔액 분포가 심하지만 대체로 일정 범위 내에 있음.
- **mo_sin_old_rev_tl_op (오래된 계좌 유지 기간)**: 평균 약 16년의 신용 히스토리를 보유. 오래된 계좌가 신용 평가에 긍정적 역할을 했을 것.

## 🛠️ EDA-Based Action Plan

| 관찰된 특성 | 적용할 전처리 | 이유 |
|:---|:---|:---|
| 연소득, 리볼빙 사용률 등 연속형 변수의 한쪽 꼬리 분포 | 로그 변환 (`np.log1p`) | 정규성 확보 및 이상치 영향 최소화 |
| FICO 점수대가 상향 편향 | 표준화 (`StandardScaler`) | 변수 간 스케일 차이를 줄여 선형모델 안정성 확보 |
| inq_last_6mths, delinq_2yrs, pub_rec 등 이산형 변수 | 그대로 유지 또는 필요시 더미 변환 | 이산형 변수는 선형모델에서 의미를 가질 수 있음 |
| 일부 변수 결측치 존재 | nanflag 생성 + 평균 대체 | 결측 여부 자체가 중요한 정보일 수 있음 |
| hardship 관련 플래그 변수들 | 희귀 이벤트로 별도 처리 검토 | 전체 데이터에 비해 비율이 너무 낮아 모델 왜곡 가능성 존재 |

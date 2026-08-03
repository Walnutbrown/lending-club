
# 📊 Sharpe Ratio 극대화 모델 구축 요약

---

## 🎯 프로젝트 목표

Lending Club 대출 데이터를 기반으로,  
**Sharpe Ratio(초과 수익률 대비 위험)**를 극대화하는 포트폴리오를 구성할 수 있도록  
default 예측 모델과 전략을 설계한다.

---

## 📐 핵심 개념 정리

### ✅ Sharpe Ratio 정의

\[
\text{Sharpe Ratio} = \frac{E[R] - r_f}{\sigma_R}
\]

- **E[R]**: 기대 수익률 (예: 대출 승인 시 발생 가능한 수익)
- **r_f**: 무위험 수익률 (예: 미국채 수익률, 보통 2~3%)
- **σ_R**: 수익률의 표준편차 (리스크)

→ 결국 "수익이 크고, 변동성은 적은" 포트폴리오가 Sharpe Ratio가 높음

---

## 🧱 모델 설계 개요

### ✅ 왜 default 확률을 예측하는가?

- `int_rate`는 내생변수 → 예측에 사용 불가
- 대신 **default 여부(0 or 1)를 종속변수로 설정**
- 예측한 부도 확률 `P(default)`를 기반으로 기대 수익률을 계산하여 전략 수립

---

## 💡 기대 수익률 계산 방식

```python
expected_return = (1 - P_default) * r - P_default * 1.0
```

- `r`: 해당 대출의 이자율 (관측값으로만 사용, 입력변수 아님)
- `P_default`: 모델이 예측한 부도 확률
- 부도 시 -1, 정상 상환 시 이자율만큼 수익 발생으로 가정

---

## 🧪 전체 실험 프로세스

---

### 1️⃣ 학습 단계 (Train - 60%)

- 목적: `default`를 예측하는 분류 모델 학습
- 입력: 신청 시점에 관측 가능한 정보 (소득, 고용연수, FICO 점수, 목적 등)
- 출력: 각 대출의 `P(default)` 예측값

→ 이 모델은 **미래의 대출 신청자가 부도 날 확률을 추정**하는 역할

---

### 2️⃣ 검증 단계 (Validation - 20%)

- 목적: 어떤 전략이 **Sharpe Ratio를 가장 높이는지 실험**
- 방법:
  1. 모델로 각 대출의 `P(default)` 예측
  2. 기대 수익률 계산:
     ```python
     expected_return = (1 - P_default) * int_rate - P_default * 1.0
     ```
  3. 다양한 전략 실험:
     - **Cutoff 방식**:
       ```python
       if P_default < θ:
           approve = True
       else:
           approve = False
       ```
     - **Sharpe 기준 방식**:
       ```python
       if expected_return > risk_free_rate:
           approve = True
       else:
           approve = False
       ```
  4. 승인된 대출들로 포트폴리오 구성 → 기대 수익률 평균과 표준편차 계산
  5. Sharpe Ratio 측정

- 🔍 다양한 cutoff θ에 대해 반복 → **Sharpe Ratio가 최대가 되는 θ\*** 선택

---

### 3️⃣ 테스트 단계 (Test - 20%)

- 목적: **검증에서 선택된 전략(θ\*)의 실제 성과 평가**
- 방법:
  1. 학습된 모델로 test set의 `P(default)` 예측
  2. 검증에서 선택한 전략(θ\*) 그대로 적용 → 승인 여부 결정
  3. 승인된 대출로 포트폴리오 구성
  4. 기대 수익률 및 변동성 계산
  5. **Sharpe Ratio 최종 평가**

→ 전략이 실제 unseen 데이터에서도 Sharpe Ratio가 높게 나오는지 확인

---

## ⚠️ 변수 선택 시 주의사항

- 예측 시점 기준의 정보만 사용할 것 (정보집합 조건)
- 아래 변수들은 **내생 변수이므로 예측에 사용 금지**:
  - `int_rate` (이자율)
  - `grade`, `sub_grade` (Lending Club이 사후 평가한 등급)
  - `loan_status`, `funded_amnt`, `issue_d` 등 사후 변수

- 단, `int_rate`는 **수익률 계산 시 관측값으로 사용 가능** (target 구성용)

---

## ✅ 최종 전략 기준 요약

전략의 핵심은 다음과 같다:

```python
# 기대 수익률이 무위험 수익률보다 높은 경우에만 대출 승인
if expected_return > risk_free_rate:
    approve = True
else:
    approve = False
```

또는

```python
# default 확률이 θ보다 낮으면 승인
if P_default < θ_star:
    approve = True
else:
    approve = False
```

→ 승인된 포트폴리오의 Sharpe Ratio가 미국채보다 높으면 전략 성공

---

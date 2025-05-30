import pandas as pd

def main():
    # 전처리된 feature 파일 불러오기
<<<<<<< HEAD
    df = pd.read_csv('data/processed/lendingclub_features_for_lightgbm.csv')
=======
    df = pd.read_csv('../../data/processed/lendingclub_features_for_lightgbm.csv')
>>>>>>> fdc21f29decd5b56c3acce4eecb3fe029be56124
    # df = pd.read_csv('../../01_data/01-3_processed/lendingclub_features_for_linear.csv')
    # df = pd.read_csv('../../01_data/01-3_processed/lendingclub_features_for_tree.csv')

    # 제외할 컬럼 리스트
    exclude_cols = [
        'cash_flow'
        'collection_recovery_fee'
        'default',
        'funded_amnt',
        'funded_amnt_inv',
        'grade',
        'id',
        'initial_list_status',
        'int_rate',
        'issue_d',
        'last_credit_pull_d',
        'last_fico_range_high',
        'last_fico_range_low',
        'last_pymnt_amnt',
        'last_pymnt_d',
        'loan_amnt',
        'loan_status',
        'member_id',
        'next_pymnt_d',
        'out_prncp',
        'out_prncp_inv',
        'policy_code',
        'pymnt_plan',
        'recoveries',
        'sub_grade',
        'term',
        'title',
        'total_pymnt',
        'total_pymnt_inv',
        'total_rec_int',
        'total_rec_late_fee',
        'total_rec_prncp',
        'url',
        'zip_code',
        'hardship_flag',
        'hardship_type',
        'hardship_reason',
        'hardship_status',
        'deferral_term',
        'hardship_amount',
        'hardship_start_date',
        'hardship_end_date',
        'payment_plan_start_date',
        'hardship_length',
        'hardship_dpd',
        'hardship_loan_status',
        'orig_projected_additional_accrued_interest',
        'hardship_payoff_balance_amount',
        'hardship_last_payment_amount',
        'disbursement_method',
        'debt_settlement_flag',
        'debt_settlement_flag_date',
        'settlement_status',
        'settlement_date',
        'settlement_amount',
        'settlement_percentage',
        'settlement_term'
    ]

    # feature 리스트 생성
    feature_list = [col for col in df.columns if col not in exclude_cols]

    # 저장
    features_df = pd.DataFrame({'feature': feature_list})
<<<<<<< HEAD
    features_df.to_csv('data/processed/features_final_list_lightgbm.csv', index=False)
=======
    features_df.to_csv('../../data/processed/features_final_list_lightgbm.csv', index=False)
>>>>>>> fdc21f29decd5b56c3acce4eecb3fe029be56124

    print(f"✅ features_final_list_lightgbm.csv 생성 완료! ({len(feature_list)}개 변수)")

if __name__ == "__main__":
    main()
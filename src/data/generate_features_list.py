"""Generate model-specific feature lists without editing source comments."""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_FILES = {
    "lightgbm": (
        "lendingclub_features_for_lightgbm.csv",
        "features_final_list_lightgbm.csv",
    ),
    "linear": (
        "lendingclub_features_for_linear.csv",
        "features_final_list_linear.csv",
    ),
    "tree": (
        "lendingclub_features_for_tree.csv",
        "features_final_list_rf_xg.csv",
    ),
    "rf_xg": (
        "lendingclub_features_for_tree.csv",
        "features_final_list_rf_xg.csv",
    ),
}

EXCLUDE_COLS = {
    "cash_flow", "collection_recovery_fee", "collections_12_mths_ex_med", "default",
    "funded_amnt", "funded_amnt_inv", "grade", "id", "initial_list_status",
    "installment", "int_rate", "issue_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "last_pymnt_amnt", "last_pymnt_d", "last_pymnt_num",
    "loan_amnt", "loan_status", "member_id", "next_pymnt_d", "out_prncp",
    "out_prncp_inv", "policy_code", "pymnt_plan", "recoveries", "sub_grade", "term",
    "title", "total_pymnt", "total_pymnt_inv", "total_rec_int", "total_rec_late_fee",
    "total_rec_prncp", "url", "zip_code", "hardship_flag", "hardship_type",
    "hardship_reason", "hardship_status", "deferral_term", "hardship_amount",
    "hardship_start_date", "hardship_end_date", "payment_plan_start_date",
    "hardship_length", "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
    "hardship_last_payment_amount", "disbursement_method", "debt_settlement_flag",
    "debt_settlement_flag_date", "settlement_status", "settlement_date",
    "settlement_amount", "settlement_percentage", "settlement_term",
}


def generate_feature_list(model: str = "lightgbm") -> Path:
    """Generate and save a feature list for one preprocessing family."""
    model = model.lower()
    if model not in MODEL_FILES:
        choices = ", ".join(sorted(MODEL_FILES))
        raise ValueError(f"Unknown model '{model}'. Choose one of: {choices}.")

    input_name, output_name = MODEL_FILES[model]
    input_path = PROJECT_DIR / "data" / "processed" / input_name
    output_path = PROJECT_DIR / "data" / "processed" / output_name
    if not input_path.exists():
        raise FileNotFoundError(
            f"입력 파일이 없습니다: {input_path}\n"
            f"먼저 해당 모델의 전처리 스크립트를 실행하세요."
        )

    df = pd.read_csv(input_path, low_memory=False)
    feature_list = [col for col in df.columns if col not in EXCLUDE_COLS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feature_list}).to_csv(output_path, index=False)
    print(f"✅ {model}: {output_path.name} 생성 완료 ({len(feature_list)}개 변수)")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a feature list for a selected preprocessing/model family."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FILES),
        default="lightgbm",
        help="Feature-table family to read (default: lightgbm).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate_feature_list(parse_args().model)

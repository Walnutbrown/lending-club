from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]


FEATURES = [
    "acc_now_delinq", "acc_open_past_24mths", "addr_state", "all_util",
    "annual_inc", "annual_inc_joint", "application_type", "avg_cur_bal",
    "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths",
    "collections_12_mths_ex_med", "delinq_2yrs", "delinq_amnt", "dti",
    "dti_joint", "earliest_cr_line_num", "emp_length", "fico_range_high",
    "fico_range_low", "home_ownership", "il_util", "inq_fi", "inq_last_12m",
    "inq_last_6mths", "max_bal_bc", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc", "mths_since_last_delinq",
    "mths_since_last_major_derog", "mths_since_last_record", "mths_since_rcnt_il",
    "mths_since_recent_bc", "mths_since_recent_bc_dlq", "mths_since_recent_inq",
    "mths_since_recent_revol_delinq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
    "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m",
    "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "open_acc",
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "open_rv_12m",
    "open_rv_24m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec",
    "pub_rec_bankruptcies", "purpose", "revol_bal", "revol_util", "tax_liens",
    "tot_coll_amt", "tot_cur_bal", "tot_hi_cred_lim", "total_acc",
    "total_bal_ex_mort", "total_bal_il", "total_bc_limit", "total_cu_tl",
    "total_il_high_credit_limit", "total_rev_hi_lim", "verification_status",
    "revol_bal_joint", "sec_app_fico_range_low", "sec_app_fico_range_high",
    "sec_app_earliest_cr_line_num", "sec_app_inq_last_6mths", "sec_app_mort_acc",
    "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il",
    "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
]


def main():
    input_path = PROJECT_DIR / "data" / "interim" / "lendingclub_clean.csv"
    df = pd.read_csv(input_path)
    df["term"] = df["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)

    object_cols = df.select_dtypes(include="object").columns.tolist()
    percent_cols = [
        col for col in object_cols
        if df[col].dropna().map(lambda value: "%" in str(value)).any()
    ]
    for col in percent_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", "", regex=False), errors="coerce")
    for col in ["percent_bc_gt_75", "all_util", "il_util"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["issue_d", "last_pymnt_d", "earliest_cr_line", "sec_app_earliest_cr_line"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["last_pymnt_num"] = (
        (df["last_pymnt_d"].dt.year - df["issue_d"].dt.year) * 12
        + (df["last_pymnt_d"].dt.month - df["issue_d"].dt.month)
    ).fillna(0).astype(int)
    for source, target in [
        ("earliest_cr_line", "earliest_cr_line_num"),
        ("sec_app_earliest_cr_line", "sec_app_earliest_cr_line_num"),
    ]:
        df[target] = (
            (df[source].dt.year - df["issue_d"].dt.year) * 12
            + (df[source].dt.month - df["issue_d"].dt.month)
        )
    df.drop(columns=["earliest_cr_line", "sec_app_earliest_cr_line"], inplace=True)

    for col in FEATURES:
        if col not in df.columns or not df[col].isna().any():
            continue
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1).drop(columns=[col])
        elif np.issubdtype(df[col].dtype, np.number):
            missing = df[col].isna().astype(int)
            df[f"{col}__nanflag"] = missing
            df[col] = df[col].fillna(0)
            df[f"{col}__nan_x_value"] = df[col] * missing

    output_path = PROJECT_DIR / "data" / "processed" / "lendingclub_features_for_linear.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ 파일 저장 완료: {output_path}")


if __name__ == "__main__":
    main()

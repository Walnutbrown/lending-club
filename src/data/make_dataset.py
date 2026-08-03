from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]


def main():
    """Create the cleaned Lending Club dataset from the raw export."""
    input_path = PROJECT_DIR / "data" / "raw" / "lendingclub.csv"
    df = pd.read_csv(input_path, low_memory=False)
    print(df.head(3))

    def process_emp_length(value):
        if pd.isna(value):
            return None
        value = str(value)
        if "< 1" in value:
            return 0.5
        if "10+" in value:
            return 10.0
        extracted = pd.to_numeric(
            pd.Series(value).str.extract(r"(\d+)")[0], errors="coerce"
        )
        return extracted.iloc[0]

    df["emp_length"] = df["emp_length"].apply(process_emp_length)

    # Keep only observations with an unambiguous repayment outcome.
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    status_mapping = {"Fully Paid": 0, "Charged Off": 1, "Default": 1}
    df["default"] = df["loan_status"].map(status_mapping)
    df = df.drop(columns=["loan_status"])
    df = df[df["default"].notna()]
    print(f"전처리 후 데이터 크기: {df.shape}")

    output_path = PROJECT_DIR / "data" / "interim" / "lendingclub_clean.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd


def create_cash_flow(row):
    """Create monthly investor cash flows for one loan observation."""
    try:
        loan_amnt = float(row["loan_amnt"])
        installment = float(row["installment"])
        term = int(float(row["term"]))
        default = int(float(row["default"]))
        last_pymnt_num = int(float(row["last_pymnt_num"]))
        recoveries = float(row["recoveries"])
        collection_recovery_fee = float(row["collection_recovery_fee"])
    except (KeyError, TypeError, ValueError):
        return np.nan

    values = [loan_amnt, installment, recoveries, collection_recovery_fee]
    if term <= 0 or not np.all(np.isfinite(values)):
        return np.nan

    cash_flow = [-loan_amnt]
    for month in range(1, term + 1):
        if default == 1:
            if month == last_pymnt_num + 1:
                cash_flow.append(recoveries - collection_recovery_fee)
            elif month <= last_pymnt_num:
                cash_flow.append(installment)
            else:
                cash_flow.append(0.0)
        else:
            cash_flow.append(installment)
    return cash_flow


cash_flow_cache = {}


def get_cash_flow(row):
    """Cached wrapper used by repeated threshold evaluations."""
    try:
        key = (
            float(row["loan_amnt"]),
            float(row["installment"]),
            int(float(row["term"])),
            int(float(row["default"])),
            int(float(row["last_pymnt_num"])),
            float(row["recoveries"]),
            float(row["collection_recovery_fee"]),
        )
    except (KeyError, TypeError, ValueError):
        return np.nan

    if key not in cash_flow_cache:
        cash_flow_cache[key] = create_cash_flow(row)
    return cash_flow_cache[key]

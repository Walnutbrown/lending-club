import numpy as np
import numpy_financial as npf
import pandas as pd
import fredapi as Fred
from utils.make_cashflow import create_cash_flow

# IRR 계산 함수
def calculate_irr(cash_flow):
    irr_monthly = npf.irr(cash_flow)
    if irr_monthly is not None and not np.isnan(irr_monthly):
        irr_annual = (1 + irr_monthly) ** 12 - 1
    else:
        irr_annual = np.nan
    return irr_annual

# Sharpe Ratio 계산 함수
def calculate_sharpe(returns, risk_free):
    returns     = np.asarray(returns, dtype=float) # 벡터
    risk_free   = np.asarray(risk_free, dtype=float) # 벡터
    mask = ~np.isnan(returns) & ~np.isnan(risk_free)
    excess = returns[mask] - risk_free[mask] # nan이 없을 때
    if excess.size == 0 or np.nanstd(returns, ddof = 1) == 0:
        return np.nan
    return np.nanmean(excess) / np.nanstd(excess, ddof = 1) 

def calculate_sharpe_from_df(df):
    df['cash_flow'] = df.apply(create_cash_flow, axis=1)
    df['irr'] = df['cash_flow'].apply(calculate_irr)
    sharpe_ratio = calculate_sharpe(df['irr'], df['risk_free_rate'])
    return sharpe_ratio
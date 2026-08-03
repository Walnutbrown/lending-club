import numpy as np
import numpy_financial as npf

from utils.make_cashflow import create_cash_flow


def calculate_irr(cash_flow):
    """Convert monthly IRR to an annualized IRR."""
    try:
        irr_monthly = npf.irr(cash_flow)
    except (TypeError, ValueError):
        return np.nan
    if irr_monthly is not None and np.isfinite(irr_monthly):
        return (1 + irr_monthly) ** 12 - 1
    return np.nan


def calculate_sharpe(returns, risk_free):
    """Calculate the mean excess return divided by excess-return volatility."""
    returns = np.asarray(returns, dtype=float)
    risk_free = np.asarray(risk_free, dtype=float)
    mask = np.isfinite(returns) & np.isfinite(risk_free)
    excess = returns[mask] - risk_free[mask]
    if excess.size < 2:
        return np.nan
    excess_std = np.nanstd(excess, ddof=1)
    if not np.isfinite(excess_std) or excess_std == 0:
        return np.nan
    return np.nanmean(excess) / excess_std


def calculate_sharpe_from_df(df):
    df = df.copy()
    df["cash_flow"] = df.apply(create_cash_flow, axis=1)
    df["irr"] = df["cash_flow"].apply(calculate_irr)
    return calculate_sharpe(df["irr"], df["risk_free_rate"])


irr_cache = {}


def get_irr(cash_flow):
    """Cached IRR calculation for repeated investment-policy evaluations."""
    if not isinstance(cash_flow, list) or not cash_flow:
        return np.nan
    key = tuple(round(float(value), 6) for value in cash_flow)
    if key not in irr_cache:
        irr_cache[key] = calculate_irr(cash_flow)
    return irr_cache[key]


def precompute_cashflow_and_irr(df):
    """Precompute cash flows and IRR once before threshold search."""
    df = df.copy()
    df["cash_flow"] = df.apply(create_cash_flow, axis=1)
    df["irr"] = df["cash_flow"].apply(get_irr)
    df["irr"] = df["irr"].fillna(df["risk_free_rate"])
    return df


def compute_sharpe_for_threshold(df, threshold):
    """Evaluate the risk-adjusted return of loans below a score threshold."""
    selected = df.loc[df["pred_prob"] <= threshold]
    if selected.empty:
        return np.nan
    return calculate_sharpe(selected["irr"].to_numpy(), selected["risk_free_rate"].to_numpy())

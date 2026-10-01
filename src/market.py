"""Market side: clean hourly bars, basket returns, abnormal returns, diagnostics.

Basket construction follows Aloosh, Choi & Ouzan (meme stock indices): an
equally weighted portfolio from Yahoo Finance prices, log returns x 100.
Cap weights (shares x price before the window) are a robustness check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.unitroot import VarianceRatio
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.config import (BASKET, BETA_END, BETA_START, ET, RAW, SUB_BASKETS, WINDOW_END,
                        WINDOW_START)

BAR_STARTS = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]


def load_daily() -> pd.DataFrame:
    d = pd.read_parquet(RAW / "market" / "daily.parquet")
    close = d["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    # ^VIX prints on exchange holidays (Memorial Day, Labor Day), and the multi-ticker
    # download unions those dates in as all-NaN stock rows. Left in, they make the
    # NEXT day's log return NaN, which on Labor Day wiped out Sept 8, our event day.
    close = close[close["SPY"].notna()]
    return close


def load_hourly() -> dict[str, pd.DataFrame]:
    """Regular-session hourly bars only (PROGRESS issue 1: ^VIX extended hours
    had polluted the index). Returns dict of close / volume / high / low."""
    h = pd.read_parquet(RAW / "market" / "hourly.parquet")
    idx = pd.DatetimeIndex(h.index).tz_convert(ET)
    h.index = idx
    keep = idx.strftime("%H:%M").isin(BAR_STARTS) & h["Close"]["SPY"].notna().to_numpy()
    vix_raw = h["Close"]["^VIX"].dropna()   # VIX prints on the hour, the stock bars on :30
    h = h[keep]
    per_day = pd.Series(1, index=h.index).groupby(h.index.date).sum()
    assert (per_day == 7).all(), f"expected 7 bars per day, got {per_day[per_day != 7]}"
    out = {k: h[k].copy() for k in ("Close", "Volume", "High", "Low")}
    # as-of join: last VIX print at or before each bar start (was NaN on every bar before)
    out["Close"]["^VIX"] = vix_raw.reindex(vix_raw.index.union(h.index)).ffill().reindex(h.index)
    return out


def log_ret(px: pd.DataFrame) -> pd.DataFrame:
    return 100 * np.log(px / px.shift(1))


def cap_weights(daily_close: pd.DataFrame) -> pd.Series:
    sh = pd.read_csv(RAW / "market" / "shares.csv").set_index("ticker")["shares"]
    px = daily_close.loc[: pd.Timestamp(WINDOW_START.date()) - pd.Timedelta(days=1), BASKET].iloc[-1]
    mc = sh.reindex(BASKET) * px
    return mc / mc.sum()


def basket_returns(ret: pd.DataFrame, weights: pd.Series | None = None,
                   members: list[str] = BASKET) -> pd.Series:
    r = ret[members]
    if weights is None:
        return r.mean(axis=1, skipna=True)
    w = weights.reindex(members)
    # renormalise over names that traded in the bar; min_count stops an all-NaN
    # row (first bar) from summing to a fake 0.0 return
    wr = (r * w).sum(axis=1, min_count=1)
    return wr / r.notna().mul(w).sum(axis=1).replace(0, np.nan)


def market_model_betas(daily_ret: pd.DataFrame, bench: str = "SPY") -> pd.Series:
    """OLS betas on daily returns over the pre-period estimation window."""
    est = daily_ret.loc[BETA_START:BETA_END]
    betas = {}
    cols = BASKET + ["EW", "CW"] + [f"sub_{k}" for k in SUB_BASKETS]
    for c in cols:
        if c not in est:
            continue
        xy = est[[c, bench]].dropna()
        y, x = xy[c], sm.add_constant(xy[bench])
        betas[c] = sm.OLS(y, x).fit().params[bench]
    return pd.Series(betas)


def build_returns() -> dict[str, pd.DataFrame]:
    """Daily and hourly return panels with EW/CW baskets, sub-baskets and abnormal returns."""
    dc = load_daily()
    w = cap_weights(dc)
    out = {}
    dret = log_ret(dc)
    dret["EW"] = basket_returns(dret)
    dret["CW"] = basket_returns(dret, w)
    for k, mem in SUB_BASKETS.items():
        dret[f"sub_{k}"] = basket_returns(dret, members=mem)
    betas = market_model_betas(dret)
    for c, b in betas.items():
        dret[f"AR_{c}"] = dret[c] - b * dret["SPY"]
    dret["SPREAD_EW"] = dret["EW"] - dret["SPY"]
    out["daily"], out["betas"], out["weights"] = dret, betas, w

    hb = load_hourly()
    hret = log_ret(hb["Close"])
    hret["EW"] = basket_returns(hret)
    hret["CW"] = basket_returns(hret, w)
    for k, mem in SUB_BASKETS.items():
        hret[f"sub_{k}"] = basket_returns(hret, members=mem)
    for c, b in betas.items():
        hret[f"AR_{c}"] = hret[c] - b * hret["SPY"]
    hret["SPREAD_EW"] = hret["EW"] - hret["SPY"]
    hret["VIX"] = hb["Close"]["^VIX"]
    # Amihud illiquidity per bar for the EW basket: mean_i |r_i| / dollar volume_i (in $bn)
    dv = (hb["Close"][BASKET] * hb["Volume"][BASKET]) / 1e9
    hret["AMIHUD_EW"] = (hret[BASKET].abs() / dv).replace([np.inf], np.nan).mean(axis=1)
    out["hourly"] = hret.iloc[1:]
    out["hourly_close"] = hb["Close"]
    return out


def in_window(df: pd.DataFrame, start=WINDOW_START, end=WINDOW_END) -> pd.DataFrame:
    idx = df.index
    if idx.tz is None:
        return df.loc[str(start.date()): str(end.date())]
    return df[(idx >= start) & (idx < end)]


# ------------------------------------------------------------------ diagnostics
def descriptives(r: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Aloosh et al. Table 2 style."""
    rows = []
    for c in cols:
        x = r[c].dropna()
        rows.append({"series": c, "N": len(x), "Mean": x.mean(), "SD": x.std(), "Max": x.max(),
                     "Min": x.min(), "Kurt": x.kurt(), "Skew": x.skew()})
    return pd.DataFrame(rows).set_index("series")


def efficiency_tests(x: pd.Series, lags: int = 7) -> dict:
    """Weak-form checks as in Aloosh et al.: Ljung-Box, variance ratio, runs."""
    x = x.dropna()
    lb = acorr_ljungbox(x, lags=[lags], return_df=True)["lb_pvalue"].iloc[0]
    vr = VarianceRatio(x.to_numpy(), lags=2, robust=True).pvalue
    runs_p = runstest_1samp(x.to_numpy(), cutoff="median", correction=True)[1]
    return {"N": len(x), "LjungBox_p": lb, "VR2_p": vr, "Runs_p": runs_p}


def realized_vol(x: pd.Series, window: int = 14) -> pd.Series:
    """Rolling realized volatility, sqrt(sum r^2), 14-hour window as in Aloosh et al."""
    return np.sqrt((x ** 2).rolling(window).sum())

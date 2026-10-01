"""Download daily + hourly prices for the AI basket and benchmarks (yfinance).

Daily from BETA_START (for market-model betas) and 60-minute bars for the
baseline week + main window. Also grabs shares outstanding for cap weights.
"""
import _bootstrap  # noqa: F401
import warnings

import pandas as pd
import yfinance as yf

from src.config import BASKET, BENCHMARKS, BETA_START, RAW

warnings.filterwarnings("ignore")
OUT = RAW / "market"
OUT.mkdir(parents=True, exist_ok=True)
tickers = BASKET + BENCHMARKS

daily = yf.download(tickers, start=BETA_START, end="2026-09-30", interval="1d",
                    auto_adjust=True, progress=False, group_by="column")
daily.to_parquet(OUT / "daily.parquet")
print("daily", daily.shape, daily.index.min().date(), daily.index.max().date())

hourly = yf.download(tickers, start="2026-09-01", end="2026-09-30", interval="60m",
                     auto_adjust=True, progress=False, group_by="column", prepost=False)
hourly.to_parquet(OUT / "hourly.parquet")
print("hourly", hourly.shape, hourly.index.min(), hourly.index.max())

rows = []
for t in BASKET:
    try:
        fi = yf.Ticker(t).fast_info
        rows.append({"ticker": t, "shares": fi.get("shares"), "market_cap_now": fi.get("market_cap")})
    except Exception as e:  # noqa: BLE001
        rows.append({"ticker": t, "shares": None, "market_cap_now": None, "err": str(e)})
pd.DataFrame(rows).to_csv(OUT / "shares.csv", index=False)
print(pd.DataFrame(rows))

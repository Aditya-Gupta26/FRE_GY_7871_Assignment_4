"""Turn scored posts into sentiment time series aligned to trading bars.

Timing rule (no look-ahead)
- Hourly bars start at 09:30, 10:30, ... 15:30 ET and close at 10:30, ... 15:30, 16:00.
  Bar b's return runs from the close of bar b-1 to the close of bar b.
  The first bar of a day carries the overnight gap.
- A post belongs to bar b if close(b-1) <= post time < close(b). So sentiment
  S_b is contemporaneous with r_b, and a Granger lag S_{b-k} (k >= 1) only
  uses posts written before r_b's interval starts.
- Daily: the interval is [16:00 on the previous trading day, 16:00 today), so
  weekend and holiday posts roll into the next session (Monday).

Measures per interval (from the class papers)
- n: volume (Souza et al.'s V)
- net: mean(P(pos) - P(neg))
- neg_share / pos_share: share of posts labelled NEGATIVE / POSITIVE
- s_rel: (G - B) / (G + B) with label counts, Souza et al.'s S_R
- p_pos: mean P(pos), Smailovic et al.'s positive-sentiment probability
- net_w: engagement-weighted net, weights log(1 + likes + reposts)
- vader: mean VADER compound
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ET

BAR_CLOSES = ["10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"]


def bar_calendar(hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per bar: start (index of the price frame) and close time."""
    starts = pd.DatetimeIndex(hourly_index).tz_convert(ET)
    closes = []
    for s in starts:
        hh = s.strftime("%H:%M")
        c = "16:00" if hh == "15:30" else f"{int(hh[:2]) + 1:02d}:30"
        closes.append(s.normalize() + pd.Timedelta(hours=int(c[:2]), minutes=int(c[3:])))
    cal = pd.DataFrame({"bar_start": starts, "bar_close": pd.DatetimeIndex(closes)})
    return cal.sort_values("bar_start").reset_index(drop=True)


def assign_bars(ts: pd.Series, cal: pd.DataFrame) -> pd.Series:
    """Map each timestamp to the bar_start whose information interval contains it.
    NaT when the post is before the first interval or after the last close."""
    ts = pd.to_datetime(ts, utc=True).dt.tz_convert(ET)
    closes = cal["bar_close"].to_numpy()
    pos = np.searchsorted(closes, ts.to_numpy(), side="right")  # first close > ts
    ok = pos < len(cal)
    first_open = cal["bar_close"].iloc[0] - pd.Timedelta(hours=18)  # overnight before 1st bar
    ok &= (ts >= first_open).to_numpy()
    out = pd.Series(pd.NaT, index=ts.index, dtype=f"datetime64[ns, {ET.key}]")
    out[ok] = cal["bar_start"].to_numpy()[pos[ok]]
    return out


def daily_session(ts: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Map timestamps to the trading day whose [16:00 prev, 16:00 today) interval contains them."""
    ts = pd.to_datetime(ts, utc=True).dt.tz_convert(ET)
    closes = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(ET) + pd.Timedelta(hours=16)
                               for d in trading_days])
    pos = np.searchsorted(closes.to_numpy(), ts.to_numpy(), side="right")
    ok = pos < len(closes)
    out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    out[ok] = pd.DatetimeIndex(trading_days).to_numpy()[pos[ok]]
    return out


def aggregate(df: pd.DataFrame, key: str, eng_col: str | None = "engagement") -> pd.DataFrame:
    """Per-interval sentiment measures. Needs positiveScore/negativeScore/sentiment."""
    d = df.dropna(subset=[key]).copy()
    d["is_neg"] = (d["sentiment"] == "NEGATIVE").astype(float)
    d["is_pos"] = (d["sentiment"] == "POSITIVE").astype(float)
    d["net"] = d["positiveScore"] - d["negativeScore"]
    if eng_col and eng_col in d:
        d["w"] = np.log1p(d[eng_col].fillna(0).clip(lower=0)) + 1.0
    else:
        d["w"] = 1.0
    g = d.groupby(key)
    out = pd.DataFrame({
        "n": g.size(),
        "net": g["net"].mean(),
        "neg_share": g["is_neg"].mean(),
        "pos_share": g["is_pos"].mean(),
        "p_pos": g["positiveScore"].mean(),
        "p_neg": g["negativeScore"].mean(),
        "vader": g["vader_compound"].mean() if "vader_compound" in d else np.nan,
    })
    G, B = g["is_pos"].sum(), g["is_neg"].sum()
    out["s_rel"] = (G - B) / (G + B).replace(0, np.nan)
    out["net_w"] = g.apply(lambda x: np.average(x["net"], weights=x["w"]), include_groups=False)
    return out


def calendar_daily(df: pd.DataFrame, ts_col: str = "ts_utc") -> pd.DataFrame:
    """Plain ET calendar-day aggregation, for trend plots (weekends included)."""
    d = df.copy()
    d["day"] = pd.to_datetime(d[ts_col], utc=True).dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    return aggregate(d, "day")

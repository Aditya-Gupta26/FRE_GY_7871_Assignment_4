"""The whole analysis in one place, so scripts/07_run_analysis.py and
analysis.ipynb run exactly the same code and every number in REPORT.md comes
from one path.

Every step returns plain DataFrames/dicts and also writes them to
outputs/tables/ (CSV) and outputs/results.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import bias, granger, market, plots, trend
from src.config import (BASELINE_START, ET, FOMC_DAYS, OUTPUTS, PROCESSED, SUB_BASKETS, TABLES,
                        WINDOW_END, WINDOW_START)
from src.index import aggregate, assign_bars, bar_calendar, calendar_daily, daily_session

TABLES.mkdir(parents=True, exist_ok=True)
SPLIT = str(WINDOW_START.date())  # baseline week before, event window from


def _save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TABLES / f"{name}.csv")
    return df


# ------------------------------------------------------------------ loading
def load_sources() -> dict[str, pd.DataFrame]:
    src = {}
    for name in ["twitter", "reddit", "news", "arxiv"]:
        p = PROCESSED / f"{name}_scored.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d["ts_utc"] = pd.to_datetime(d["ts_utc"], utc=True)
            d["is_neg"] = (d["sentiment"] == "NEGATIVE").astype(float)
            d["is_pos"] = (d["sentiment"] == "POSITIVE").astype(float)
            src[name] = d
    return src


def streams(src: dict) -> dict[str, pd.DataFrame]:
    """The text streams we build indices for."""
    out = {}
    if "twitter" in src:
        tw = src["twitter"]
        for fam in ["GEN", "RISK", "FIN"]:
            out[f"tw_{fam}"] = tw[tw["family"] == fam]
    if "reddit" in src:
        rd = src["reddit"]
        out["reddit"] = rd
        # E1: AI-community talk vs investor talk
        out["reddit_ai"] = rd[rd["sub_type"] == "ai"]
        out["reddit_fin"] = rd[rd["sub_type"] == "finance"]
    if "news" in src:
        nw = src["news"]
        out["news_ai"] = nw[nw["is_ai"]]
    return out


# ------------------------------------------------------------------ indices
def build_indices(st: dict, R: dict) -> dict:
    cal = bar_calendar(R["hourly_close"].index)
    days = pd.DatetimeIndex(sorted(set(cal["bar_start"].dt.tz_localize(None).dt.normalize())))
    hourly, sess, cald = {}, {}, {}
    for name, d in st.items():
        d = d.copy()
        d["bar"] = assign_bars(d["ts_utc"], cal)
        d["session"] = daily_session(d["ts_utc"], days)
        hourly[name] = aggregate(d, "bar").reindex(cal["bar_start"])
        sess[name] = aggregate(d, "session").reindex(days)
        cald[name] = calendar_daily(d)
    return {"cal": cal, "days": days, "hourly": hourly, "session": sess, "calendar": cald}


def twitter_volume(freq: str = "D") -> pd.Series:
    """Estimated RISK tweet volume from sub-window fill rates (see src/twitter.py)."""
    from src.twitter import load_window_meta
    rows = []
    for m in load_window_meta():
        if m["family"] != "RISK":
            continue
        s, u = pd.Timestamp(m["since_utc"]), pd.Timestamp(m["until_utc"])
        win = (u - s).total_seconds()
        if m["n"] < 20 and not m["has_next_page"]:
            est = m["n"]
        else:
            ts = pd.to_datetime(pd.Series(m["created"]), format="%a %b %d %H:%M:%S %z %Y", utc=True)
            span = max((ts.max() - ts.min()).total_seconds(), 30.0)
            est = m["n"] * win / span
        rows.append({"t": s.tz_convert(ET), "est": est})
    v = pd.DataFrame(rows).set_index("t")["est"].sort_index()
    return v.resample(freq).sum() if freq else v


# ------------------------------------------------------------------ Q1 trend
def trend_tables(idx: dict, measure: str = "net") -> pd.DataFrame:
    rows = []
    for name, d in idx["calendar"].items():
        y = d[measure]
        y.index = pd.to_datetime(y.index)
        win = y[(y.index >= SPLIT) & (y.index < str(WINDOW_END.date()) + " 23:59")]
        rows.append({"stream": name, **trend.ols_trend(win), **trend.mann_kendall(win),
                     **{f"prepost_{k}": v for k, v in trend.pre_post(y, SPLIT).items()}})
    # one file per measure (they used to overwrite each other)
    return _save(pd.DataFrame(rows).set_index("stream"), f"q1_trend_tests_{measure}")


def event_windows(idx: dict, measure: str = "net", pre: int = 2, post: int = 2) -> pd.DataFrame:
    """Mean sentiment in the `pre` days before vs `post` days from each shock (calendar days)."""
    from src.config import EVENTS
    rows = []
    for e in EVENTS:
        t0 = pd.Timestamp(e["ts"]).tz_localize(None).normalize()
        for name, d in idx["calendar"].items():
            y = d[measure]
            y.index = pd.to_datetime(y.index)
            b = y[(y.index >= t0 - pd.Timedelta(days=pre)) & (y.index < t0)].mean()
            a = y[(y.index >= t0) & (y.index < t0 + pd.Timedelta(days=post))].mean()
            rows.append({"event": e["label"], "stream": name, "before": b, "after": a,
                         "change": a - b})
    return _save(pd.DataFrame(rows), "q1_event_windows")


# ------------------------------------------------------------------ Q2 Granger
def _exog_hourly(cal: pd.DataFrame) -> pd.DataFrame:
    s = cal.set_index("bar_start")
    ex = pd.DataFrame(index=s.index)
    ex["fomc"] = s.index.strftime("%Y-%m-%d").isin(FOMC_DAYS).astype(float)
    ex["overnight"] = (s.index.strftime("%H:%M") == "09:30").astype(float)
    return ex


def granger_grid(idx: dict, R: dict, ret_col: str = "AR_EW", measure: str = "net",
                 lags=(1, 2, 3), perm: bool = True, streams_=None, diff: bool = True,
                 freq: str = "hourly", targets: str = "all") -> pd.DataFrame:
    """Sentiment <-> returns, both directions, per stream. Hourly or session-daily."""
    if freq == "hourly":
        ret = market.in_window(R["hourly"])[ret_col]
        ex = _exog_hourly(idx["cal"]).reindex(ret.index)
        src = idx["hourly"]
    else:
        ret = market.in_window(R["daily"])[ret_col]
        ex = None
        src = {k: v.set_axis(pd.DatetimeIndex(v.index)) for k, v in idx["session"].items()}
    rows = []
    for name, d in src.items():
        if streams_ and name not in streams_:
            continue
        s = d[measure].reindex(ret.index)
        s = s.interpolate(limit_direction="both")          # rare empty bars
        if measure == "n":
            # abnormal attention: log volume minus its bar-of-day mean. Raw counts jump
            # mechanically at every open (the overnight bar holds ~17 h of posts)
            s = np.log1p(s)
            if freq == "hourly":
                tod = s.index.strftime("%H:%M")
            else:
                tod = s.index.dayofweek == 0            # Monday sessions hold the weekend
            s = s - s.groupby(tod).transform("mean")
        x = s.diff() if diff else s
        x = x.rename(name)
        frame = pd.concat([ret.rename("ret"), x], axis=1).dropna()
        e = None if ex is None else ex.loc[frame.index]
        keep = None
        if targets == "intraday":   # drop the 09:30 bar (overnight gap) as a TARGET only
            keep = frame.index[frame.index.strftime("%H:%M") != "09:30"]
            e = e[["fomc"]] if e is not None else None
        tab = granger.both_directions(frame["ret"], frame[name], lags=lags, exog=e, perm=perm,
                                      label=name, keep=keep)
        tab["freq"], tab["ret_col"], tab["measure"], tab["diff"] = freq, ret_col, measure, diff
        tab["targets"] = targets
        rows.append(tab)
    out = pd.concat(rows, ignore_index=True)
    out["p_fdr"] = granger.fdr(out["p"])
    return out


def var_results(idx: dict, R: dict, name: str, ret_col: str = "AR_EW", measure: str = "net") -> dict:
    ret = market.in_window(R["hourly"])[ret_col]
    s = idx["hourly"][name][measure].reindex(ret.index).interpolate(limit_direction="both")
    df = pd.concat([ret.rename("ret"), s.diff().rename("dsent")], axis=1).dropna()
    ex = _exog_hourly(idx["cal"]).reindex(df.index)[["fomc", "overnight"]]
    # standardise the shock so the IRF is "per 1 sd change in sentiment"
    df["dsent"] = (df["dsent"] - df["dsent"].mean()) / df["dsent"].std()
    return granger.var_analysis(df, "dsent", "ret", maxlags=7, exog=ex, ic="bic", fixed_p=3)


def leave_one_day_out(idx: dict, R: dict, name: str, lag: int = 3, ret_col: str = "AR_EW",
                      measure: str = "net") -> pd.DataFrame:
    """Re-run the Granger F-test dropping one trading day at a time (all 16)."""
    ret = market.in_window(R["hourly"])[ret_col]
    s = idx["hourly"][name][measure].reindex(ret.index).interpolate(limit_direction="both").diff()
    ex = _exog_hourly(idx["cal"]).reindex(ret.index)
    f = pd.concat([ret.rename("ret"), s.rename("x")], axis=1).dropna()
    rows = [{"dropped_day": "none", **granger.granger_f(f["ret"], f["x"], lag, ex.loc[f.index])}]
    for d in sorted(set(f.index.date)):
        m = f.index.date != d
        rows.append({"dropped_day": str(d), **granger.granger_f(f["ret"][m], f["x"][m], lag, ex.loc[f.index[m]])})
    return pd.DataFrame(rows)


def per_ticker(idx: dict, R: dict, name: str, lags=(1, 2, 3), measure: str = "net") -> pd.DataFrame:
    from src.config import BASKET
    cols = [f"AR_{t}" for t in BASKET] + [f"AR_sub_{k}" for k in SUB_BASKETS] + ["AR_EW", "AR_CW"]
    hr = market.in_window(R["hourly"])
    s = idx["hourly"][name][measure].reindex(hr.index).interpolate(limit_direction="both").diff()
    ex = _exog_hourly(idx["cal"]).reindex(hr.index)
    out = {}
    for c in cols:
        f = pd.concat([hr[c].rename("ret"), s.rename("x")], axis=1).dropna()
        out[c.replace("AR_", "")] = {f"lag {p}": granger.granger_f(f["ret"], f["x"], p, ex.loc[f.index])["p"]
                                     for p in lags}
    return pd.DataFrame(out).T


# ------------------------------------------------------------------ Q3 bias
def with_tool(d: pd.DataFrame, tool: str) -> pd.DataFrame:
    """Recompute the label columns from ONE tool, so every group is judged by the same
    classifier (Sacerdote et al. use one classifier for all outlet groups)."""
    d = d.copy()
    P = d[[f"{tool}_neg", f"{tool}_neu", f"{tool}_pos"]].to_numpy()
    lab = P.argmax(1)
    d["is_neg"], d["is_pos"] = (lab == 0).astype(float), (lab == 2).astype(float)
    d["negativeScore"], d["positiveScore"] = d[f"{tool}_neg"], d[f"{tool}_pos"]
    # Sacerdote et al.'s classifier is binary (every article is pos or neg), so their
    # "91% negative" is comparable to P(neg) > P(pos), not to a 3-class share
    d["p_neg_bin"] = d[f"{tool}_neg"] / (d[f"{tool}_neg"] + d[f"{tool}_pos"])
    d["is_neg_bin"] = (d["p_neg_bin"] > 0.5).astype(float)
    return d


def bias_frame(src: dict, tool: str = "rob", sample_social: int = 4000, seed: int = 7871) -> pd.DataFrame:
    """Stack news headlines (AI stories), arXiv abstracts, and samples of tweets/Reddit,
    all labelled by the same `tool`."""
    parts = []
    if "news" in src:
        n = src["news"]
        keep = n["is_ai"] & (n["from_gen"] if "from_gen" in n else True)
        parts.append(n[keep].assign(kind="news"))
    if "arxiv" in src:
        parts.append(src["arxiv"].assign(kind="science"))
    for k, g in [("twitter", "TWITTER"), ("reddit", "REDDIT")]:
        if k in src:
            d = src[k]
            parts.append(d.sample(min(sample_social, len(d)), random_state=seed).assign(group=g, kind="social"))
    cols = ["ts_utc", "group", "kind", "is_neg", "is_pos", "is_neg_bin", "p_neg_bin", "negativeScore",
            "positiveScore", "hl_neg", "hl_pos", "lm_neg", "n_tokens", "clean"]
    parts = [with_tool(p, tool) for p in parts]
    b = pd.concat([p[[c for c in cols if c in p]] for p in parts], ignore_index=True)
    t = b["ts_utc"]
    return b[(t >= WINDOW_START) & (t < WINDOW_END)]


def dump_results(results: dict, path=OUTPUTS / "results.json"):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        raise TypeError(type(o))
    def keys(o):
        if isinstance(o, dict):
            return {str(k.date()) if isinstance(k, pd.Timestamp) and k == k.normalize() else str(k)
                    if not isinstance(k, (str, int, float, bool)) else k: keys(v) for k, v in o.items()}
        if isinstance(o, list):
            return [keys(v) for v in o]
        return o
    path.write_text(json.dumps(keys(results), indent=1, default=conv))

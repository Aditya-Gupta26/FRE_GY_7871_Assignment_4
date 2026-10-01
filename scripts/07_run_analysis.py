"""Run the full analysis: every table in outputs/tables, every figure in
outputs/figures, and a results.json that REPORT.md numbers are copied from.
Needs data/processed/*_scored.parquet from script 05.
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from src import bias, granger, market, pipeline, plots, trend
from src.config import BASKET, SUB_BASKETS, TABLES, WINDOW_END, WINDOW_START

import sys
# one classifier for every group in the bias tests; picked from the validation F1
BIAS_TOOL = sys.argv[1] if len(sys.argv) > 1 else "rob"
res = {"bias_tool": BIAS_TOOL}
src = pipeline.load_sources()
print("sources:", {k: len(v) for k, v in src.items()})
R = market.build_returns()
st = pipeline.streams(src)
idx = pipeline.build_indices(st, R)
names = list(st)
main = "tw_GEN" if "tw_GEN" in names else names[0]
# headline streams for the main Granger grids (sub-streams would double count in the FDR)
MAIN = [k for k in ["tw_GEN", "tw_RISK", "tw_FIN", "reddit", "news_ai"] if k in names]
E1_STREAMS = [k for k in ["tw_RISK", "tw_FIN", "reddit_ai", "reddit_fin"] if k in names]

# ---------------------------------------------------------------- corpus counts
counts = []
for k, d in st.items():
    t = d["ts_utc"]
    counts.append({"stream": k, "baseline_week": int(((t >= "2026-09-01") & (t < WINDOW_START)).sum()),
                   "window": int(((t >= WINDOW_START) & (t < WINDOW_END)).sum())})
for k in ["news", "arxiv"]:
    if k in src:
        t = src[k]["ts_utc"]
        counts.append({"stream": k + "_all", "baseline_week": int((t < WINDOW_START).sum()),
                       "window": int(((t >= WINDOW_START) & (t < WINDOW_END)).sum())})
res["counts"] = pipeline._save(pd.DataFrame(counts).set_index("stream"), "corpus_counts").to_dict("index")

# ---------------------------------------------------------------- market diagnostics
hw = market.in_window(R["hourly"])
dw = market.in_window(R["daily"])
desc = market.descriptives(hw, ["EW", "CW", "SPY", "AR_EW", "sub_compute", "sub_platforms"])
pipeline._save(desc, "market_descriptives_hourly")
eff = pd.DataFrame({c: market.efficiency_tests(hw[c]) for c in ["EW", "AR_EW", "SPY", "sub_compute"]}).T
pipeline._save(eff, "market_efficiency_hourly")
res["efficiency"] = eff.to_dict("index")
res["daily_returns"] = dw[["EW", "CW", "SPY", "AR_EW", "sub_compute", "sub_platforms",
                           "sub_anthropic_exposed"]].round(3).to_dict("index")
res["betas"] = R["betas"].round(3).to_dict()

# ---------------------------------------------------------------- Q1 trend
tt = pipeline.trend_tables(idx, "net")
tn = pipeline.trend_tables(idx, "neg_share").add_suffix("_negshare")
res["trend_net"] = tt.to_dict("index")
res["trend_negshare"] = tn.to_dict("index")
ev = pipeline.event_windows(idx, "net")
res["event_windows"] = ev.to_dict("records")
try:
    vol_daily = pipeline.twitter_volume("D")
    vol_daily.index = vol_daily.index.tz_localize(None)
    vol_label = "AI-risk tweets\nper day (est.)"
except Exception as e:  # noqa: BLE001
    print("no twitter volume:", e)
    vol_daily = idx["calendar"][main]["n"]
    vol_label = f"{main} items\nper day"
spike = {k: trend.spike_test(v) for k, v in idx["hourly"].items()}
res["spike_test"] = spike
pipeline._save(pd.DataFrame(spike).T, "q1_spike_test")

# keyness: most negative 3 days vs the rest, main stream
d = st[main].copy()
d["day"] = d["ts_utc"].dt.tz_convert("America/New_York").dt.date
dm = idx["calendar"][main]["net"]
worst = [x.date() for x in pd.to_datetime(dm.loc[str(WINDOW_START.date()):].nsmallest(3).index)]
kw = trend.keyness(d.loc[d["day"].isin(worst), "clean"].tolist(), d.loc[~d["day"].isin(worst), "clean"].tolist())
res["keyness_worst_days"] = {"days": [str(x) for x in worst], "top": kw.head(12).to_dict("records")}
pipeline._save(kw, "q1_keyness_worst_days")
# keyness for every main stream too (the report quotes the main one, notebook shows all)
kw_all = {}
for k in [x for x in ["tw_GEN", "tw_RISK", "reddit", "news_ai"] if x in names]:
    dk = st[k].copy()
    dk["day"] = dk["ts_utc"].dt.tz_convert("America/New_York").dt.date
    sk = idx["calendar"][k]["net"]
    wk = [x.date() for x in pd.to_datetime(sk.loc[str(WINDOW_START.date()):].nsmallest(3).index)]
    tab = trend.keyness(dk.loc[dk["day"].isin(wk), "clean"].tolist(), dk.loc[~dk["day"].isin(wk), "clean"].tolist())
    kw_all[k] = {"days": [str(x) for x in wk], "top": tab.head(10)["word"].tolist()}
res["keyness_by_stream"] = kw_all
# rank of Sept 9 (day after the Hubinger post) among window days, per stream
res["sept9_rank"] = {k: int(pd.Series(idx["calendar"][k]["net"]).set_axis(pd.to_datetime(idx["calendar"][k]["net"].index))
                         .loc[str(WINDOW_START.date()):].rank().loc["2026-09-09"]) for k in names}

# leave-one-day-out check of the main Reddit result (lag 3, all bars), saved so it is auditable
from src import granger as _g
_ret = market.in_window(R["hourly"])["AR_EW"]
_ex = pipeline._exog_hourly(idx["cal"]).reindex(_ret.index)
_x = idx["hourly"]["reddit"]["net"].reindex(_ret.index).interpolate(limit_direction="both").diff()
_f = pd.concat([_ret.rename("ret"), _x.rename("x")], axis=1).dropna()
_lodo = []
for _day in sorted(set(_f.index.date)):
    _m = _f.index.date != _day
    _r = _g.granger_f(_f["ret"][_m], _f["x"][_m], 3, _ex.loc[_f.index[_m]])
    _lodo.append({"dropped_day": str(_day), "p": _r["p"], "sum_coef": _r["sum_coef"]})
_lodo = pd.DataFrame(_lodo)
pipeline._save(_lodo.set_index("dropped_day"), "q2_reddit_lag3_leave_one_day_out")
res["reddit_lodo_max_p"] = float(_lodo["p"].max())

# figure 1
labels = {"tw_GEN": "Twitter general AI", "tw_RISK": "Twitter AI-risk", "tw_FIN": "Twitter $cashtags",
          "reddit": "Reddit", "news_ai": "News headlines", "reddit_ai": "Reddit AI subs",
          "reddit_fin": "Reddit finance subs"}
show = [k for k in ["tw_GEN", "tw_RISK", "reddit", "news_ai"] if k in idx["calendar"]]
daily_series = {labels[k]: idx["calendar"][k]["net"] for k in show}
cum = {"AI basket (EW)": R["daily"].loc["2026-09-01":, "EW"].fillna(0).cumsum(),
       "S&P 500 (SPY)": R["daily"].loc["2026-09-01":, "SPY"].fillna(0).cumsum()}
fig1_stream = "tw_GEN" if "tw_GEN" in names else "reddit"
_d = st[fig1_stream].copy()
_d["day"] = _d["ts_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
_d = _d[_d["day"] >= "2026-09-01"]
pos_ct = _d[_d["sentiment"] == "POSITIVE"].groupby("day").size()
neg_ct = _d[_d["sentiment"] == "NEGATIVE"].groupby("day").size()
pos_ct = pos_ct.reindex(sorted(set(pos_ct.index) | set(neg_ct.index)), fill_value=0)
plots.fig_smailovic(pos_ct, neg_ct.reindex(pos_ct.index, fill_value=0), daily_series, cum,
                    labels.get(fig1_stream, fig1_stream))
plots.fig_trend(daily_series, cum, vol_daily.loc["2026-09-01":], vol_label=vol_label,
                fname="figE_sentiment_by_source.png")

# ---------------------------------------------------------------- Q2 Granger
g_h = pipeline.granger_grid(idx, R, "AR_EW", "net", lags=(1, 2, 3), perm=True, streams_=MAIN)
g_d = pipeline.granger_grid(idx, R, "AR_EW", "net", lags=(1, 2), perm=True, freq="daily", streams_=MAIN)
pipeline._save(g_h, "q2_granger_hourly")
g_hi = pipeline.granger_grid(idx, R, "AR_EW", "net", lags=(1, 2, 3), perm=True, targets="intraday",
                            streams_=MAIN)
pipeline._save(g_hi, "q2_granger_hourly_intraday_targets")
res["granger_hourly_intraday"] = g_hi.round(4).to_dict("records")
pipeline._save(g_d, "q2_granger_daily")
res["granger_hourly"] = g_h.round(4).to_dict("records")
res["granger_daily"] = g_d.round(4).to_dict("records")
# robustness grid (no permutation, it is slow)
rob = []
for rc in ["AR_EW", "EW", "AR_CW", "SPREAD_EW"]:
    for meas in ["net", "neg_share", "p_pos", "s_rel", "vader", "n"]:
        for dif in [True, False]:
            try:
                g = pipeline.granger_grid(idx, R, rc, meas, lags=(1, 2, 3), perm=False, diff=dif,
                                          streams_=MAIN)
                rob.append(g)
            except Exception as e:  # noqa: BLE001
                print("rob fail", rc, meas, dif, e)
# same grid with the overnight (09:30) bar removed as a target
for rc in ["AR_EW", "EW"]:
    for meas in ["net", "neg_share", "p_pos", "s_rel", "vader", "n"]:
        try:
            rob.append(pipeline.granger_grid(idx, R, rc, meas, lags=(1, 2, 3), perm=False,
                                             diff=True, targets="intraday", streams_=MAIN))
        except Exception as e:  # noqa: BLE001
            print("rob intraday fail", rc, meas, e)
rob = pd.concat(rob, ignore_index=True)
rob["p_fdr_all"] = granger.fdr(rob["p"])
pipeline._save(rob, "q2_granger_robustness")
sig = rob[rob["p"] < 0.05]
res["robustness_summary"] = {"n_tests": len(rob), "n_p_below_05": int(len(sig)),
                             "n_fdr_below_10": int((rob["p_fdr_all"] < 0.10).sum()),
                             "expected_false_pos_05": 0.05 * len(rob),
                             "significant": sig[["test", "direction", "lag", "ret_col", "measure", "diff",
                                                 "targets", "p", "sum_coef"]].round(4).to_dict("records")}
by = rob.assign(sig=rob["p"] < 0.05, fdr=rob["p_fdr_all"] < 0.10).groupby(["targets", "test", "direction"])
res["robustness_by_group"] = by.agg(n=("p", "size"), n_sig05=("sig", "sum"), n_fdr10=("fdr", "sum"),
                                    mean_sign=("sum_coef", lambda x: float(np.sign(x).mean()))).reset_index().to_dict("records")
# VAR + IRF for the main streams
irfs = {}
for k in MAIN:
    v = pipeline.var_results(idx, R, k)
    res.setdefault("var", {})[k] = {kk: vv for kk, vv in v.items()}
    if k in ["tw_GEN", "tw_RISK", "reddit"] or len(irfs) < 2:
        irfs[labels.get(k, k)] = v
heat_stream = "reddit" if "reddit" in names else main
loo = pipeline.leave_one_day_out(idx, R, "reddit", lag=3) if "reddit" in names else None
if loo is not None:
    pipeline._save(loo, "q2_leave_one_day_out_reddit_lag3")
    res["leave_one_day_out"] = {"max_p": float(loo["p"].iloc[1:].max()), "full_p": float(loo["p"].iloc[0]),
                                "min_sum_coef": float(loo["sum_coef"].iloc[1:].min())}
heat = pipeline.per_ticker(idx, R, heat_stream)
pipeline._save(heat, "q2_per_ticker_pvalues")
res["per_ticker"] = {"stream": heat_stream, "table": heat.round(4).to_dict("index")}
# Figure C heatmap uses general-AI Twitter (the Twitter stream with Granger links); the Reddit
# per-name table above stays saved, since the report's "chip names" claim rests on it
heat_rows = [t for t in BASKET] + ["sub_compute", "sub_platforms", "sub_ai_native", "sub_anthropic_exposed", "EW"]
if "tw_GEN" in names:
    heat_tw = pipeline.per_ticker(idx, R, "tw_GEN")
    pipeline._save(heat_tw, "q2_per_ticker_pvalues_twitter")
    res["per_ticker_twitter"] = {"stream": "tw_GEN", "table": heat_tw.round(4).to_dict("index")}
    plots.fig_granger(dict(list(irfs.items())[:3]), heat_tw.loc[heat_rows], fname="figC_irf_heatmap.png",
                      heat_stream="Twitter general AI")
else:
    plots.fig_granger(dict(list(irfs.items())[:3]), heat.loc[heat_rows], fname="figC_irf_heatmap.png",
                      heat_stream=heat_stream)
GRAPH = [k for k in ["tw_GEN", "tw_RISK", "tw_FIN", "reddit_ai", "reddit_fin", "news_ai"] if k in names]
gg_all = pipeline.granger_grid(idx, R, "AR_EW", "net", lags=(1, 2, 3), perm=False, streams_=GRAPH)
gg_intra = pipeline.granger_grid(idx, R, "AR_EW", "net", lags=(1, 2, 3), perm=False, streams_=GRAPH,
                                 targets="intraday")
pipeline._save(pd.concat([gg_all, gg_intra]), "q2_granger_graph_links")
short = {"tw_GEN": "Twitter\ngeneral", "tw_RISK": "Twitter\nAI-risk", "tw_FIN": "Twitter\n$cashtags",
         "reddit_ai": "Reddit\nAI subs", "reddit_fin": "Reddit\nfinance", "news_ai": "News\nheadlines"}
plots.fig_souza_graph({"(a) Hourly, all bars": gg_all, "(b) Hourly, intraday targets only": gg_intra}, short)

# ---------------------------------------------------------------- Q3 bias
B = pipeline.bias_frame(src, tool=BIAS_TOOL)
_nw = pipeline.with_tool(src["news"], BIAS_TOOL) if "news" in src else None
_pl = _nw[_nw["is_placebo"] & (_nw["ts_utc"] >= WINDOW_START) & (_nw["ts_utc"] < WINDOW_END)] if _nw is not None else None
# one pooled reference for the Hu-Liu z-score (AI + placebo + science + social), as in Sacerdote
HL_REF = pd.concat([B["hl_neg"], _pl["hl_neg"]]) if _pl is not None else B["hl_neg"]
gn = bias.group_negativity(B, ref=HL_REF)
pipeline._save(gn, "q3_group_negativity")
res["group_negativity"] = gn.round(4).to_dict("index")
plac = None
if "news" in src:
    n = pipeline.with_tool(src["news"], BIAS_TOOL)
    n = n[(n["ts_utc"] >= WINDOW_START) & (n["ts_utc"] < WINDOW_END)]
    pl = n[n["is_placebo"]]
    placebo = bias.group_negativity(pl, ref=HL_REF) if len(pl) > 50 else None
    if placebo is not None:
        placebo = placebo[placebo["N"] >= 50]   # tech press has ~1 placebo headline
        pipeline._save(placebo, "q3_placebo_negativity")
        res["placebo_negativity"] = placebo.round(4).to_dict("index")
        plac = placebo
    # same sampling query for every group (AI_GEN); risk-query-only items would inflate
    # doom shares for the general-outlet groups by construction
    ai = n[n["is_ai"] & n["from_gen"]]
    L1 = bias.lpm(ai, "is_neg_bin")
    L1b = bias.lpm(ai, "is_neg")
    L2 = bias.lpm(ai[ai["is_risk"]], "is_neg_bin") if ai["is_risk"].sum() > 100 else None
    L3 = bias.lpm(ai.assign(hlz=bias.standardise(ai["hl_neg"])), "hlz")
    res["lpm"] = {"all_ai": {"coef": L1.round(4).to_dict("index"), **L1.attrs},
                  "all_ai_3class": {"coef": L1b.round(4).to_dict("index"), **L1b.attrs},
                  "hl_dict": {"coef": L3.round(4).to_dict("index"), **L3.attrs}}
    if L2 is not None:
        res["lpm"]["risk_only"] = {"coef": L2.round(4).to_dict("index"), **L2.attrs}
    pipeline._save(L1, "q3_lpm_all_ai")
    DID = bias.lpm_did(ai, pl)
    pipeline._save(DID, "q3_lpm_did_ai_vs_placebo")
    res["lpm_did"] = {"coef": DID.round(4).to_dict("index"), **DID.attrs}
    other = "fin" if BIAS_TOOL == "rob" else "rob"
    n2 = pipeline.with_tool(src["news"], other)
    n2 = n2[(n2["ts_utc"] >= WINDOW_START) & (n2["ts_utc"] < WINDOW_END)]
    DID2 = bias.lpm_did(n2[n2["is_ai"] & n2["from_gen"]], n2[n2["is_placebo"]])
    res["lpm_did_other_tool"] = {"tool": other, "coef": DID2.round(4).to_dict("index"), **DID2.attrs}
    # tone vs objective benchmark (daily news negativity vs AI basket return)
    # daily share of negative AI headlines (same bias tool), on trading-session days
    from src.index import daily_session
    ai_s = ai.assign(session=daily_session(ai["ts_utc"], pd.DatetimeIndex(dw.index)))
    dn = ai_s.groupby("session")["is_neg_bin"].mean()
    res["tone_vs_market"] = bias.tone_vs_market(dn, dw["EW"])
    res["tone_vs_market_ar"] = bias.tone_vs_market(dn, dw["AR_EW"])
    fc = bias.frame_counts(ai)
    pipeline._save(fc, "q3_frame_counts")
    res["frame_counts"] = fc.round(4).to_dict("index")
    s1 = bias.s1_framing(n[n["ts_utc"] >= "2026-09-27"])
    res["s1_framing"] = s1.round(3).to_dict("index")
    res["base_rate"] = bias.base_rate_framing(n)
    # Sacerdote Fig 1 analog: daily P(negative) for US major vs international major
    from src.config import CONS_TRUST_PEW2019, OUTLET_SHORT
    ai_day = ai.assign(day=ai["ts_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize())
    dneg = {"US Mainstream": ai_day[ai_day["group"] == "US_MAJOR"].groupby("day")["is_neg_bin"].mean(),
            "International Mainstream": ai_day[ai_day["group"] == "INTL_MAJOR"].groupby("day")["is_neg_bin"].mean()}
    # abnormal (AI-specific) basket return: the series headline negativity actually tracks
    bask = R["daily"].loc[str(WINDOW_START.date()):, "AR_EW"].fillna(0).cumsum()
    # Sacerdote Fig 5 analog: outlet Hu-Liu z (pooled reference) vs conservative trust
    ai_z = ai.assign(z=bias.standardise(ai["hl_neg"], HL_REF))
    oz = ai_z[ai_z["outlet"].isin(CONS_TRUST_PEW2019)].groupby("outlet").agg(huliu_z=("z", "mean"), N=("z", "size"))
    oz["cons_trust"] = [CONS_TRUST_PEW2019[o] for o in oz.index]
    oz["name"] = [OUTLET_SHORT[o] for o in oz.index]
    pipeline._save(oz, "q3_outlet_negativity_vs_trust")
    res["outlet_vs_trust"] = oz.round(3).to_dict("index")
    res["outlet_vs_trust_corr"] = float(oz["huliu_z"].corr(oz["cons_trust"]))
    plots.fig_sacerdote1_5({k: v.loc[str(WINDOW_START.date()):] for k, v in dneg.items()}, bask, oz)
    ideol = ai[ai["outlet"].isin(["foxnews.com", "nypost.com", "cnn.com", "nytimes.com",
                                  "washingtonpost.com", "usatoday.com", "npr.org", "nbcnews.com"])]
    res["ideology"] = ideol.groupby("outlet")[["is_neg_bin", "is_neg"]].mean().assign(
        N=ideol.groupby("outlet").size()).round(3).to_dict("index")
plots.fig_bias(gn, plac, fname="figF_bias_two_measures.png")
plots.fig_sacerdote2(gn, plac)
if "twitter" in src:
    tw = pipeline.with_tool(src["twitter"], BIAS_TOOL)
    tw = tw[(tw["ts_utc"] >= WINDOW_START) & (tw["ts_utc"] < WINDOW_END)].copy()
    tw["log_fol"] = np.log1p(pd.to_numeric(tw["author_followers"], errors="coerce").fillna(0))
    res["demand_twitter"] = bias.demand_side(tw, "engagement", "log_fol + C(family)").round(4).to_dict("index")
if "reddit" in src:
    rd = pipeline.with_tool(src["reddit"], BIAS_TOOL)
    rd = rd[(rd["ts_utc"] >= WINDOW_START) & (rd["ts_utc"] < WINDOW_END)]
    res["demand_reddit"] = bias.demand_side(rd, "score", "C(subreddit) + C(kind)").round(4).to_dict("index")

# ---------------------------------------------------------------- extra questions
# E1: doom talk vs business talk (RISK family vs $cashtag family), already in g_h rows
# E2: volume vs tone, returns -> sentiment (in g_h / rob)
# E3: volatility and liquidity around shocks, compute vs platforms
rv = {k: market.realized_vol(R["hourly"][c]) for k, c in [("compute", "sub_compute"),
                                                           ("platforms", "sub_platforms"), ("SPY", "SPY")]}
plots.fig_vol(rv, show_from=pd.Timestamp(WINDOW_START), fname="figD_volatility_simple.png")
_shade = {"shock 1": ("2026-09-08 09:30", "2026-09-11 16:00"), "shock 2": ("2026-09-14 09:30", "2026-09-16 16:00"),
          "shock 3": ("2026-09-28 09:30", "2026-09-29 16:00")}
plots.fig_aloosh_vol({"compute": rv["compute"], "platforms": rv["platforms"], "S&P 500 (SPY)": rv["SPY"]}, _shade)
th_stream = "tw_GEN" if "tw_GEN" in names else "reddit"
_t = st[th_stream].copy()
_t["hour"] = _t["ts_utc"].dt.tz_convert("America/New_York").dt.floor("h")
_t = _t[_t["ts_utc"] >= WINDOW_START]
hh = _t.groupby("hour").agg(share=("is_risk", "mean"), pos=("vader_pos", "mean"), neg=("vader_neg", "mean"))
hh = hh.rolling(6, min_periods=1).mean()
plots.fig_thelwall(hh["share"], hh["pos"], hh["neg"], labels.get(th_stream, th_stream))
win = {"pre (Sept 1 to 7)": ("2026-09-01", "2026-09-08"), "shock 1 (Sept 8 to 11)": ("2026-09-08", "2026-09-12"),
       "shock 2 (Sept 14 to 16)": ("2026-09-14", "2026-09-17"), "calm (Sept 17 to 25)": ("2026-09-17", "2026-09-26"),
       "shock 3 (Sept 28 to 29)": ("2026-09-28", "2026-09-30")}
hall = R["hourly"]
rows = []
for lab, (a, b) in win.items():
    x = hall[(hall.index >= pd.Timestamp(a, tz="America/New_York")) & (hall.index < pd.Timestamp(b, tz="America/New_York"))]
    for c in ["sub_compute", "sub_platforms", "sub_anthropic_exposed", "EW", "SPY"]:
        rows.append({"window": lab, "series": c, "hourly_sd": x[c].std(), "realized": np.sqrt((x[c] ** 2).sum()),
                     "mean_ret": x[c].mean(), "amihud_ew": x["AMIHUD_EW"].mean(), "vix_mean": x["VIX"].mean(),
                     "n": len(x)})
E3 = pd.DataFrame(rows)
pipeline._save(E3, "e3_vol_windows")
res["e3"] = E3.round(4).to_dict("records")
g_sub = []
for c in ["AR_sub_compute", "AR_sub_platforms", "AR_sub_anthropic_exposed", "AR_sub_ai_native"]:
    g_sub.append(pipeline.granger_grid(idx, R, c, "net", lags=(1, 2, 3), perm=False, streams_=MAIN))
g_sub = pd.concat(g_sub, ignore_index=True)
pipeline._save(g_sub, "e3_granger_subbaskets")
res["granger_subbaskets"] = g_sub.round(4).to_dict("records")

# E1: doom talk vs market talk (Twitter RISK vs $cashtag families; Reddit AI subs vs finance subs)
e1 = []
for tg in ["all", "intraday"]:
    e1.append(pipeline.granger_grid(idx, R, "AR_EW", "net", lags=(1, 2, 3), perm=True, streams_=E1_STREAMS,
                                    targets=tg))
e1.append(pipeline.granger_grid(idx, R, "AR_EW", "net", lags=(1, 2), perm=True, streams_=E1_STREAMS,
                                freq="daily"))
e1 = pd.concat(e1, ignore_index=True)
pipeline._save(e1, "e1_doom_vs_market_talk")
res["e1"] = e1.round(4).to_dict("records")
# E2: attention (volume) instead of tone
e2 = rob[(rob["measure"] == "n") & (rob["ret_col"] == "AR_EW") & (rob["diff"])]
res["e2"] = e2[["test", "direction", "lag", "targets", "N", "p", "sum_coef"]].round(4).to_dict("records")

pipeline.dump_results(res)
print("done. tables:", len(list(TABLES.glob('*.csv'))))

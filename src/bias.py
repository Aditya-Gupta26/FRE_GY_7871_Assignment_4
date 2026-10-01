"""Media-bias tests in the style of Sacerdote, Sehgal & Cook (2020).

(a) negativity by outlet group, using two measures as in the paper:
    - a standardised Hu-Liu negative-word share (dictionary)
    - a classifier share-negative (supervised)
    plus the same outlets' non-AI coverage as a placebo baseline
(b) a linear probability model: neg_i ~ group dummies + day FE + log length,
    with international major as the omitted group (their Table 2)
(c) tone vs an objective benchmark: does news negativity track the AI basket?
    (their Figure 1, where tone ignored case counts)
(d) topic-selection counts: doom frames vs benefit frames (their Table 3)
(e) headline vs body negativity (sensationalism)
(f) per-outlet negativity, e.g. Fox vs CNN (their Figure 5)
(g) demand side: do negative posts get more engagement? (their NYT most-read test)
(h) base-rate framing: probability language vs "kill all humans" language
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

GROUP_ORDER = ["US_MAJOR", "US_GENERAL", "INTL_MAJOR", "INTL_GENERAL", "FIN_PRESS",
               "TECH_PRESS", "SCIENCE", "TWITTER", "REDDIT"]
DOOM_RE = re.compile(r"kill (?:us|all|every|humans|humanity)|extinct|existential|doom|wipe out|"
                     r"end (?:of )?humanity|destroy (?:us|human)|apocalyp|catastroph|terminator|"
                     r"gambling with our lives|end ?game", re.I)
BENEFIT_RE = re.compile(r"breakthrough|\bcure|boost|productiv|record|surge|soar|rall(?:y|ies)|"
                        r"revenue|profit|growth|opportunit|beat(?:s)? (?:estimates|expectations)|"
                        r"trillion|valuation|jump|gain|bull", re.I)
PROB_RE = re.compile(r"\d+\s?(?:%|percent)|chance|probab|odds|p\(doom\)|pdoom", re.I)
KILL_RE = re.compile(r"kill (?:us|all|every|humans|humanity)|wipe out|end humanity|destroy human", re.I)
S1_RE = re.compile(r"\bipo\b|s-1|prospectus|filing|listing|go(?:es|ing)? public", re.I)
S1_FIN_RE = re.compile(r"\$\s?\d|trillion|billion|loss|revenue|valuation|compute|raise", re.I)


def standardise(x: pd.Series, ref: pd.Series | None = None) -> pd.Series:
    """z-score against a reference sample (Sacerdote standardise once, in the broad sample)."""
    ref = x if ref is None else ref
    return (x - ref.mean()) / ref.std()


def group_negativity(df: pd.DataFrame, neg_label_col: str = "is_neg", dict_col: str = "hl_neg",
                     group_col: str = "group", ref: pd.Series | None = None) -> pd.DataFrame:
    """`ref`: the pooled sample to standardise against. Sacerdote et al. standardise once,
    in the broad sample (incl. non-COVID articles), so AI and placebo z-scores compare."""
    d = df.copy()
    d["dict_z"] = standardise(d[dict_col], ref)
    g = d.groupby(group_col)
    out = pd.DataFrame({"N": g.size(), "share_negative": g[neg_label_col].mean(),
                        "share_neg_binary": g["is_neg_bin"].mean() if "is_neg_bin" in d else np.nan,
                        "binary_se": g["is_neg_bin"].sem() if "is_neg_bin" in d else np.nan,
                        "p_neg_mean": g["negativeScore"].mean(), "huliu_z": g["dict_z"].mean(),
                        "huliu_se": g["dict_z"].sem(), "share_neg_se": g[neg_label_col].sem()})
    return out.reindex([x for x in GROUP_ORDER if x in out.index] +
                       [x for x in out.index if x not in GROUP_ORDER])


def lpm(df: pd.DataFrame, y: str = "is_neg", omit: str = "INTL_MAJOR") -> pd.DataFrame:
    d = df.copy()
    d["day"] = pd.to_datetime(d["ts_utc"]).dt.tz_convert("America/New_York").dt.date.astype(str)
    d["loglen"] = np.log1p(d["n_tokens"])
    f = f"{y} ~ C(group, Treatment('{omit}')) + C(day) + loglen"
    fit = smf.ols(f, data=d).fit(cov_type="HC1")
    rows = []
    for k, v in fit.params.items():
        if k.startswith("C(group"):
            name = k.split("[T.")[1].rstrip("]")
            rows.append({"group": name, "coef": v, "se": fit.bse[k], "p": fit.pvalues[k]})
    res = pd.DataFrame(rows).set_index("group")
    res.attrs.update({"N": int(fit.nobs), "R2": fit.rsquared,
                      "omitted_mean": d.loc[d["group"] == omit, y].mean()})
    return res


def tone_vs_market(daily_news: pd.Series, daily_ret: pd.Series) -> dict:
    d = pd.concat({"neg": daily_news, "ret": daily_ret}, axis=1).dropna()
    fit = smf.ols("neg ~ ret", data=d).fit(cov_type="HC1")
    up, down = d.loc[d["ret"] > 0, "neg"], d.loc[d["ret"] <= 0, "neg"]
    from scipy import stats
    t = stats.ttest_ind(up, down, equal_var=False)
    return {"N": len(d), "corr": d["neg"].corr(d["ret"]), "slope": fit.params["ret"],
            "slope_p": fit.pvalues["ret"], "neg_up_days": up.mean(), "neg_down_days": down.mean(),
            "updown_p": t.pvalue, "n_up": len(up), "n_down": len(down)}


def frame_counts(df: pd.DataFrame, text_col: str = "title", group_col: str = "group") -> pd.DataFrame:
    d = df.copy()
    d["doom"] = d[text_col].str.contains(DOOM_RE)
    d["benefit"] = d[text_col].str.contains(BENEFIT_RE)
    g = d.groupby(group_col)
    out = pd.DataFrame({"N": g.size(), "doom_n": g["doom"].sum(), "benefit_n": g["benefit"].sum()})
    out["doom_share"] = out["doom_n"] / out["N"]
    out["benefit_share"] = out["benefit_n"] / out["N"]
    out["doom_to_benefit"] = out["doom_n"] / out["benefit_n"].replace(0, np.nan)
    return out.reindex([x for x in GROUP_ORDER if x in out.index])


def s1_framing(df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
    """Anthropic IPO/S-1 headlines (Sept 28 to 29): lead with doom or with the numbers?"""
    d = df[df[text_col].str.contains("anthropic", case=False) & df[text_col].str.contains(S1_RE)].copy()
    d["risk_frame"] = d[text_col].str.contains(DOOM_RE) | d[text_col].str.contains(r"\brisk", case=False)
    d["fin_frame"] = d[text_col].str.contains(S1_FIN_RE)
    return d.groupby("group").agg(N=("risk_frame", "size"), risk_frame=("risk_frame", "mean"),
                                  fin_frame=("fin_frame", "mean"))


def base_rate_framing(df: pd.DataFrame, text_col: str = "title") -> dict:
    """Among headlines about the Coxon/Hubinger episode, how many give the probability?"""
    d = df[df[text_col].str.contains(r"hubinger|coxon|anthropic researcher|anthropic insider|"
                                     r"anthropic employee|anthropic alignment", case=False)]
    return {"N": len(d), "share_probability": d[text_col].str.contains(PROB_RE).mean(),
            "share_kill_language": d[text_col].str.contains(KILL_RE).mean(),
            "share_doom_any": d[text_col].str.contains(DOOM_RE).mean()}


def demand_side(df: pd.DataFrame, eng: str, controls: str) -> pd.DataFrame:
    d = df.copy()
    d["log_eng"] = np.log1p(d[eng].clip(lower=0))
    d["day"] = pd.to_datetime(d["ts_utc"]).dt.tz_convert("America/New_York").dt.date.astype(str)
    fit = smf.ols(f"log_eng ~ is_neg + is_pos + {controls} + C(day)", data=d).fit(cov_type="HC1")
    return pd.DataFrame({"coef": fit.params[["is_neg", "is_pos"]], "se": fit.bse[["is_neg", "is_pos"]],
                         "p": fit.pvalues[["is_neg", "is_pos"]]}).assign(N=int(fit.nobs))


def lpm_did(ai: pd.DataFrame, placebo: pd.DataFrame, y: str = "is_neg_bin",
            omit: str = "INTL_MAJOR", groups=("US_MAJOR", "US_GENERAL", "INTL_MAJOR",
                                                "INTL_GENERAL", "FIN_PRESS")) -> pd.DataFrame:
    """Is the US-major gap AI-specific? Pool AI and same-outlet placebo headlines:
    y ~ group * is_ai_story + day FE + log length. The group x AI interaction is the
    part of the gap that is not just the outlet being negative about everything."""
    d = pd.concat([ai.assign(ai_story=1.0), placebo.assign(ai_story=0.0)], ignore_index=True)
    d = d[d["group"].isin(groups)].copy()
    d["day"] = pd.to_datetime(d["ts_utc"]).dt.tz_convert("America/New_York").dt.date.astype(str)
    d["loglen"] = np.log1p(d["n_tokens"])
    f = f"{y} ~ C(group, Treatment('{omit}')) * ai_story + C(day) + loglen"
    fit = smf.ols(f, data=d).fit(cov_type="HC1")
    rows = []
    for k, v in fit.params.items():
        if k.startswith("C(group") or k == "ai_story":
            rows.append({"term": k.replace(f"C(group, Treatment('{omit}'))", "").replace("[T.", "").replace("]", ""),
                         "coef": v, "se": fit.bse[k], "p": fit.pvalues[k]})
    res = pd.DataFrame(rows).set_index("term")
    res.attrs.update({"N": int(fit.nobs), "R2": fit.rsquared})
    return res

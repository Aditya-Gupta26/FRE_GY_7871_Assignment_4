"""Trend tests for the sentiment series (assignment Q1).

- OLS on a time trend with Newey-West (HAC) standard errors
- Mann-Kendall monotonic trend test (non-parametric)
- baseline week (Sept 1 to 7) vs event window (Sept 8 to 29), Welch t-test
- event windows around each shock
- Thelwall et al. (2011) spike test: are high-volume hours more negative?
- keyness (chi-square) of words on the most negative days vs the rest,
  i.e. the co-word comparison from the Thelwall slides
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.api as sm
from scipy import stats

STOP = set("""a an the and or but if of to in on for with at by from is are was were be been being it its
this that these those as i you he she we they me my our your their them his her not no so do does did
just about into than then there here what which who whom will would can could should may might have has
had http user amp rt s t re ve ll d m im dont its it's i'm""".split())
WORD = re.compile(r"[a-z][a-z'\-]{2,}")


def ols_trend(y: pd.Series, maxlags: int = 2) -> dict:
    y = y.dropna()
    t = np.arange(len(y), dtype=float)
    fit = sm.OLS(y.to_numpy(), sm.add_constant(t)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"slope_per_day": float(fit.params[1]), "t": float(fit.tvalues[1]),
            "p": float(fit.pvalues[1]), "N": len(y)}


def mann_kendall(y: pd.Series) -> dict:
    r = mk.original_test(y.dropna().to_numpy())
    return {"mk_trend": r.trend, "mk_tau": float(r.Tau), "mk_p": float(r.p)}


def pre_post(y: pd.Series, split: str) -> dict:
    pre, post = y[y.index < split].dropna(), y[y.index >= split].dropna()
    t = stats.ttest_ind(post, pre, equal_var=False)
    return {"pre_mean": pre.mean(), "post_mean": post.mean(), "diff": post.mean() - pre.mean(),
            "t": float(t.statistic), "p": float(t.pvalue), "n_pre": len(pre), "n_post": len(post)}


def spike_test(hourly: pd.DataFrame, y: str = "neg_share", n: str = "n") -> dict:
    d = hourly[[y, n]].dropna()
    d = d[d[n] > 0]
    fit = sm.OLS(d[y], sm.add_constant(np.log(d[n]))).fit(cov_type="HC1")
    return {"coef_log_volume": float(fit.params.iloc[1]), "t": float(fit.tvalues.iloc[1]),
            "p": float(fit.pvalues.iloc[1]), "N": len(d)}


def tokens(text: str) -> list[str]:
    return [w for w in WORD.findall((text or "").lower()) if w not in STOP]


def keyness(texts_a: list[str], texts_b: list[str], top: int = 20, min_count: int = 10) -> pd.DataFrame:
    """Words over-represented in A vs B (chi-square with Yates, signed)."""
    ca, cb = Counter(), Counter()
    for t in texts_a:
        ca.update(set(tokens(t)))
    for t in texts_b:
        cb.update(set(tokens(t)))
    na, nb = len(texts_a), len(texts_b)
    rows = []
    for w in set(ca) | set(cb):
        a, b = ca[w], cb[w]
        if a + b < min_count:
            continue
        tab = np.array([[a, na - a], [b, nb - b]])
        chi2 = stats.chi2_contingency(tab, correction=True)[0]
        sign = 1 if a / na > b / nb else -1
        rows.append({"word": w, "docs_A": a, "docs_B": b, "rate_A": a / na, "rate_B": b / nb,
                     "chi2_signed": sign * chi2})
    # ties broken by the word itself, so the order does not depend on set iteration order
    out = pd.DataFrame(rows).sort_values(["chi2_signed", "word"], ascending=[False, True])
    return out.head(top)

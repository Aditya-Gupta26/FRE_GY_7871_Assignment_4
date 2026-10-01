"""Figures for the report (static PNGs for the PDF).

Style follows the dataviz reference palette: categorical slots in fixed order
(blue, orange, aqua, yellow), validated for CVD separation; one y-axis per
panel (no dual axes, so sentiment and returns get stacked panels); thin 1.6 pt
lines; recessive grid; legend plus direct end-of-line labels (the contrast
relief for aqua/yellow); ink colours for all text.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.config import EVENTS, FIGURES  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"
DIVERGING = LinearSegmentedColormap.from_list("bluered", ["#1c5cab", "#86b6ef", "#f0efec",
                                                          "#ef9a98", "#b8302f"])
SEQ_BLUE = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 7, "lines.linewidth": 1.6, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK,
})


def _events(ax, kinds=("risk", "business", "macro"), label=True, y=0.98):
    k = 0
    for e in EVENTS:
        if e["kind"] not in kinds:
            continue
        x = pd.Timestamp(e["ts"]).tz_localize(None)
        ls = {"risk": "-", "business": "--", "macro": ":"}[e["kind"]]
        ax.axvline(x, color=MUTED, lw=0.8, ls=ls, zorder=0)
        if label and e.get("short"):
            ax.text(x, 1.01 + 0.055 * (k % 2), e["short"], transform=ax.get_xaxis_transform(),
                    rotation=0, va="bottom", ha="center", fontsize=6, color=INK2)
            k += 1


def _spread_labels(ax, items: list[tuple], min_frac: float = 0.075):
    """items: (x_last, y_last, text, color). Place end labels with a minimum vertical gap."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_frac
    items = sorted(items, key=lambda t: t[1])
    ys = [it[1] for it in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (x, y, text, color), yl in zip(items, ys):
        ax.plot(x, y, "o", ms=3, color=color)
        ax.annotate(text, (x, y), xytext=(x, yl), textcoords="data", va="center", fontsize=7,
                    color=INK2, xycoords="data", annotation_clip=False,
                    bbox=None, ha="left")


def _direct_label(ax, s: pd.Series, text: str, color: str):
    s = s.dropna()
    if len(s):
        ax.annotate(text, (s.index[-1], s.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK2)
        ax.plot(s.index[-1], s.iloc[-1], "o", ms=3, color=color)


def fig_trend(daily: dict[str, pd.Series], cumret: dict[str, pd.Series], vol: pd.Series,
              fname: str = "fig1_sentiment_trend.png", ylab: str = "net sentiment (P(pos) - P(neg))",
              vol_label: str = "RISK tweets\n(est. per day)"):
    """Top: daily net sentiment per source (3-day MA solid, raw faint). Middle: volume.
    Bottom: cumulative log return of the AI basket vs SPY. Stacked, one axis each."""
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True,
                            gridspec_kw={"height_ratios": [2.2, 0.8, 1.4], "hspace": 0.12})
    ax = axs[0]
    for i, (name, s) in enumerate(daily.items()):
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        c = SERIES[i]
        ax.plot(s.index, s.values, color=c, lw=0.8, alpha=0.35)
        ma = s.rolling(3, min_periods=1, center=False).mean()
        ax.plot(ma.index, ma.values, color=c, label=name)
        _direct_label(ax, ma, name, c)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel(ylab)
    ax.set_title("AI sentiment by source, Sept 1 to 29 (3-day moving average, raw daily faint)",
                 pad=22)
    ax.legend(loc="lower left", ncol=len(daily))
    _events(ax)
    ax = axs[1]
    v = vol.copy()
    v.index = pd.to_datetime(v.index)
    ax.bar(v.index, v.values, width=0.7, color=SERIES[0], alpha=0.55, edgecolor=SURFACE, lw=1)
    ax.set_ylabel(vol_label)
    _events(ax, label=False)
    ax = axs[2]
    for i, (name, s) in enumerate(cumret.items()):
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        ax.plot(s.index, s.values, color=SERIES[i], label=name, drawstyle="default")
        _direct_label(ax, s, name, SERIES[i])
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("cumulative log return (%)")
    ax.legend(loc="upper left", ncol=len(cumret))
    _events(ax, label=False)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.align_ylabels(axs)
    fig.savefig(FIGURES / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return FIGURES / fname


def fig_granger(irfs: dict[str, dict], heat: pd.DataFrame, fname: str = "fig2_granger.png"):
    """Left: IRF of AI-basket abnormal return to a 1-sd sentiment shock (with 90% bands).
    Right: per-ticker Granger p-values (sent -> ret), diverging around p = 0.10."""
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    for i, (name, r) in enumerate(irfs.items()):
        h = np.arange(len(r["irf"]))
        c = SERIES[i]
        if r.get("irf_lo") is not None:
            ax.fill_between(h, r["irf_lo"], r["irf_hi"], color=c, alpha=0.15, lw=0)
        ax.plot(h, r["irf"], color=c, marker="o", ms=3, label=name)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("hours after the sentiment shock")
    ax.set_ylabel("abnormal return response (%)")
    ax.set_title("Impulse response, AI basket (EW)")
    ax.legend(loc="upper right")
    ax = fig.add_subplot(gs[1])
    H = heat.copy()
    H.index = [{"sub_compute": "Compute", "sub_platforms": "Platforms",
                "sub_anthropic_exposed": "Anthropic investors", "EW": "EW basket",
                "sub_ai_native": "AI-native", "CW": "CW basket"}.get(i, i) for i in H.index]
    im = ax.imshow(-np.log10(H.clip(lower=1e-4)), cmap=SEQ_BLUE, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(H.shape[1]), H.columns, rotation=0)
    ax.set_yticks(range(H.shape[0]), H.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(H.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                color="#ffffff" if -np.log10(max(v, 1e-4)) > 1.8 else INK)
    ax.set_title("Granger p-value by name (sent -> ret)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("-log10(p)", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(FIGURES / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return FIGURES / fname


def fig_bias(neg: pd.DataFrame, placebo: pd.DataFrame | None, fname: str = "fig3_bias.png"):
    """Sacerdote Fig 2 style, both of their measures: (a) classifier share with
    P(neg) > P(pos); (b) Hu-Liu negative-word share, z-scored in the pooled sample.
    AI coverage vs the same outlets' non-AI coverage (placebo). One axis per panel."""
    order = [g for g in neg.index][::-1]
    names = {"US_MAJOR": "US major", "US_GENERAL": "US general", "INTL_MAJOR": "Intl. major",
             "INTL_GENERAL": "Intl. general", "FIN_PRESS": "Financial press", "TECH_PRESS": "Tech press",
             "SCIENCE": "Science (arXiv)", "TWITTER": "Twitter", "REDDIT": "Reddit"}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(order))
    h = 0.38
    for ax, col, pcol, title, xl in [
            (axs[0], "share_neg_binary", "share_neg_binary", "(a) Classifier: P(neg) > P(pos)", "share of items"),
            (axs[1], "huliu_z", "huliu_z", "(b) Hu-Liu negative words (z-score)", "sd from pooled mean")]:
        v = neg.reindex(order)[col]
        ax.barh(y + h / 2, v, height=h, color=SERIES[0], edgecolor=SURFACE, lw=1, label="AI coverage")
        for yi, val in zip(y, v):
            ax.text(val, yi + h / 2, f" {val:.2f}" if val >= 0 else f"{val:.2f} ", va="center",
                    ha="left" if val >= 0 else "right", fontsize=6, color=INK2)
        if placebo is not None:
            pv = placebo.reindex(order)[pcol]
            ax.barh(y - h / 2, pv, height=h, color=SERIES[1], edgecolor=SURFACE, lw=1,
                    label="same outlets, non-AI")
            for yi, val in zip(y, pv):
                if pd.notna(val):
                    ax.text(val, yi - h / 2, f" {val:.2f}" if val >= 0 else f"{val:.2f} ", va="center",
                            ha="left" if val >= 0 else "right", fontsize=6, color=INK2)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, [names.get(g, g) for g in order])
    lo, hi = axs[1].get_xlim()
    axs[1].set_xlim(lo - 0.08, hi + 0.08)
    axs[0].set_xlim(0, max(1.0, axs[0].get_xlim()[1]))
    hnd, lab = axs[0].get_legend_handles_labels()
    fig.legend(hnd, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.13), fontsize=7)
    fig.suptitle("How negative is AI coverage? By outlet group, Sept 8 to 29", x=0.02, ha="left",
                 fontsize=9, fontweight="bold", y=1.0)
    fig.savefig(FIGURES / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return FIGURES / fname


def fig_vol(rv: dict[str, pd.Series], fname: str = "fig4_volatility.png", show_from=None):
    """Rolling 14-hour realized volatility, compute vs platforms vs SPY (Aloosh et al.).
    Computed on all bars, shown from `show_from`; x axis is bar order (trading hours only)."""
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    idx = list(rv.values())[0].index
    keep = idx >= show_from if show_from is not None else np.ones(len(idx), bool)
    idx = idx[keep]
    for i, (name, s) in enumerate(rv.items()):
        y = s[keep].values
        x = np.arange(len(y))
        ax.plot(x, y, color=SERIES[i], label=name)
        _direct_label(ax, pd.Series(y, index=x), name, SERIES[i])
    days = pd.Series(range(len(idx)), index=idx).groupby(idx.date).first()
    ax.set_xticks(days.values, [f"{pd.Timestamp(d).day}" for d in days.index])
    ax.set_xlabel("September, trading days (7 hourly bars each)")
    for e in EVENTS:
        if e["kind"] == "macro":
            continue
        pos = np.searchsorted(idx, pd.Timestamp(e["ts"]))
        if 0 < pos < len(idx) and e.get("short"):
            ax.axvline(pos, color=INK2, lw=0.8, ls="-" if e["kind"] == "risk" else "--", zorder=1)
            ax.text(pos, 1.01, e["short"], transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=6, color=INK2)
    ax.set_ylabel("realized vol, 14h (%)")
    ax.set_title("Rolling 14-hour realized volatility of hourly returns", pad=14)
    ax.legend(loc="upper right", ncol=3)
    fig.savefig(FIGURES / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return FIGURES / fname


# ============================================================================
# Paper-style recreations. Each one copies the layout of a figure in the class
# readings; the only deliberate change is that a second y-axis in the original
# becomes its own panel on the same time axis (one axis per panel).
# ============================================================================
POS, NEG = "#2a78d6", "#e34948"          # diverging pair: positive blue, negative red
DARK_BLUE, LIGHT_BLUE = "#1c5cab", "#9ec5f4"


def _callout(ax, x, y, text, dx=0, dy=24):
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6,
                ha="center", va="bottom", color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="#eef4fc", ec="#86b6ef", lw=0.6),
                arrowprops=dict(arrowstyle="-", color="#86b6ef", lw=0.6))


def fig_smailovic(pos: pd.Series, neg: pd.Series, net: dict[str, pd.Series],
                  cumret: dict[str, pd.Series], stream_name: str,
                  fname: str = "fig1_sentiment_trend.png"):
    """Smailovic et al. Figure 1: number of positive posts per day above zero and negative
    posts below zero, with event call-out boxes. Their price line sits on a second axis;
    here it is its own panel. A middle panel adds net sentiment for every source."""
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 4.1), sharex=True,
                            gridspec_kw={"height_ratios": [1.9, 1.3, 1.1], "hspace": 0.1})
    ax = axs[0]
    x = pd.to_datetime(pos.index)
    ax.bar(x, pos.values, width=0.8, color=POS, edgecolor=SURFACE, lw=0.8, label="positive posts")
    ax.bar(x, -neg.reindex(pos.index).values, width=0.8, color=NEG, edgecolor=SURFACE, lw=0.8,
           label="negative posts")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel(f"posts per day\n({stream_name})")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):,.0f}"))
    ax.set_title(f"Positive (up) and negative (down) {stream_name} posts per day, Sept 1 to 29", pad=16)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=6.5)
    nmax = neg.max()
    ax.set_ylim(-nmax * 1.55, pos.max() * 1.15)
    k = 0
    for e in EVENTS:
        if not e.get("short"):
            continue
        t = pd.Timestamp(e["ts"]).tz_localize(None).normalize()
        if e["label"].startswith("Amodei"):          # Saturday essay: point at the Monday bar
            t = t + pd.Timedelta(days=2)
        if t in set(x):
            yb = -neg.get(t.strftime("%Y-%m-%d"), neg.reindex(pd.to_datetime(neg.index)).get(t, nmax))
            ylab = -nmax * (1.30 if k % 2 == 0 else 1.48)
            ax.annotate(e["short"], xy=(t, yb), xytext=(t, ylab), fontsize=6, ha="center", va="center",
                        color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="#eef4fc", ec="#86b6ef", lw=0.6),
                        arrowprops=dict(arrowstyle="-", color="#86b6ef", lw=0.6))
            k += 1
    ax = axs[1]
    ends = []
    for i, (name, s) in enumerate(net.items()):
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        ma = s.rolling(3, min_periods=1).mean().dropna()
        ax.plot(ma.index, ma.values, color=SERIES[i], label=name)
        ends.append((ma.index[-1] + pd.Timedelta(hours=10), ma.iloc[-1], name, SERIES[i]))
    _spread_labels(ax, ends, 0.11)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("net sentiment\n(3-day MA)")
    ax.legend(loc="lower left", ncol=len(net), fontsize=6.5)
    _events(ax, label=False)
    ax = axs[2]
    for i, (name, s) in enumerate(cumret.items()):
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        ax.plot(s.index, s.values, color=SERIES[i], label=name)
        _direct_label(ax, s, name, SERIES[i])
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("cumulative log\nreturn (%)")
    ax.legend(loc="upper left", ncol=2, fontsize=6.5)
    _events(ax, label=False)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.align_ylabels(axs)
    fig.savefig(FIGURES / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return FIGURES / fname


def fig_souza_graph(panels: dict[str, pd.DataFrame], labels: dict[str, str],
                    fname: str = "fig2_granger.png", alpha: float = 0.05):
    """Souza et al. Figures 4 and 5: a Granger-causality graph. The returns node is in the
    middle, the sentiment nodes sit around it, and there is an arrow only where the test
    is significant (p < alpha at some lag). Arrow labels give the lag(s) and the sign
    (sum of lag coefficients); sentiment -> return arrows are coloured by sign, and
    return -> sentiment arrows are grey. Non-significant nodes are drawn faded."""
    from matplotlib.patches import Circle, FancyArrowPatch
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(7.2, 2.45))
    axs = np.atleast_1d(axs)
    for ax, (title, tab) in zip(axs, panels.items()):
        streams = list(dict.fromkeys(tab["test"]))
        k = len(streams)
        ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, k, endpoint=False)
        pos = {s: (np.cos(a) * 1.0, np.sin(a) * 1.0) for s, a in zip(streams, ang)}
        R_node, r_node = 0.27, 0.235
        ax.add_patch(Circle((0, 0), R_node, fc="#c3c2b7", ec=INK2, lw=0.8, zorder=3))
        ax.text(0, 0, "AI basket\nabnormal\nreturn", ha="center", va="center", fontsize=6, zorder=4)
        for s in streams:
            t = tab[tab["test"] == s]
            fwd = t[(t["direction"] == "sent -> ret") & (t["p"] < alpha)]
            rev = t[(t["direction"] == "ret -> sent") & (t["p"] < alpha)]
            sig = len(fwd) or len(rev)
            x0, y0 = pos[s]
            ax.add_patch(Circle((x0, y0), r_node, fc=SURFACE, ec=INK if sig else AXIS,
                                lw=0.9 if sig else 0.6, zorder=3))
            ax.text(x0, y0, labels.get(s, s), ha="center", va="center", fontsize=5.6,
                    color=INK if sig else MUTED, zorder=4)
            d = np.hypot(x0, y0)
            ux, uy = -x0 / d, -y0 / d
            start = (x0 + ux * r_node, y0 + uy * r_node)
            end = (-ux * R_node, -uy * R_node)
            if len(fwd):
                sgn = np.sign(fwd["sum_coef"]).mean()
                col = POS if sgn > 0 else NEG
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                             color=col, lw=1.4, zorder=2,
                                             connectionstyle="arc3,rad=0.12"))
                lab = ",".join(f"L{int(l)}" for l in fwd["lag"]) + (" (+)" if sgn > 0 else " (-)")
                ax.text((start[0] + end[0]) / 2 + uy * 0.2, (start[1] + end[1]) / 2 - ux * 0.2, lab,
                        fontsize=5.8, color=col, ha="center", va="center", zorder=5)
            if len(rev):
                ax.add_patch(FancyArrowPatch(end, start, arrowstyle="-|>", mutation_scale=8,
                                             color=MUTED, lw=1.0, zorder=2,
                                             connectionstyle="arc3,rad=0.12"))
                lab = ",".join(f"L{int(l)}" for l in rev["lag"])
                ax.text((start[0] + end[0]) / 2 - uy * 0.2, (start[1] + end[1]) / 2 + ux * 0.2, lab,
                        fontsize=5.8, color=MUTED, ha="center", va="center", zorder=5)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=8)
    fig.text(0.5, 0.04, f"Arrow = Granger-causes at p < {alpha} (lags shown). Blue: sentiment up then returns up; "
             "red: the opposite; grey: returns lead sentiment. Faded nodes: no significant link.",
             ha="center", fontsize=6, color=INK2)
    fig.savefig(FIGURES / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return FIGURES / fname


SAC_NAMES = {"US_MAJOR": "US Major Media", "US_GENERAL": "US General Media", "INTL_MAJOR": "Intl Major Media",
             "INTL_GENERAL": "Intl General Media", "FIN_PRESS": "Financial Press", "TECH_PRESS": "Tech Press",
             "SCIENCE": "Scientific (arXiv, AI safety)", "TWITTER": "Twitter", "REDDIT": "Reddit"}


def fig_sacerdote2(neg: pd.DataFrame, placebo: pd.DataFrame | None, fname: str = "fig3_bias.png"):
    """Sacerdote et al. Figure 2: share of negative words (Hu-Liu), standardised, by source,
    dark bars for the topic (AI) and light bars for the same sources' non-topic articles."""
    rows = []
    for g in ["US_MAJOR", "US_GENERAL", "INTL_MAJOR", "INTL_GENERAL", "FIN_PRESS", "TECH_PRESS",
              "SCIENCE", "REDDIT", "TWITTER"]:
        if placebo is not None and g in placebo.index:
            rows.append((f"{SAC_NAMES[g]} Non AI", placebo.loc[g, "huliu_z"], LIGHT_BLUE))
        if g in neg.index:
            rows.append((f"{SAC_NAMES[g]} AI", neg.loc[g, "huliu_z"], DARK_BLUE))
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.62, color=[r[2] for r in rows], edgecolor=SURFACE, lw=1)
    for yi, (_, v, _) in zip(y, rows):
        ax.text(v, yi, f" {v:.2f}" if v >= 0 else f"{v:.2f} ", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.5, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Share of negative words (Hu-Liu), standardised in the pooled sample")
    ax.set_title("Media negativity by source, AI and non-AI headlines, Sept 8 to 29")
    ax.grid(axis="y", visible=False)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 0.06, hi + 0.06)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=DARK_BLUE, label="AI headlines"),
                       Patch(color=LIGHT_BLUE, label="same outlets, non-AI (placebo)")],
              loc="lower right", fontsize=6.5)
    fig.savefig(FIGURES / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return FIGURES / fname


def fig_sacerdote1_5(daily_neg: dict[str, pd.Series], basket: pd.Series, outlets: pd.DataFrame,
                     fname: str = "fig4_sacerdote_fig1_fig5.png"):
    """(a) Sacerdote et al. Figure 1: probability of negativity over time, US vs international
    mainstream, against an objective series (for them new cases, here the AI basket). Their
    case line is on a second axis; here it is a thin panel under the negativity panel.
    (b) Sacerdote et al. Figure 5: outlet negativity vs the share of conservatives who trust
    the outlet (Pew 2019, the x-values read off their Figure 5)."""
    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[2.2, 1], hspace=0.08, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    cols = [SERIES[5], SERIES[0]]
    for i, (name, s) in enumerate(daily_neg.items()):
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        ax.plot(s.index, s.values, color=cols[i % 2], marker="o", ms=2.5, label=name)
    ax.set_ylabel("probability of\nnegativity")
    ax.set_title("(a) Negativity over time vs the AI basket", fontsize=8)
    ax.legend(loc="lower left", fontsize=6)
    ax.tick_params(labelbottom=False)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    b = basket.copy()
    b.index = pd.to_datetime(b.index)
    ax2.plot(b.index, b.values, color=SERIES[7], label="AI basket, cum. abnormal return (%)")
    ax2.axhline(0, color=AXIS, lw=0.8)
    ax2.set_ylabel("basket (%)")
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.legend(loc="upper left", fontsize=6)
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.scatter(outlets["cons_trust"], outlets["huliu_z"], s=18, color=DARK_BLUE, zorder=3)
    nudge = {"NPR": (-16, -7), "NBC": (4, 1), "CBS": (4, 1), "ABC": (4, 2)}
    for _, r in outlets.iterrows():
        ax3.annotate(r["name"], (r["cons_trust"], r["huliu_z"]), xytext=nudge.get(r["name"], (3, 2)),
                     textcoords="offset points", fontsize=6, color=INK2)
    ax3.axhline(0, color=AXIS, lw=0.8)
    ax3.set_xlim(0, 0.8)
    ax3.set_xlabel("Probability trusted source among conservatives")
    ax3.set_ylabel("Share of negative words (std.)")
    ax3.set_title("(b) Negativity and audience politics", fontsize=8)
    fig.savefig(FIGURES / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return FIGURES / fname


def fig_aloosh_vol(rv: dict[str, pd.Series], windows: dict[str, tuple], fname: str = "figA_aloosh_volatility.png"):
    """Aloosh, Choi and Ouzan Figure 4: rolling 14-hour realized volatility, one stacked
    panel per index, with the event period shaded (their trading-ban band). The x axis is
    bar order (trading hours only), so nights and weekends are not drawn as fake lines."""
    idx = list(rv.values())[0].dropna().index
    fig, axs = plt.subplots(len(rv), 1, figsize=(7.2, 1.25 * len(rv) + 0.5), sharex=True)
    xpos = np.arange(len(idx))
    for ax, (name, s) in zip(axs, rv.items()):
        ax.plot(xpos, s.reindex(idx).values, color="#8a2c2c", lw=1.2)
        for lab, (a, b) in windows.items():
            ia = np.searchsorted(idx, pd.Timestamp(a, tz=idx.tz))
            ib = np.searchsorted(idx, pd.Timestamp(b, tz=idx.tz))
            ax.axvspan(ia - 0.5, ib - 0.5, color="#e1e0d9", alpha=0.9, lw=0)
            ax.axvline(ia - 0.5, color=AXIS, lw=0.8)   # windows can touch in trading time
        ax.set_ylabel(f"Volatility of\n{name}", fontsize=7)
    for lab, (a, b) in windows.items():
        ia = np.searchsorted(idx, pd.Timestamp(a, tz=idx.tz))
        axs[0].text(ia, 1.03, lab, transform=axs[0].get_xaxis_transform(), fontsize=6, color=INK2)
    days = pd.Series(xpos, index=idx).groupby(idx.date).first()
    axs[-1].set_xticks(days.values, [f"{pd.Timestamp(d).day}" for d in days.index])
    axs[-1].set_xlabel("September, trading days (7 hourly bars each)")
    fig.suptitle("Realized volatility in 14-hour rolling windows, shock periods shaded", x=0.02, ha="left",
                 fontsize=9, fontweight="bold")
    fig.savefig(FIGURES / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return FIGURES / fname


def fig_thelwall(share: pd.Series, pos_strength: pd.Series, neg_strength: pd.Series, stream_name: str,
                 fname: str = "figB_thelwall_event.png"):
    """Thelwall slides (Chile / #oscars examples): top, the proportion of posts matching the
    risk terms over time; bottom, average positive and negative sentiment strength."""
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 3.4), sharex=True, gridspec_kw={"hspace": 0.08})
    x = share.index.tz_localize(None) if share.index.tz is not None else share.index
    axs[0].plot(x, share.values * 100, color=SERIES[0], lw=1.0)
    axs[0].set_ylabel("% of posts\nmatching risk terms")
    axs[0].set_title(f"Proportion of {stream_name} posts about AI risk, and sentiment strength (hourly)",
                     pad=22)
    _events(axs[0], label=True)
    axs[1].plot(x, pos_strength.values, color=POS, lw=1.0, label="avg. positive strength")
    axs[1].plot(x, neg_strength.values, color=NEG, lw=1.0, label="avg. negative strength")
    axs[1].set_ylabel("sentiment strength\n(VADER pos / neg)")
    axs[1].legend(loc="upper left", ncol=2, fontsize=6.5)
    _events(axs[1], label=False)
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.savefig(FIGURES / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return FIGURES / fname

"""Build one headline-level news frame from GDELT + Google News RSS, plus arXiv.

Outlet groups follow Sacerdote, Sehgal & Cook (2020): US major, US general,
international major, international general. We add financial press, tech press
and a scientific benchmark (arXiv abstracts), the analog of their journals.
"""
from __future__ import annotations

import glob
import json
import re
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import feedparser
import pandas as pd

from src.config import OUTLET_GROUPS, RAW
from src.text_clean import fix_gdelt_title

NEWS = RAW / "news"
RISK_RE = re.compile(r"extinct|existential|doom|kill (?:us|all|every)|wipe out|catastroph|"
                     r"superintelligen|\bsafety\b|end humanity|destroy human|p\(doom\)|"
                     r"slow(?:down| down)|pause", re.I)
AI_RE = re.compile(r"\bai\b|\ba\.i\.|artificial intelligence|anthropic|openai|chatgpt|claude|"
                   r"gemini|\bagi\b|nvidia|chatbot|llm|superintelligen|deepmind|machine learning",
                   re.I)


def outlet_group(domain: str, country: str | None) -> str:
    d = (domain or "").lower().removeprefix("www.")
    for g, doms in OUTLET_GROUPS.items():
        if any(d == x or d.endswith("." + x) for x in doms):
            return g
    if country is None:
        return "UNKNOWN"
    return "US_GENERAL" if country == "United States" else "INTL_GENERAL"


def outlet_name(domain: str) -> str:
    d = (domain or "").lower().removeprefix("www.")
    for x in sum(OUTLET_GROUPS.values(), []):
        if d == x or d.endswith("." + x):
            return x
    return d


def load_gdelt() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(NEWS / "gdelt" / "*.json")):
        blob = json.load(open(f))
        for a in blob["articles"]:
            rows.append({"url": a.get("url"), "title_raw": a.get("title"),
                         "domain": a.get("domain"), "country": a.get("sourcecountry"),
                         "seendate": a.get("seendate"), "topic": blob["topic"],
                         "scope": blob["scope"], "src": "gdelt"})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts_utc"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", utc=True)
    df["title"] = df["title_raw"].map(fix_gdelt_title)
    # one row per URL; remember every topic it matched
    topics = df.groupby("url")["topic"].agg(lambda s: sorted(set(s)))
    df = df.sort_values("ts_utc").drop_duplicates("url").set_index("url")
    df["topics"] = topics
    return df.reset_index()


CC_TLDS = (".uk", ".in", ".au", ".ca", ".pk", ".nz", ".ie", ".sg", ".za", ".ng", ".ke", ".ph",
           ".my", ".ae", ".hk", ".bd", ".lk", ".np", ".gh")


def _rss_rows(blob: dict, kind: str) -> list[dict]:
    rows = []
    for it in blob["items"]:
        title = it.get("title") or ""
        src = it.get("source") or ""
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        href = it.get("source_href") or ""
        dom = (urlparse(href).netloc or blob.get("site", "")).lower().removeprefix("www.")
        try:
            ts = parsedate_to_datetime(it.get("published"))
        except Exception:  # noqa: BLE001
            ts = None
        rows.append({"url": it.get("link"), "title": title.strip(), "domain": dom, "ts_utc": ts,
                     "source_name": src, "site_query": blob.get("site"),
                     "edition": blob.get("edition"), "topic": blob.get("topic", "AI_GEN"),
                     "src": kind})
    return rows


def load_rss() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(NEWS / "rss" / "*.json")):
        rows += _rss_rows(json.load(open(f)), "rss_site")
    for f in glob.glob(str(NEWS / "rss_editions" / "*.json")):
        rows += _rss_rows(json.load(open(f)), "rss_edition")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df.dropna(subset=["ts_utc"])
    # site: search sometimes returns syndicated copies hosted elsewhere; drop those
    site = df["src"] == "rss_site"

    def same_site(r):
        q = r["site_query"]
        return r["domain"] == q or r["domain"].endswith("." + q) or q.endswith(r["domain"])
    ok = pd.Series(True, index=df.index)
    ok[site] = df[site].apply(same_site, axis=1)
    df = df[ok].copy()
    # country proxy: ccTLD first, else the edition the item came from
    def country(r):
        if r["domain"].endswith(CC_TLDS):
            return "Other"
        if r["src"] == "rss_edition":
            return "United States" if r["edition"] == "US" else "Other"
        return None
    df["country"] = df.apply(country, axis=1)
    df["topics"] = df["topic"].map(lambda t: [t])
    return df


def build_news() -> pd.DataFrame:
    g, r = load_gdelt(), load_rss()
    df = pd.concat([g, r], ignore_index=True, sort=False)
    df["title"] = df["title"].fillna("").str.strip()
    df = df[df["title"].str.len() > 15]
    df["group"] = [outlet_group(d, c) for d, c in zip(df["domain"], df["country"])]
    # site: rows outside the named groups (rare) fall back on the TLD
    unk = df["group"] == "UNKNOWN"
    df.loc[unk, "group"] = df.loc[unk, "domain"].map(
        lambda d: "INTL_GENERAL" if d.endswith(CC_TLDS) else "US_GENERAL")
    df["outlet"] = df["domain"].map(outlet_name)
    df["tkey"] = df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    # remember EVERY query a headline came back from before dropping duplicates, so that
    # group comparisons can keep only items sampled by the same (AI_GEN) query
    tops = df.groupby(["tkey", "outlet"])["topics"].agg(lambda s: sorted({t for x in s for t in x}))
    df = df.sort_values(["src", "ts_utc"]).drop_duplicates(["tkey", "outlet"])
    df["topics"] = [tops.loc[(k, o)] for k, o in zip(df["tkey"], df["outlet"])]
    df["from_gen"] = df["topics"].map(lambda t: "AI_GEN" in t)
    df["is_ai"] = df["title"].str.contains(AI_RE)
    df["is_risk"] = df["title"].str.contains(RISK_RE) & df["is_ai"]
    df["is_placebo"] = df["topics"].map(lambda t: t == ["PLACEBO"]) & ~df["is_ai"]
    return df.drop(columns=["tkey"]).reset_index(drop=True)


def urls_for_fulltext(n_general: int = 2500, seed: int = 7871) -> list[str]:
    """Body text for every named-group GDELT article + a random sample of the rest."""
    g = load_gdelt()
    if g.empty:
        return []
    g["group"] = [outlet_group(d, c) for d, c in zip(g["domain"], g["country"])]
    named = g[g["group"].isin(["US_MAJOR", "INTL_MAJOR", "FIN_PRESS", "TECH_PRESS"])]
    rest = g[~g.index.isin(named.index)]
    rest = rest.sample(min(n_general, len(rest)), random_state=seed)
    return named["url"].tolist() + rest["url"].tolist()


def load_arxiv() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(NEWS / "arxiv" / "*.xml"))):
        for e in feedparser.parse(open(f, "rb").read()).entries:
            rows.append({"url": e.get("id"), "title": " ".join(e.get("title", "").split()),
                         "abstract": " ".join(e.get("summary", "").split()),
                         "ts_utc": pd.to_datetime(e.get("published"), utc=True),
                         "group": "SCIENCE", "outlet": "arxiv.org"})
    return pd.DataFrame(rows).drop_duplicates("url")

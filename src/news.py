"""News collection: GDELT DOC 2.0, Google News RSS, arXiv, and article full text.

GDELT artlist caps each call at 250 records, so time is sliced into 6-hour
windows. Per window we make one US-only call and one all-English call and take
non-US articles from the second one locally (more robust than relying on a
negated country filter). GDELT asks for one request every 5 seconds and
answers with a plain-text warning when you go faster, so we back off on that.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import feedparser
import requests

from src.config import (ARXIV_API, BASELINE_START, GDELT_DOC, GDELT_QUERIES, GDELT_SLEEP,
                        GDELT_WINDOW_HOURS, GOOGLE_NEWS_RSS, RAW, RSS_SITES, UTC, WINDOW_END)

OUT = RAW / "news"
UA = {"User-Agent": "Mozilla/5.0 (academic research; NYU FRE-GY 7871 course project)"}
_last_call = [0.0]


def _gdelt(params: dict, retries: int = 8) -> dict | None:
    for attempt in range(retries):
        wait = GDELT_SLEEP - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = requests.get(GDELT_DOC, params=params, headers=UA, timeout=60)
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
            continue
        txt = r.text.strip()
        if txt.startswith("Please limit") or r.status_code == 429:
            # every retry during a ban seems to extend it, so wait a long time
            print(f"gdelt rate-limited, sleeping {300 * (attempt + 1)} s", flush=True)
            time.sleep(300 * (attempt + 1))
            continue
        if not txt:
            return {}
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # GDELT sometimes returns invalid JSON escapes in titles
            try:
                return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", txt))
            except json.JSONDecodeError:
                print("bad json:", txt[:120])
                return None
    return None


def _fmt(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y%m%d%H%M%S")


def gdelt_windows(hours: int, start=BASELINE_START, end=WINDOW_END):
    t = start
    while t < end:
        yield t, min(t + timedelta(hours=hours), end)
        t += timedelta(hours=hours)


def gdelt_plan():
    """(topic, scope, since, until) jobs. scope: US / ALL / UK / IN."""
    plan = []
    for topic in ("AI_GEN", "AI_RISK"):
        for s, u in gdelt_windows(GDELT_WINDOW_HOURS):
            plan += [(topic, "US", s, u), (topic, "ALL", s, u)]
        for s, u in gdelt_windows(24):
            plan += [(topic, "UK", s, u), (topic, "IN", s, u)]
    for s, u in gdelt_windows(24):
        plan += [("PLACEBO", "US", s, u), ("PLACEBO", "ALL", s, u)]
    return plan


def collect_gdelt_artlists() -> int:
    d = OUT / "gdelt"
    d.mkdir(parents=True, exist_ok=True)
    plan = gdelt_plan()
    got = 0
    for i, (topic, scope, s, u) in enumerate(plan):
        path = d / f"{topic}_{scope}_{_fmt(s)}.json"
        if path.exists():
            continue
        q = GDELT_QUERIES[topic] + " sourcelang:english"
        if scope != "ALL":
            q += f" sourcecountry:{scope}"
        js = _gdelt({"query": q, "mode": "artlist", "maxrecords": 250, "format": "json",
                     "startdatetime": _fmt(s), "enddatetime": _fmt(u), "sort": "hybridrel"})
        if js is None:
            print("failed", topic, scope, s)
            continue
        arts = js.get("articles", [])
        path.write_text(json.dumps({"topic": topic, "scope": scope, "since": s.isoformat(),
                                    "until": u.isoformat(), "articles": arts}))
        got += len(arts)
        if i % 25 == 0:
            print(f"gdelt {i}/{len(plan)} {topic} {scope} {s:%m-%d %H}h -> {len(arts)} (total new {got})",
                  flush=True)
    return got


def collect_gdelt_timelines() -> None:
    d = OUT / "gdelt_timeline"
    d.mkdir(parents=True, exist_ok=True)
    for topic in GDELT_QUERIES:
        for scope in ("US", "UK", "IN", "ALL"):
            for mode in ("timelinevolraw", "timelinetone"):
                path = d / f"{topic}_{scope}_{mode}.json"
                if path.exists():
                    continue
                q = GDELT_QUERIES[topic] + " sourcelang:english"
                if scope != "ALL":
                    q += f" sourcecountry:{scope}"
                js = _gdelt({"query": q, "mode": mode, "format": "json", "timelinesmooth": 0,
                             "startdatetime": _fmt(BASELINE_START), "enddatetime": _fmt(WINDOW_END)})
                if js is not None:
                    path.write_text(json.dumps(js))
                    print("timeline", topic, scope, mode, flush=True)


# ---------------------------------------------------------------- Google News RSS
RSS_QUERIES = {
    "AI_GEN": '(AI OR "artificial intelligence" OR Anthropic OR OpenAI OR ChatGPT)',
    "AI_RISK": ('(AI OR Anthropic OR OpenAI) (extinction OR existential OR doom OR "AI safety" '
                'OR "kill all humans" OR superintelligence OR slowdown)'),
    "PLACEBO": '("Federal Reserve" OR inflation OR "housing market" OR "oil prices" OR "retail sales")',
}
# Google News editions stand in for source country when there is no site: filter
EDITIONS = {"US": ("en-US", "US", "US:en"), "GB": ("en-GB", "GB", "GB:en"),
            "IN": ("en-IN", "IN", "IN:en"), "AU": ("en-AU", "AU", "AU:en"),
            "CA": ("en-CA", "CA", "CA:en")}


def _rss_get(q: str, edition: str = "US") -> list[dict]:
    hl, gl, ceid = EDITIONS[edition]
    url = f"{GOOGLE_NEWS_RSS}?q={urllib.parse.quote(q)}&hl={hl}&gl={gl}&ceid={urllib.parse.quote(ceid)}"
    r = None
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    if r is None or r.status_code != 200:
        return []
    feed = feedparser.parse(r.content)
    return [{"title": e.get("title"), "link": e.get("link"), "published": e.get("published"),
             "source": (e.get("source") or {}).get("title"),
             "source_href": (e.get("source") or {}).get("href"),
             "summary": e.get("summary")} for e in feed.entries]


def collect_rss(topic: str = "AI_GEN", days_per_window: int = 3) -> int:
    """site: queries on the named outlets (US major, intl major, some finance)."""
    d = OUT / "rss"
    d.mkdir(parents=True, exist_ok=True)
    total = 0
    tag = "" if topic == "AI_GEN" else f"{topic}_"
    for site in RSS_SITES:
        t = BASELINE_START
        while t < WINDOW_END:
            u = min(t + timedelta(days=days_per_window), WINDOW_END + timedelta(days=1))
            path = d / f"{tag}{site.replace('/', '_')}_{t:%Y%m%d}.json"
            if not path.exists():
                q = f"{RSS_QUERIES[topic]} site:{site} after:{t:%Y-%m-%d} before:{u:%Y-%m-%d}"
                items = _rss_get(q)
                path.write_text(json.dumps({"site": site, "topic": topic, "since": t.isoformat(),
                                            "until": u.isoformat(), "items": items}))
                total += len(items)
                time.sleep(1.5)
            t = u
        print("rss", topic, site, flush=True)
    return total


def collect_rss_editions(topics=("AI_GEN", "AI_RISK", "PLACEBO")) -> int:
    """No site: filter, one call per edition x day x topic -> general outlets."""
    d = OUT / "rss_editions"
    d.mkdir(parents=True, exist_ok=True)
    total = 0
    for topic in topics:
        for ed in EDITIONS:
            t = BASELINE_START
            while t < WINDOW_END:
                u = t + timedelta(days=1)
                path = d / f"{topic}_{ed}_{t:%Y%m%d}.json"
                if not path.exists():
                    q = f"{RSS_QUERIES[topic]} after:{t:%Y-%m-%d} before:{u:%Y-%m-%d}"
                    items = _rss_get(q, ed)
                    path.write_text(json.dumps({"edition": ed, "topic": topic, "since": t.isoformat(),
                                                "until": u.isoformat(), "items": items}))
                    total += len(items)
                    time.sleep(1.5)
                t = u
            print("rss edition", topic, ed, flush=True)
    return total


# ---------------------------------------------------------------- arXiv (scientific benchmark)
def collect_arxiv(page: int = 1000) -> int:
    """Paginate the arXiv API (it caps a page, and our first pull hit the cap)."""
    d = OUT / "arxiv"
    d.mkdir(parents=True, exist_ok=True)
    q = ("(cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.CY) AND "
         "(abs:safety OR abs:alignment OR abs:risk OR abs:catastrophic OR abs:existential) AND "
         f"submittedDate:[{BASELINE_START:%Y%m%d}0000 TO {WINDOW_END:%Y%m%d}2359]")
    total, start = 0, 0
    while True:
        path = d / f"arxiv_safety_{start:05d}.xml"
        if not path.exists():
            params = {"search_query": q, "start": start, "max_results": page,
                      "sortBy": "submittedDate", "sortOrder": "ascending"}
            r = requests.get(ARXIV_API, params=params, headers=UA, timeout=180)
            path.write_bytes(r.content)
            time.sleep(3.5)  # arXiv asks for 3 s between calls
        n = len(feedparser.parse(path.read_bytes()).entries)
        total += n
        if n < page or start > 20000:
            break
        start += page
    return total


# ---------------------------------------------------------------- full text
def _fetch_text(url: str) -> dict:
    import trafilatura
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code != 200:
            return {"url": url, "status": r.status_code, "text": None}
        txt = trafilatura.extract(r.text, include_comments=False, include_tables=False)
        return {"url": url, "status": 200, "text": txt}
    except Exception as e:  # noqa: BLE001
        return {"url": url, "status": -1, "text": None, "err": str(e)[:100]}


def collect_fulltext(urls: list[str], workers: int = 16) -> int:
    d = OUT / "fulltext"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "fulltext.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["url"] for line in path.open()}
    todo = [u for u in dict.fromkeys(urls) if u not in done]
    n_ok = 0
    with path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_text, u) for u in todo]
        for i, fut in enumerate(as_completed(futs)):
            row = fut.result()
            n_ok += bool(row.get("text"))
            f.write(json.dumps(row) + "\n")
            if i % 500 == 0:
                print(f"fulltext {i}/{len(todo)} ok={n_ok}", flush=True)
    return n_ok

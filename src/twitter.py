"""Twitter/X collection through twitterapi.io advanced search.

Design choices (see PROGRESS.md for why):
- The official X API only covers the last 7 days on pay-per-use, so we buy
  historical search from a data provider, as the Thelwall slides suggest.
- Lean, bar-aligned sampling: inside trading hours each hourly bar gets two
  30-minute sub-windows; overnight and weekend time gets 2-hour sub-windows
  (all of it pools into the next 09:30 bar). One "Latest" page (up to 20
  tweets) per sub-window, so the sample is spread across time instead of
  piling up in the final minutes.
- The provider bills per returned tweet, so every page is cached to disk and
  re-runs are free.
- Volume: if a sub-window returns < 20 tweets with no next page, that is the
  exact count. If it is capped, we estimate the rate from the time span the
  20 tweets cover (Latest results come newest first).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.config import (ET, OFFHOURS_SUBWINDOW_MIN, RAW, ROOT, TRADING_SUBWINDOW_MIN,
                        TWITTER_ENDPOINT, TWITTER_QUERIES, TWITTER_QUERY_EXTRA, TWITTER_START,
                        UTC, WINDOW_END)

load_dotenv(ROOT / ".env")
OUT = RAW / "twitter"
KEEP_FIELDS = ["id", "text", "createdAt", "lang", "likeCount", "retweetCount", "replyCount",
               "quoteCount", "viewCount", "bookmarkCount", "isReply", "inReplyToId",
               "conversationId", "url"]


def _api_key() -> str:
    key = os.getenv("TWITTERAPI_IO_KEY")
    if not key:
        raise SystemExit("TWITTERAPI_IO_KEY missing: put it in Assignment_4/.env")
    return key


def _is_trading(t: datetime) -> bool:
    """Regular session 09:30 to 16:00 ET on weekdays (no exchange holiday inside Sept 8 to 29)."""
    t = t.astimezone(ET)
    mins = t.hour * 60 + t.minute
    return t.weekday() < 5 and 9 * 60 + 30 <= mins < 16 * 60


def lean_windows(start: datetime = TWITTER_START, end: datetime = WINDOW_END):
    """Contiguous sub-windows covering start..end: 30 minutes inside trading hours (2 per
    hourly bar), 120 minutes outside them (these hours pool into the next 09:30 bar)."""
    t = start.astimezone(ET)
    while t < end:
        step = timedelta(minutes=TRADING_SUBWINDOW_MIN if _is_trading(t) else OFFHOURS_SUBWINDOW_MIN)
        u = t + step
        # never let an off-hours window run past the 09:30 open, or a trading one past 16:00
        open_ = t.replace(hour=9, minute=30, second=0, microsecond=0)
        close = t.replace(hour=16, minute=0, second=0, microsecond=0)
        if not _is_trading(t) and t < open_ < u and t.weekday() < 5:
            u = open_
        if _is_trading(t) and u > close:
            u = close
        u = min(u, end)
        yield t, u
        t = u


def _slim(tw: dict) -> dict:
    row = {k: tw.get(k) for k in KEEP_FIELDS}
    a = tw.get("author") or {}
    row.update({"author_user": a.get("userName"), "author_followers": a.get("followers"),
                "author_verified": a.get("isBlueVerified"), "author_created": a.get("createdAt")})
    return row


import threading

_throttle_lock = threading.Lock()
_next_slot = [0.0]
MIN_INTERVAL = [float(os.getenv("TWITTER_MIN_INTERVAL", "0"))]   # seconds between calls, all threads


def _wait_slot():
    """Global pacing across worker threads. Starts unthrottled; the first 429 that names a
    QPS limit (free tier: 'one request every 5 seconds') switches pacing on."""
    with _throttle_lock:
        now = time.time()
        wait = _next_slot[0] - now
        _next_slot[0] = max(now, _next_slot[0]) + MIN_INTERVAL[0]
    if wait > 0:
        time.sleep(wait)


def fetch_page(query: str, since: datetime, until: datetime, key: str, retries: int = 8) -> dict:
    q = f"{query}{TWITTER_QUERY_EXTRA} since_time:{int(since.timestamp())} until_time:{int(until.timestamp())}"
    params = {"query": q, "queryType": "Latest", "cursor": ""}
    for attempt in range(retries):
        _wait_slot()
        try:
            r = requests.get(TWITTER_ENDPOINT, params=params, headers={"X-API-Key": key},
                             timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                if "5 seconds" in r.text and MIN_INTERVAL[0] < 5.2:
                    MIN_INTERVAL[0] = 5.2
                    print("rate limit: free tier, pacing to one call per 5.2 s", flush=True)
                time.sleep(max(MIN_INTERVAL[0], 2 ** attempt))
                continue
            if r.status_code in (402, 403) or "credit" in r.text.lower():
                raise SystemExit(f"stopping: account out of credits or not allowed ({r.status_code}): {r.text[:200]}")
            if r.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gave up on {q}")


def account_credits(key: str) -> dict:
    r = requests.get("https://api.twitterapi.io/oapi/my/info", headers={"X-API-Key": key}, timeout=20)
    return r.json() if r.status_code == 200 else {}


def collect_subwindow(family: str, since: datetime, until: datetime, key: str) -> dict:
    """Fetch one page for one sub-window, cache it, return a small summary."""
    fdir = OUT / family
    fdir.mkdir(parents=True, exist_ok=True)
    tag = since.astimezone(UTC).strftime("%Y%m%dT%H%M")
    path = fdir / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text())["meta"]
    data = fetch_page(TWITTER_QUERIES[family], since, until, key)
    tweets = [_slim(t) for t in data.get("tweets", [])]
    meta = {"family": family, "since_utc": since.astimezone(UTC).isoformat(),
            "until_utc": until.astimezone(UTC).isoformat(), "n": len(tweets),
            "has_next_page": bool(data.get("has_next_page"))}
    path.write_text(json.dumps({"meta": meta, "tweets": tweets}))
    return meta


def jobs(families=("GEN", "RISK", "FIN"), start=TWITTER_START, end=WINDOW_END):
    for fam in families:
        for s, u in lean_windows(start, end):
            yield fam, s, u


def collect_all(families=("GEN", "RISK", "FIN"), start=TWITTER_START, end=WINDOW_END,
                workers: int = 8, limit: int | None = None) -> list[dict]:
    key = _api_key()
    todo = list(jobs(families, start, end))
    if limit:
        todo = todo[:limit]
    metas, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(collect_subwindow, f, s, u, key): (f, s) for f, s, u in todo}
        for fut in as_completed(futs):
            try:
                metas.append(fut.result())
            except Exception as e:  # noqa: BLE001
                f, s = futs[fut]
                print(f"FAILED {f} {s}: {e}")
            done += 1
            if done % 250 == 0:
                n = sum(m["n"] for m in metas)
                print(f"{done}/{len(todo)} sub-windows, {n} tweets so far", flush=True)
    return metas


def fetch_tweets_by_id(ids: list[str]) -> list[dict]:
    key = _api_key()
    r = requests.get("https://api.twitterapi.io/twitter/tweets",
                     params={"tweet_ids": ",".join(ids)}, headers={"X-API-Key": key},
                     timeout=30)
    r.raise_for_status()
    return r.json().get("tweets", [])


def load_raw() -> list[dict]:
    """All cached tweets with family + sub-window meta attached."""
    rows = []
    for fam_dir in sorted(OUT.glob("*")):
        if not fam_dir.is_dir():
            continue
        for p in sorted(fam_dir.glob("*.json")):
            blob = json.loads(p.read_text())
            m = blob["meta"]
            for t in blob["tweets"]:
                t = dict(t)
                t["family"] = m["family"]
                t["win_since_utc"] = m["since_utc"]
                t["win_until_utc"] = m["until_utc"]
                t["win_capped"] = m["has_next_page"] or m["n"] >= 20
                rows.append(t)
    return rows


def load_window_meta() -> list[dict]:
    metas = []
    for p in sorted(OUT.glob("*/*.json")):
        blob = json.loads(p.read_text())
        m = dict(blob["meta"])
        ts = [t.get("createdAt") for t in blob["tweets"] if t.get("createdAt")]
        m["created"] = ts
        metas.append(m)
    return metas


if __name__ == "__main__":
    print(len(list(jobs())), "sub-windows planned")

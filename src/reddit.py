"""Reddit collection via the Arctic Shift archive (free, no auth).

Arctic Shift's full-text params only work with a subreddit filter, and they fail
on very active subs. So:
- AI subs: pull every post + comment in the window (topic is already AI).
- finance subs: pull every post, then keep AI/cashtag ones locally. For comments
  we try the `body` keyword search and fall back to skipping if the server refuses.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import requests

from src.config import (BASELINE_START, RAW, REDDIT_AI_SUBS, REDDIT_API, REDDIT_FIN_SUBS,
                        WINDOW_END)

OUT = RAW / "reddit"
POST_FIELDS = "id,created_utc,subreddit,author,title,selftext,score,num_comments,url,link_flair_text"
COMMENT_FIELDS = "id,created_utc,subreddit,author,body,score,link_id,parent_id"


def _get(path: str, params: dict, retries: int = 6) -> list[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(f"{REDDIT_API}/{path}", params=params, timeout=60)
            if r.status_code == 200:
                js = r.json()
                if js.get("error"):
                    raise RuntimeError(js["error"])
                return js.get("data", [])
            if r.status_code == 429:
                reset = float(r.headers.get("X-RateLimit-Reset", 2 ** attempt))
                time.sleep(min(max(reset, 1), 60))
                continue
            if r.status_code >= 500 or (r.status_code == 422 and "Timeout" in r.text):
                time.sleep(3 * 2 ** attempt)  # server asks us to slow down
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gave up on {path} {params}")


def pull(kind: str, sub: str, start: datetime = BASELINE_START, end: datetime = WINDOW_END,
         extra: dict | None = None, max_items: int = 400_000) -> list[dict]:
    """Paginate ascending by created_utc until `end`."""
    fields = POST_FIELDS if kind == "posts" else COMMENT_FIELDS
    after = int(start.timestamp())
    before = int(end.timestamp())
    rows, seen = [], set()
    while True:
        # "auto" (100-1000 rows) is not allowed together with keyword search
        limit = 100 if (extra or kind == "comments") else "auto"
        params = {"subreddit": sub, "after": after, "before": before, "limit": limit,
                  "sort": "asc", "fields": fields}
        if extra:
            params.update(extra)
        batch = _get(f"{kind}/search", params)
        new = [b for b in batch if b["id"] not in seen]
        if not new:
            break
        for b in new:
            seen.add(b["id"])
        rows.extend(new)
        last = max(int(b["created_utc"]) for b in new)
        if last <= after or len(rows) >= max_items:
            break
        after = last  # created_utc ties are handled by the seen-set
        time.sleep(0.6)
    return rows


def pull_stratified(kind: str, sub: str, hours: int = 6, per_window: int = 100,
                    start: datetime = BASELINE_START, end: datetime = WINDOW_END) -> list[dict]:
    """Time-stratified sample: the first `per_window` items of every `hours`-hour window
    (ascending; descending sort times out server-side on big subs). Each window is
    cached to its own file, so an interrupted run resumes instead of starting over."""
    from datetime import timedelta
    fields = POST_FIELDS if kind == "posts" else COMMENT_FIELDS
    cache = OUT / "strat_cache" / f"{kind}_{sub}"
    cache.mkdir(parents=True, exist_ok=True)
    rows, seen, t = [], set(), start
    while t < end:
        u = min(t + timedelta(hours=hours), end)
        f = cache / f"{int(t.timestamp())}_{hours}h.json"
        if f.exists():
            batch = json.loads(f.read_text())
        else:
            params = {"subreddit": sub, "after": int(t.timestamp()), "before": int(u.timestamp()),
                      "limit": per_window, "sort": "asc", "fields": fields}
            batch = _get(f"{kind}/search", params)
            f.write_text(json.dumps(batch))
            time.sleep(0.4)
        for b in batch:
            if b["id"] not in seen:
                seen.add(b["id"])
                rows.append(b)
        t = u
    return rows


BIG_AI_SUBS = {"singularity", "OpenAI", "ClaudeAI"}


def collect_all() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    plan = [("posts", s, None) for s in REDDIT_AI_SUBS + REDDIT_FIN_SUBS]
    # ArtificialInteligence: posts only (its comments add little and the API is slow)
    plan += [("comments", s, None) for s in REDDIT_AI_SUBS if s != "ArtificialInteligence"]
    # r/technology: posts only (AI is in a huge share of its comments; the keyword pull ran 25+ min)
    plan += [("comments", s, {"body": "AI"}) for s in REDDIT_FIN_SUBS if s != "technology"]
    for kind, sub, extra in plan:
        tag = f"{kind}_{sub}" + ("_kwAI" if extra else "")
        path = OUT / f"{tag}.jsonl"
        if path.exists():
            summary[tag] = sum(1 for _ in path.open())
            continue
        try:
            if kind == "comments" and extra is None and sub in BIG_AI_SUBS:
                rows = pull_stratified(kind, sub)
                tag_note = " (stratified 6h x 100)"
            else:
                rows = pull(kind, sub, extra=extra)
                tag_note = ""
        except Exception as e:  # noqa: BLE001
            print(f"skip {tag}: {e}")
            summary[tag] = f"error: {e}"
            continue
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary[tag] = len(rows)
        print(tag + tag_note, len(rows), flush=True)
    return summary


def load_raw() -> list[dict]:
    rows = []
    for p in sorted(OUT.glob("*.jsonl")):
        kind = p.stem.split("_")[0]
        for line in p.open():
            r = json.loads(line)
            r["kind"] = kind
            r["text"] = (r.get("title", "") + "\n" + (r.get("selftext") or "")).strip() \
                if kind == "posts" else r.get("body", "")
            rows.append(r)
    return rows

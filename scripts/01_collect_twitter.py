"""Collect tweets for the three query families (GEN, RISK, FIN) + the seed post.

Usage:
  python scripts/01_collect_twitter.py --probe     # 6 sub-windows only, check cost/format
  python scripts/01_collect_twitter.py             # full run (cached, resumable)
"""
import _bootstrap  # noqa: F401
import argparse
import json

from src.config import RAW, SEED_TWEETS
from src.twitter import _api_key, account_credits, collect_all, fetch_tweets_by_id, jobs

ap = argparse.ArgumentParser()
ap.add_argument("--probe", action="store_true")
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

planned = list(jobs())
print(f"{len(planned)} sub-windows planned (max ~{len(planned) * 20:,} tweets)")
cred = account_credits(_api_key())
need = len(planned) * 20 * 15          # $0.15 per 1,000 tweets = 15 credits per tweet
have = cred.get("recharge_credits", 0) + cred.get("total_bonus_credits", 0)
print(f"credits: have {have:,}, full pull needs up to {need:,}")
if not args.probe and have < need * 0.5:
    raise SystemExit("not enough credits for the full pull; top up the twitterapi.io account first")

if args.probe:
    metas = collect_all(limit=6, workers=2)
    for m in metas:
        print(m)
else:
    seed_path = RAW / "twitter" / "seed_tweets.json"
    if not seed_path.exists():
        seed = fetch_tweets_by_id(list(SEED_TWEETS.values()))
        seed_path.write_text(json.dumps(seed, indent=1))
        print("seed tweets saved:", len(seed))
    metas = collect_all(workers=args.workers)
    print("done:", len(metas), "sub-windows,", sum(m["n"] for m in metas), "tweets")

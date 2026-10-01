"""Clean + score every source. Each source is cached in data/processed/.

  python scripts/05_clean_and_score.py twitter
  python scripts/05_clean_and_score.py reddit
  python scripts/05_clean_and_score.py news
  python scripts/05_clean_and_score.py arxiv
"""
import _bootstrap  # noqa: F401
import re
import sys

import numpy as np
import pandas as pd

from src.config import BASELINE_START, PROCESSED, RANDOM_SEED, REDDIT_FIN_SUBS, WINDOW_END
from src.news_corpus import AI_RE, RISK_RE, build_news, load_arxiv
from src.sentiment import add_class_schema, score_frame
from src.text_clean import clean_frame

PROCESSED.mkdir(parents=True, exist_ok=True)
CASHTAG_RE = re.compile(r"\$(?:NVDA|MSFT|GOOGL?|META|AMZN|AMD|AVGO|TSM|ORCL|MU|ARM|PLTR|CRWV|SMCI)\b", re.I)


def in_range(df, col="ts_utc"):
    t = pd.to_datetime(df[col], utc=True)
    return df[(t >= BASELINE_START) & (t < WINDOW_END)]


def twitter_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signal-to-noise filters, applied in order, with a waterfall of what each removes.
    Thresholds and reasons live in src/config.py (TW_*)."""
    from src.config import (TW_MAX_PER_AUTHOR_DAY, TW_MIN_ACCOUNT_AGE_DAYS, TW_MIN_FOLLOWERS,
                            TW_MIN_VIEWS, TW_PROMO_RE)
    steps = [("downloaded (all families, deduplicated by family x id)", len(df))]
    d = df.copy()
    d = d[d["lang"].fillna("en").eq("en")]
    steps.append(("English (lang field)", len(d)))
    if d["viewCount"].notna().mean() > 0.5:
        d = d[d["viewCount"].fillna(0) >= TW_MIN_VIEWS]
        steps.append((f"views >= {TW_MIN_VIEWS}", len(d)))
    else:
        print("WARNING: viewCount mostly missing, views filter skipped")
    fol = pd.to_numeric(d["author_followers"], errors="coerce")
    d = d[fol.fillna(0) >= TW_MIN_FOLLOWERS]
    steps.append((f"author followers >= {TW_MIN_FOLLOWERS}", len(d)))
    created = pd.to_datetime(d["author_created"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
    age = (d["ts_utc"] - created).dt.days
    d = d[age.isna() | (age >= TW_MIN_ACCOUNT_AGE_DAYS)]
    steps.append((f"account at least {TW_MIN_ACCOUNT_AGE_DAYS} days old", len(d)))
    d = d[~d["text"].str.contains(TW_PROMO_RE, case=False, regex=True, na=False)]
    steps.append(("no giveaway / promo / pump language", len(d)))
    # relevance: the AI term must be in the tweet text itself (not only in a handle or link)
    body = d["text"].str.replace(r"https?://\S+|@\w+", " ", regex=True)
    rel = np.where(d["family"].eq("FIN"), body.str.contains(CASHTAG_RE), body.str.contains(AI_RE))
    d = d[rel]
    steps.append(("AI term (or cashtag for FIN) in the text itself", len(d)))
    d["day"] = d["ts_utc"].dt.tz_convert("America/New_York").dt.date
    d = d.sort_values("engagement", ascending=False).groupby(["family", "author_user", "day"]).head(
        TW_MAX_PER_AUTHOR_DAY).drop(columns=["day"])
    steps.append((f"at most {TW_MAX_PER_AUTHOR_DAY} tweets per author per day per family", len(d)))
    return d, pd.DataFrame(steps, columns=["step", "tweets_left"])


def twitter():
    from src.config import TABLES, TW_MIN_WORDS
    from src.twitter import load_raw
    df = pd.DataFrame(load_raw())
    print("raw tweets:", len(df), df["family"].value_counts().to_dict())
    df["ts_utc"] = pd.to_datetime(df["createdAt"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    df = in_range(df)
    for c in ["likeCount", "retweetCount", "replyCount", "quoteCount", "viewCount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["engagement"] = df[["likeCount", "retweetCount", "replyCount", "quoteCount"]].fillna(0).sum(axis=1)
    df = df.drop_duplicates(["family", "id"])
    df, waterfall = twitter_filters(df)
    # then the shared cleaning: tokens, English, spam tags, exact + near duplicates, min words
    parts = []
    for fam, g in df.groupby("family"):
        c = clean_frame(g, min_words=TW_MIN_WORDS)
        parts.append(c)
    d = pd.concat(parts, ignore_index=True)
    waterfall.loc[len(waterfall)] = [f"cleaning: >= {TW_MIN_WORDS} words, spam tags, exact + near duplicates", len(d)]
    waterfall.to_csv(TABLES / "twitter_filter_waterfall.csv", index=False)
    print(waterfall.to_string(index=False))
    d["is_risk"] = d["clean"].str.contains(RISK_RE) & d["clean"].str.contains(AI_RE)
    s = add_class_schema(score_frame(d, models=("roberta", "finbert")), primary="rob")
    s.to_parquet(PROCESSED / "twitter_scored.parquet")
    print("saved", len(s))


def reddit(max_comments_per_sub_day: int = 300):
    from src.reddit import load_raw
    df = pd.DataFrame(load_raw())
    df["ts_utc"] = pd.to_datetime(pd.to_numeric(df["created_utc"]), unit="s", utc=True)
    df = in_range(df)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df["num_comments"] = pd.to_numeric(df.get("num_comments"), errors="coerce").fillna(0)
    df["engagement"] = df["score"].clip(lower=0) + df["num_comments"]
    df = df[~df["text"].isin(["[removed]", "[deleted]", ""])]
    df["text"] = df["text"].str.replace(r"\n?\[(removed|deleted)\]$", "", regex=True)
    fin = df["subreddit"].isin(REDDIT_FIN_SUBS)
    relevant = df["text"].str.contains(AI_RE) | df["text"].str.contains(CASHTAG_RE)
    df = df[~fin | relevant]
    # cap comments per sub-day so a few huge threads don't dominate (keeps run time sane)
    com = df[df["kind"] == "comments"].copy()
    com["day"] = com["ts_utc"].dt.tz_convert("America/New_York").dt.date
    # shuffle then head(): same random cap per sub-day, and pandas 3's groupby.apply
    # no longer passes the grouping columns through (it dropped `day` the first time)
    com = com.sample(frac=1, random_state=RANDOM_SEED).groupby(["subreddit", "day"]).head(
        max_comments_per_sub_day)
    df = pd.concat([df[df["kind"] == "posts"], com.drop(columns=["day"])], ignore_index=True)
    print("reddit rows before cleaning:", len(df), df["kind"].value_counts().to_dict())
    c = clean_frame(df)
    c["is_risk"] = c["clean"].str.contains(RISK_RE) & c["clean"].str.contains(AI_RE)
    c["sub_type"] = np.where(c["subreddit"].isin(REDDIT_FIN_SUBS), "finance", "ai")
    s = add_class_schema(score_frame(c, models=("roberta", "finbert"), max_len=256), primary="rob")
    s.to_parquet(PROCESSED / "reddit_scored.parquet")
    print("saved", len(s))


def news():
    df = build_news()
    df = in_range(df)
    df["clean"] = df["title"]
    print("news headlines:", len(df), df["group"].value_counts().to_dict())
    s = add_class_schema(score_frame(df, models=("roberta", "finbert"), max_len=64), primary="fin")
    s.to_parquet(PROCESSED / "news_scored.parquet")
    print("saved", len(s))


def arxiv():
    df = load_arxiv()
    df = in_range(df)
    df["clean"] = df["title"] + ". " + df["abstract"]
    print("arxiv:", len(df))
    s = add_class_schema(score_frame(df, models=("roberta", "finbert"), max_len=256), primary="fin")
    s.to_parquet(PROCESSED / "arxiv_scored.parquet")
    print("saved", len(s))


if __name__ == "__main__":
    {"twitter": twitter, "reddit": reddit, "news": news, "arxiv": arxiv}[sys.argv[1]]()

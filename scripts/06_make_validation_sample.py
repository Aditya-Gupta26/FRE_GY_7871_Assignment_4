"""Draw the blind validation sample and, once labelled, score every tool against it.

  python scripts/06_make_validation_sample.py make    # writes data/labels/validation_sample.csv
  python scripts/06_make_validation_sample.py score   # reads data/labels/validation_labels.csv

Sample design (stratified so rare classes and every source get enough rows):
- part 1: 40 Reddit posts/comments (balanced over predicted label) + 50 news headlines
  (spread over outlet groups)
- part 2: 60 tweets (20 per family GEN / RISK / FIN), balanced over predicted label
The label file is shuffled and the model scores are left out, so the labeller does
not see what any tool said.
"""
import _bootstrap  # noqa: F401
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src.config import LABELS, PROCESSED, RANDOM_SEED, TABLES
from src.sentiment import label_from

rng = np.random.default_rng(RANDOM_SEED)
SAMPLE = LABELS / "validation_sample.csv"            # part 1: Reddit + news
KEY = LABELS / "validation_key.parquet"              # hidden: ids + all tool scores
DONE = LABELS / "validation_labels.csv"              # the labelled copy of SAMPLE
SAMPLE_TW = LABELS / "validation_sample_twitter.csv"  # part 2: tweets
KEY_TW = LABELS / "validation_key_twitter.parquet"
DONE_TW = LABELS / "validation_labels_twitter.csv"


def _strat(df, by, n, seed=RANDOM_SEED):
    groups = df.groupby(by)
    per = max(1, n // groups.ngroups)
    out = groups.apply(lambda g: g.sample(min(len(g), per), random_state=seed), include_groups=False)
    out = out.reset_index(level=0).reset_index(drop=True)
    if len(out) < n:
        rest = df[~df.index.isin(out.index)]
        out = pd.concat([out, rest.sample(min(n - len(out), len(rest)), random_state=seed)])
    return out.head(n)


def _write(parts, sheet_path, key_path, prefix):
    allp = pd.concat(parts, ignore_index=True)
    allp = allp.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    allp["item_id"] = [f"{prefix}{i:03d}" for i in range(len(allp))]
    allp.to_parquet(key_path)
    sheet = allp[["item_id", "source", "clean"]].rename(columns={"clean": "text"})
    sheet["sentiment"] = ""      # neg / neu / pos
    sheet["is_ai_risk"] = ""     # 1 if about AI risk / safety / doom / slowdown, else 0
    sheet.to_csv(sheet_path, index=False)
    print(f"wrote {len(sheet)} rows to {sheet_path}")


def make():
    """Part 1 (Reddit + news, 90 rows) and part 2 (tweets, 60 rows) are separate sheets,
    so labelling part 1 can happen while the Twitter pull is still running."""
    if not SAMPLE.exists():
        parts = []
        rd = pd.read_parquet(PROCESSED / "reddit_scored.parquet")
        rd = rd[rd["n_words"].between(5, 80)]
        s = _strat(rd, "sentiment", 40)
        s["source"] = "reddit"
        parts.append(s)
        nw = pd.read_parquet(PROCESSED / "news_scored.parquet")
        nw = nw[nw["is_ai"] & nw["from_gen"]]
        s = _strat(nw, "group", 50)
        s["source"] = "news"
        parts.append(s)
        _write(parts, SAMPLE, KEY, "v")
    tw_path = PROCESSED / "twitter_scored.parquet"
    if tw_path.exists() and not SAMPLE_TW.exists():
        tw = pd.read_parquet(tw_path)
        tw = tw[tw["n_words"].between(5, 60)]
        parts = []
        for fam in ["GEN", "RISK", "FIN"]:
            s = _strat(tw[tw["family"] == fam], "sentiment", 20)
            s["source"] = f"twitter_{fam}"
            parts.append(s)
        _write(parts, SAMPLE_TW, KEY_TW, "t")
    elif not tw_path.exists():
        print("no twitter_scored.parquet yet: part 2 (tweets) will be written after the Twitter pull")


def score():
    labs, keys = [pd.read_csv(DONE)], [pd.read_parquet(KEY)]
    if DONE_TW.exists():
        labs.append(pd.read_csv(DONE_TW))
        keys.append(pd.read_parquet(KEY_TW))
    lab = pd.concat(labs, ignore_index=True)
    lab["sentiment"] = lab["sentiment"].astype(str).str.strip().str.lower()
    lab = lab[lab["sentiment"].isin(["neg", "neu", "pos"])]
    key = pd.concat(keys, ignore_index=True)
    d = key.merge(lab[["item_id", "sentiment", "is_ai_risk"]].rename(
        columns={"sentiment": "human", "is_ai_risk": "human_risk"}), on="item_id")
    rows = []
    for src_name, g in [("all", d)] + list(d.groupby(np.where(d["source"] == "news", "news", "social"))):
        for tool in ["rob", "fin", "vader", "lm", "hl"]:
            pred = label_from(g, tool)
            rows.append({"subset": src_name, "tool": tool, "N": len(g),
                         "accuracy": accuracy_score(g["human"], pred),
                         "macro_f1": f1_score(g["human"], pred, average="macro"),
                         "kappa_vs_human": cohen_kappa_score(g["human"], pred)})
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "validation_scores.csv", index=False)
    print(res.round(3).to_string(index=False))
    # tool-vs-tool agreement (the autism-slides comparison)
    tools = ["rob", "fin", "vader", "lm", "hl"]
    K = pd.DataFrame(index=tools, columns=tools, dtype=float)
    for a in tools:
        for b in tools:
            K.loc[a, b] = cohen_kappa_score(label_from(d, a), label_from(d, b))
    K.to_csv(TABLES / "tool_agreement_kappa.csv")
    print(K.round(2))
    # risk-frame keyword tagger vs human
    if "is_risk" in d:
        hr = pd.to_numeric(d["human_risk"], errors="coerce")
        ok = hr.notna()
        print("risk tagger accuracy:", accuracy_score(hr[ok].astype(int), d.loc[ok, "is_risk"].astype(int)),
              "F1:", f1_score(hr[ok].astype(int), d.loc[ok, "is_risk"].astype(int)))


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()

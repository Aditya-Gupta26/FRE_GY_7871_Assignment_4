# Assignment 4: AI Risk Sentiment, the AI Equity Basket, and Media Bias (Sept 8 to 29, 2026)

FRE-GY 7871 A · NLP and the Investment Process · Fall 2026

**Question in one line:** Evan Hubinger's post on X, at 9:27 PM ET on Sept 8, 2026, said the chance that AI kills all humans is ">10% within the next decade". After that post, did sentiment around AI move in a clear direction? Did those changes Granger-cause returns in a basket of AI stocks? And is the coverage of AI risk biased, the way Sacerdote, Sehgal and Cook (2020) found for COVID-19 news?

Repo: https://github.com/Aditya-Gupta26/FRE_GY_7871_Assignment_4

`REPORT.pdf` is the write-up (built from `REPORT.md`). `analysis.ipynb` reproduces every table and figure in it. `AI_USE.md` is the AI disclosure.

---

## What this gives you

Scripts that put the whole dataset on your disk, run in this order:

| Script | Output | Time |
|---|---|---|
| `00_get_lexicons.py` | Loughran-McDonald, Harvard GI and the Hu-Liu opinion lexicon, in `data/lexicons/` | ~15 s |
| `01_collect_twitter.py` | Tweets for three query families (GEN, RISK, FIN), lean bar-aligned sampling (30-min windows in trading hours, 2-h windows overnight and weekends), `min_faves:1`, cached per sub-window in `data/raw/twitter/` | ~3 min on a paid account (about 1 h 45 on the free tier, 1 call per 5 s) |
| `02_collect_reddit.py` | Posts and comments from 7 AI subreddits and 6 finance subreddits (Arctic Shift archive), in `data/raw/reddit/` | ~60 min |
| `03_collect_news.py rss` | Google News RSS for the 35 named outlets (AI and a non-AI placebo) plus daily edition queries (US, UK, IN, AU, CA) | ~30 min |
| `03_collect_news.py arxiv` | arXiv abstracts on AI safety, alignment and risk, the scientific benchmark | ~15 s |
| `03_collect_news.py gdelt` | GDELT DOC 2.0 article lists (optional, the API rate limits hard, see below) | 1 h+ |
| `04_get_market_data.py` | Daily and 60-minute prices for the 14-name AI basket, SPY, QQQ, SMH and VIX | ~20 s |
| `05_clean_and_score.py <source>` | Cleaned, deduplicated and scored text, one parquet per source in `data/processed/` | ~40 min total on Apple GPU |
| `06_make_validation_sample.py make/score` | The 150-row blind labelling sample (two sheets: 90 Reddit and news rows, 60 tweets), and tool accuracy against the labels | ~1 min |
| `07_run_analysis.py` | Every table in `outputs/tables/`, every figure in `outputs/figures/`, and `outputs/results.json` | ~5 min |
| `make_notebook.py` | Builds `analysis.ipynb` (code cells plus the markdown in `notebook_text.py`); then run it with `jupyter nbconvert --execute --inplace` | ~3 min |
| `build_report.py` | `REPORT.md` to `REPORT.pdf` (markdown-it, HTML, WeasyPrint) | ~5 s |
| `99_lint_prose.py` | Checks the prose files for em dashes (style rule) | ~1 s |

The modules they lean on:

| Module | What it does |
|---|---|
| `src/config.py` | Window, queries, tickers, sub-baskets, outlet groups, event calendar. Every date is in ET. |
| `src/twitter.py` | twitterapi.io client. Lean, bar-aligned sub-windows (30 min in trading hours, 2 h outside), one page each, global rate-limit pacing, credit check, everything cached. |
| `src/reddit.py` | Arctic Shift client. Full pulls for small subs, time-stratified pulls for the busy ones. |
| `src/news.py`, `src/news_corpus.py` | RSS, GDELT and arXiv collection, then one de-duplicated headline frame with Sacerdote-style outlet groups. |
| `src/text_clean.py` | URL and mention tokens, HTML unescape, repeated letters, English filter, spam filter, exact and near-duplicate removal (MinHash, Jaccard 0.8). |
| `src/sentiment.py` | VADER, Twitter-RoBERTa, FinBERT, Loughran-McDonald and Hu-Liu. Output uses the same columns as the class `tweet_data.csv`. |
| `src/index.py` | Posts to time series. Each post goes to the price bar whose information interval contains it, so there is no look-ahead. |
| `src/market.py` | Clean hourly bars, the equal-weighted and cap-weighted basket, market-model abnormal returns, and the Aloosh et al. efficiency tests. |
| `src/trend.py` | Trend, Mann-Kendall, pre/post, spike test and keyness. |
| `src/granger.py` | Granger in both directions, with HAC and permutation p-values, VAR and IRF, and FDR. |
| `src/bias.py` | The Sacerdote et al. style bias tests. |
| `src/pipeline.py` | Glue code shared by `07_run_analysis.py` and the notebook. |

### If a download will not run

- **Twitter:** you need your own key from twitterapi.io, in `.env` as `TWITTERAPI_IO_KEY=...`. X's own API only searches the last 7 days on pay-per-use, so a historical window has to come from a data provider. Run `python scripts/01_collect_twitter.py --probe` first; it pulls 6 sub-windows so you can check the setup and the cost. The full pull is 1,260 pages, so at most 25,200 tweets, which is at most about $3.80 at $0.15 per 1,000.
- **Tweet filters** (in `05_clean_and_score.py`, thresholds in `src/config.py`, waterfall saved to `outputs/tables/twitter_filter_waterfall.csv`):
  - views ≥ 100
  - author followers ≥ 10
  - account at least 30 days old
  - no giveaway or pump language
  - the AI term or cashtag in the text itself
  - at most 5 tweets per author per day
  - at least 5 words, plus the shared spam and near-duplicate cleaning
- **GDELT:** it rate limits quite aggressively, and it keeps extending the block if you retry during it. The collector goes one call every 10 s and backs off for 5 to 10 minutes on a 429. The analysis does not depend on it; Google News RSS is the main news source.
- **Arctic Shift:** "Timeout. Maybe slow down a bit" is normal on busy subs. The client backs off and resumes. Descending sort times out on big subs, which is why the sampler uses ascending order.
- Every collector is resumable. Just run it again and it skips what is already cached.

## Setup

```bash
cd Assignment_4
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "TWITTERAPI_IO_KEY=your_key_here" > .env
```

Then, in order:

```bash
python scripts/00_get_lexicons.py
python scripts/04_get_market_data.py
python scripts/01_collect_twitter.py --probe     # check setup and cost first
python scripts/01_collect_twitter.py
python scripts/02_collect_reddit.py
python scripts/03_collect_news.py rss
python scripts/03_collect_news.py arxiv
python scripts/05_clean_and_score.py twitter     # also: reddit, news, arxiv
python scripts/06_make_validation_sample.py make # then label both sheets, see data/labels/LABELLING_GUIDE.md
python scripts/06_make_validation_sample.py score
python scripts/07_run_analysis.py
python scripts/build_report.py
python scripts/99_lint_prose.py
```

The run used for the report pulled 24,881 tweets for $3.72, and 18,451 survive the filters.

Nothing under `data/` is committed except the validation labels (made blind by a Claude subagent, see AI_USE.md), because those are the one thing that cannot be pulled again.

## Reading the data you end up with

```python
import pandas as pd
tw = pd.read_parquet("data/processed/twitter_scored.parquet")
tw[["ts_utc", "family", "clean", "sentiment", "positiveScore", "negativeScore", "neutralScore"]]
```

Every scored file has the class `tweet_data.csv` columns (`sentiment`, `positiveScore`, `negativeScore`, `neutralScore`) for the primary model. That is Twitter-RoBERTa for tweets and Reddit (it was trained on tweets and is clearly best on our social labels) and FinBERT for news headlines and arXiv (built for news sentences, and tied with RoBERTa on our headline labels). The bias tests rescore every group with RoBERTa so that all groups share one classifier. REPORT Section 3 and notebook Section 2 explain this in full. It also has every tool's own columns: `rob_*` for RoBERTa, `fin_*` for FinBERT, `vader_*`, `lm_*` for Loughran-McDonald and `hl_*` for Hu-Liu.

## Submitting

1. **GitHub:** the repo has the notebook with its outputs saved, all code, and `AI_USE.md`. No data files, except the two validation label sheets.
2. **Brightspace:** `REPORT.pdf`, plus the repo URL.

The class readings in `Materials/` are not committed.

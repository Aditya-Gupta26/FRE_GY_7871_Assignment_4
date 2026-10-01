"""Central config: dates, queries, tickers, outlet groups and the event calendar.

Every timestamp in this project is stored in UTC and converted to US/Eastern
(ET) only when we align text to trading sessions. The window is fixed by the
deadline: Sept 30 market data did not exist yet at the time of writing.
"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
LABELS = DATA / "labels"
LEXICONS = DATA / "lexicons"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Baseline week (pre-event contrast) + main window. Main window starts on the
# day of the Coxon resignation / Hubinger post and ends at the Sept 29 close.
BASELINE_START = datetime(2026, 9, 1, 0, 0, tzinfo=ET)
WINDOW_START = datetime(2026, 9, 8, 0, 0, tzinfo=ET)
WINDOW_END = datetime(2026, 9, 29, 16, 0, tzinfo=ET)

# Estimation window for market-model betas (daily data only)
BETA_START = "2026-03-01"
BETA_END = "2026-08-31"

# ---------------------------------------------------------------- Twitter / X
# twitterapi.io advanced search, hourly-stratified "Latest" sampling.
TWITTER_ENDPOINT = "https://api.twitterapi.io/twitter/tweet/advanced_search"
TWITTER_QUERIES = {
    "GEN": ('(AI OR "artificial intelligence" OR ChatGPT OR Claude OR Anthropic '
            'OR OpenAI OR Gemini OR AGI) lang:en -filter:retweets'),
    "RISK": ('(AI OR Anthropic OR OpenAI OR AGI OR superintelligence) '
             '(extinction OR existential OR doom OR pdoom OR "p(doom)" OR "kill all humans" '
             'OR alignment OR "AI safety" OR "AI risk" OR slowdown OR pause) '
             'lang:en -filter:retweets'),
    "FIN": ('($NVDA OR $MSFT OR $GOOGL OR $META OR $AMZN OR $AMD OR $AVGO OR $TSM '
            'OR $ORCL OR $MU OR $ARM OR $PLTR OR $CRWV OR $SMCI) lang:en -filter:retweets'),
}
# Lean sampling (see src/twitter.py): Sept 8 to 29 only; during trading hours 2 pages per
# hourly bar per family, overnight and weekends 1 page per 2 hours (those hours all pool
# into the next 09:30 bar anyway). Server-side min_faves:1 so we mostly pay for tweets that
# have some engagement; the views >= 100 rule is applied locally after download.
TWITTER_START = WINDOW_START
TWITTER_QUERY_EXTRA = " min_faves:1"
TRADING_SUBWINDOW_MIN = 30
OFFHOURS_SUBWINDOW_MIN = 120

# Signal-to-noise filters applied to downloaded tweets (05_clean_and_score.py)
TW_MIN_VIEWS = 100              # user's rule: tweets nobody saw carry no market-relevant mood
TW_MIN_FOLLOWERS = 10           # near-empty accounts are mostly bots / throwaways
TW_MIN_ACCOUNT_AGE_DAYS = 30    # brand-new accounts around a viral event are often spam
TW_MAX_PER_AUTHOR_DAY = 5       # one hyperactive account cannot dominate a day's mood
TW_MIN_WORDS = 5                # "lol", "this", link-only posts have no scoreable sentiment
TW_PROMO_RE = (r"giveaway|airdrop|whitelist|presale|follow (?:and|&|\+) (?:rt|retweet)|\bdm me\b|"
               r"promo code|use my code|join (?:my|our) (?:discord|telegram)|signals? group|"
               r"100x|to the moon|free (?:trial|crypto)")
SEED_TWEETS = {
    "hubinger": "2097497037956891126",
}

# ---------------------------------------------------------------- Reddit
REDDIT_API = "https://arctic-shift.photon-reddit.com/api"
REDDIT_AI_SUBS = ["artificial", "singularity", "OpenAI", "ClaudeAI",
                  "ControlProblem", "Futurology", "ArtificialInteligence"]
REDDIT_FIN_SUBS = ["stocks", "wallstreetbets", "investing", "StockMarket",
                   "technology", "NVDA_Stock"]

# ---------------------------------------------------------------- keywords
AI_TERMS = ["ai", "a.i.", "artificial intelligence", "chatgpt", "claude", "anthropic",
            "openai", "gemini", "agi", "llm", "superintelligence", "machine learning",
            "nvidia", "deepmind", "copilot", "muse"]
CASHTAGS = ["$NVDA", "$MSFT", "$GOOGL", "$GOOG", "$META", "$AMZN", "$AMD", "$AVGO",
            "$TSM", "$ORCL", "$MU", "$ARM", "$PLTR", "$CRWV", "$SMCI"]

# ---------------------------------------------------------------- news
GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERIES = {
    "AI_GEN": '("artificial intelligence" OR Anthropic OR OpenAI OR ChatGPT OR Nvidia OR "AI model")',
    "AI_RISK": ('("artificial intelligence" OR Anthropic OR OpenAI OR AI) '
                '(extinction OR existential OR doom OR "AI safety" OR "kill all humans" OR superintelligence)'),
    "PLACEBO": '("Federal Reserve" OR inflation OR "housing market" OR "oil prices" OR "retail sales")',
}
GDELT_WINDOW_HOURS = 12       # artlist returns max 250 records, so we slice time
GDELT_SLEEP = 10.0            # GDELT asks for 1 req / 5 s; we got 429-banned at ~6 s, so go slower
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
ARXIV_API = "http://export.arxiv.org/api/query"

# Outlet groups, Sacerdote et al. (2020) style, plus finance and tech press.
# Matched on domain suffix. Anything else falls to US_GENERAL / INTL_GENERAL
# using GDELT's sourcecountry field.
OUTLET_GROUPS = {
    "US_MAJOR": ["nytimes.com", "washingtonpost.com", "usatoday.com", "latimes.com",
                 "nypost.com", "newsweek.com", "politico.com", "thehill.com", "cnn.com",
                 "foxnews.com", "nbcnews.com", "cbsnews.com", "abcnews.go.com",
                 "abcnews.com", "msnbc.com", "npr.org"],
    "INTL_MAJOR": ["theguardian.com", "bbc.com", "bbc.co.uk", "telegraph.co.uk",
                   "independent.co.uk", "thetimes.co.uk", "dailymail.co.uk", "mirror.co.uk",
                   "standard.co.uk", "indiatimes.com", "thehindu.com", "indianexpress.com",
                   "hindustantimes.com", "ndtv.com", "smh.com.au", "theage.com.au",
                   "theaustralian.com.au", "abc.net.au", "afr.com", "theglobeandmail.com",
                   "thestar.com", "ctvnews.ca", "cbc.ca"],
    "FIN_PRESS": ["reuters.com", "bloomberg.com", "cnbc.com", "wsj.com", "ft.com",
                  "marketwatch.com", "barrons.com", "finance.yahoo.com", "benzinga.com",
                  "investing.com", "fool.com", "seekingalpha.com", "businessinsider.com",
                  "forbes.com", "fortune.com", "foxbusiness.com", "thestreet.com",
                  "247wallst.com", "investopedia.com", "kiplinger.com"],
    "TECH_PRESS": ["techcrunch.com", "theverge.com", "wired.com", "arstechnica.com",
                   "engadget.com", "zdnet.com", "venturebeat.com", "theregister.com",
                   "404media.co", "technologyreview.com", "gizmodo.com", "tomshardware.com"],
}
# the Sacerdote ideology comparison uses these named outlets
IDEOLOGY_OUTLETS = ["foxnews.com", "nypost.com", "cnn.com", "msnbc.com", "nytimes.com",
                    "washingtonpost.com", "usatoday.com", "npr.org"]
# Google News site: queries to top up the named major outlets
RSS_SITES = OUTLET_GROUPS["US_MAJOR"] + ["theguardian.com", "bbc.com", "indiatimes.com",
                                         "thehindu.com", "indianexpress.com", "smh.com.au",
                                         "theglobeandmail.com", "reuters.com", "cnbc.com",
                                         "bloomberg.com"]

# ---------------------------------------------------------------- market
BASKET = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "AMD", "TSM",
          "ORCL", "MU", "ARM", "PLTR", "CRWV", "SMCI"]
SUB_BASKETS = {
    "compute": ["NVDA", "AMD", "AVGO", "TSM", "MU", "ARM", "SMCI"],
    "platforms": ["MSFT", "GOOGL", "AMZN", "META", "ORCL"],
    "ai_native": ["PLTR", "CRWV"],
    "anthropic_exposed": ["AMZN", "GOOGL", "MSFT", "NVDA"],
}
BENCHMARKS = ["SPY", "QQQ", "SMH", "^VIX"]

# ---------------------------------------------------------------- events (ET)
EVENTS = [
    {"ts": datetime(2026, 9, 8, 12, 0, tzinfo=ET), "label": "Coxon resigns",
     "kind": "risk", "short": ""},
    {"ts": datetime(2026, 9, 8, 21, 27, tzinfo=ET), "label": "Hubinger post",
     "kind": "risk", "short": "Coxon + Hubinger"},
    {"ts": datetime(2026, 9, 12, 12, 0, tzinfo=ET), "label": "Amodei 'pace the frontier' essay",
     "kind": "risk", "short": "Amodei essay"},
    {"ts": datetime(2026, 9, 16, 14, 0, tzinfo=ET), "label": "FOMC decision",
     "kind": "macro", "short": "FOMC"},
    {"ts": datetime(2026, 9, 21, 9, 30, tzinfo=ET), "label": "Meta Muse rally",
     "kind": "business", "short": "Muse rally"},
    {"ts": datetime(2026, 9, 28, 9, 30, tzinfo=ET), "label": "OpenAI agent escape / S-1 leak",
     "kind": "risk", "short": "Escape + S-1"},
]
FOMC_DAYS = ["2026-09-15", "2026-09-16"]

HF_MODELS = {
    "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "finbert": "ProsusAI/finbert",
}
RANDOM_SEED = 7871

# x-axis of Sacerdote et al. Figure 5: probability the outlet is a trusted source among
# conservatives (Pew 2019). Values are READ OFF their Figure 5 (approximate, +/- 0.01).
CONS_TRUST_PEW2019 = {"foxnews.com": 0.74, "cnn.com": 0.22, "nytimes.com": 0.15, "thehill.com": 0.19,
                      "politico.com": 0.12, "usatoday.com": 0.20, "cbsnews.com": 0.26, "npr.org": 0.24,
                      "nbcnews.com": 0.27, "abcnews.com": 0.28, "abcnews.go.com": 0.28,
                      "nypost.com": 0.12, "newsweek.com": 0.14}
OUTLET_SHORT = {"foxnews.com": "Fox", "cnn.com": "CNN", "nytimes.com": "NYTimes", "thehill.com": "TheHill",
                "politico.com": "Politico", "usatoday.com": "USAToday", "cbsnews.com": "CBS", "npr.org": "NPR",
                "nbcnews.com": "NBC", "abcnews.com": "ABC", "abcnews.go.com": "ABC", "nypost.com": "NYPost",
                "newsweek.com": "Newsweek"}

"""Markdown cells for analysis.ipynb (kept here so make_notebook.py stays short)."""

MD = {}
MD["intro"] = """# Assignment 4: AI risk sentiment, the AI basket, and media bias

This notebook reproduces every table and figure in `REPORT.pdf`. It does not pull any data itself. The collectors and scorers are in `scripts/01` to `05`, and they write the scored text to `data/processed/`. `scripts/07_run_analysis.py` writes the tables and figures. Here I load the same scored files and run the same `src/pipeline.py` functions again, so what you see below is exactly what went into the report.

The window is Sept 8 to 29, 2026, with Sept 1 to 7 kept only as a baseline week. Sept 30 is not in because the market data for it did not exist yet at the time of writing."""

MD["data"] = """## 1. Data

Four text sources:
- **Twitter:** three query families (GEN, RISK and FIN cashtags). RISK is pulled as its own sample because risk terms are only 2% to 5% of general AI tweets in a normal hour, so a general sample alone would give about one risk tweet per window, too few for an hourly series; having it separate also lets us test doom talk against the broad AI mood (GEN) and market talk (FIN). Sampling is lean and aligned to the price bars (30-minute windows in trading hours, 2-hour windows overnight and at weekends). As discussed in class, tweets are filtered to keep the signal-to-noise ratio under control: views ≥ 100, real accounts, no promo, the AI term in the text, at most 5 per author per day, and no duplicates. The waterfall below shows what each filter removes.
- **Reddit:** 7 AI subs and 6 finance subs.
- **News headlines:** Google News RSS, split into Sacerdote-style outlet groups, plus a non-AI placebo from the same outlets.
- **arXiv abstracts:** the scientific benchmark.

The counts below are after cleaning and dedup."""

MD["validation_md"] = """## 2. Do the sentiment tools agree with a human?

The autism tools slides and Carvalho and Plastino both show that sentiment tools disagree a lot, so I did not want to trust any one of them blindly. 150 items (60 tweets, 20 per family; 40 Reddit items; 50 headlines) were labelled blind by a separate Claude instance with no context and no access to any tool's output, as a stand-in for a human coder. Below is the accuracy and macro-F1 of each tool against those labels, and the tool-vs-tool kappa. The cell after the results says which tool is the primary one for each kind of text, and why. The others are used as robustness checks."""

MD["basket_md"] = """## 3. The AI basket

The basket has 14 names, equally weighted like the meme-stock indices in Aloosh, Choi and Ouzan, with cap weights as a check. The sub-baskets are compute, platforms, AI-native, and Anthropic's investors. The abnormal return is the basket minus beta times SPY, with beta from daily data March to August.

The last table is the Aloosh et al. weak-form checks on the hourly returns: Ljung-Box, variance ratio and runs. I checked this before any sentiment went in, because if returns could already be predicted from their own past, a Granger result would need extra care."""

MD["trend_md"] = """## 4. Q1: is there a clear trend in AI sentiment?

For each stream I run three things on the daily net sentiment inside Sept 8 to 29:
- an OLS time trend with Newey-West errors
- a Mann-Kendall test
- a baseline-week vs window comparison

The figure has sentiment on top, volume in the middle and the basket's cumulative return at the bottom. They are separate panels on purpose, so there are no two y-axes on one chart."""

MD["events_md"] = """### Around each shock

This is the change in mean daily net sentiment, comparing the 2 days before each event with the 2 days from it."""

MD["spike_md"] = """### Are busy hours more negative? (Thelwall et al.)

The Thelwall slides say that spikes in Twitter volume come with small increases in negativity. Below I regress the hourly negative share on log volume for each stream. After that, the keyness table lists the words that were over-represented on the three most negative days."""

MD["granger_md"] = """## 5. Q2: does sentiment Granger-cause AI basket returns?

First, stationarity. Returns are stationary. The sentiment levels are mostly stationary too, but the assignment asks about *changes* in sentiment, so the main tests use Δsentiment and levels are a robustness check.

The main table is hourly, with N of about 110 bars, lags 1 to 3, and both directions, as in Souza et al. The FOMC and overnight-bar dummies are exogenous controls. With this few observations I report:
- the F-test p-value
- a HAC-robust p-value
- a circular-shift permutation p-value
- FDR across the grid

The daily version (N = 16) is below it, laid out like the Smailović et al. lag table. It really is too short, so I only use it as a check."""

MD["var_md"] = """### Sign of the effect: VAR and impulse response

Granger alone does not say whether sentiment pushes returns up or down. So I fit a VAR with the lag *fixed at 3*, to match the Granger table; AIC and BIC choices are shown too, and BIC alone would pick 1 lag and miss the lag-3 dynamics. Then I look at the orthogonalised impulse response of the abnormal return to a one-sd shock in Δsentiment, with 90% Monte Carlo bands, and at the sum of the lag coefficients. The heatmap shows the same test run name by name, for the Reddit stream."""

MD["robust_md"] = """### Robustness grid

The grid covers:
- returns: abnormal EW, raw EW, abnormal CW, and the EW minus SPY spread
- sentiment measures: net, share negative, P(pos), Souza's S_R, VADER, and volume
- levels vs changes
- lags 1 to 3, in both directions

That is a lot of tests, so the question to ask is how many come out significant, compared with the number chance alone would give, and how many survive FDR."""

MD["bias_md"] = """## 6. Q3: is AI risk coverage biased, like COVID coverage was?

This is the Sacerdote, Sehgal and Cook (2020) design moved over to AI. The groups are US major, US general, international major, international general, financial press, tech press, science (arXiv), Twitter and Reddit. The main measure is the share of items where P(neg) > P(pos). This is the comparable version of their binary classifier, which forced every article to be positive or negative, and that is how they got "91% negative". The 3-class share and the standardised Hu-Liu dictionary share are also shown.

Every group is scored by the same classifier, and every news group is sampled with the same AI-general query (see PROGRESS issue 11). The placebo is the same outlets' non-AI headlines."""

MD["lpm_md"] = """### The Sacerdote Table 2 regression, and the difference-in-differences

The first model is a linear probability model of a headline being negative, on outlet-group dummies, day fixed effects and log length, with international major as the omitted group, like in the paper. The second pools AI and placebo headlines together. Its group times AI-story interaction gives the part of the gap that is specific to AI, and not just the outlet being negative about everything."""

MD["bias_more_md"] = """### Other checks from the paper

- **Tone vs an objective benchmark:** for Sacerdote this was case counts. Here, does headline negativity move with the AI basket?
- **Topic selection:** doom headlines vs benefit headlines, by group.
- **The Anthropic S-1 leak:** do the headlines lead with the risk section or with the numbers?
- **Base rate:** among headlines on the Coxon and Hubinger episode, how many give the actual probability vs "kill all humans" language? For context, the Grace et al. (2024) survey of 2,778 AI researchers has a median of 5% for extremely bad outcomes.
- **The ideology split:** Fox vs CNN and others."""

MD["demand_md"] = """### Demand side

Sacerdote et al. found that NYT readers pick the negative stories. The social-media version of that is to ask whether negative posts get more engagement, after controlling for followers (Twitter) or subreddit (Reddit) and for the day."""

MD["extra_md"] = """## 7. Extra questions that came up

**E1. Does the market price "doom" talk or "business" talk?** I compare the Granger results for the RISK family (doom talk) with the FIN cashtag family (market talk)."""

MD["e2_md"] = """**E2. Is attention more informative than tone?** Here the Granger test uses the volume of posts instead of their sentiment. It is Souza et al.'s V, and Thelwall's spikes."""

MD["e3_md"] = """**E3. Did the risk shocks show up in volatility and liquidity more than in returns, and were the compute names hit harder than the platforms?** This is the Aloosh et al. before and after comparison of hourly volatility and Amihud illiquidity across the shock windows, plus the Granger test run separately on each sub-basket."""

MD["wrap"] = """## Notes

- Everything above comes from `data/processed/*` and `outputs/results.json`, which were produced by the scripts in the order given in README.
- PROGRESS.md (kept locally) has the log of what broke during the build and how it was fixed. AI_USE.md summarises it."""

# ---------------------------------------------------------------- "Why this result" cells
MD["why_validation"] = """**Why this result.** The tools disagree because they look at different things:
- Lexicons (VADER, Loughran-McDonald, Hu-Liu) count words. They miss sarcasm, negation over long sentences, and context. For example, "kill" in "this update kills it" is positive.
- FinBERT was trained on financial news, so it reads "down" and "fall" as negative even in non-market text.
- Twitter-RoBERTa was trained on tweets, so it is usually the closest to a human on short informal posts.

This is exactly the point of the autism-awareness slides: the choice of tool changes the answer, so we pick the tool by how it does against the blind labels and by what text it was trained on, not by habit."""

MD["why_basket"] = """**Why this result.**
- The basket has a beta of about 2.4 because most of it is high-beta chip names (SMCI, ARM, MU, AMD). Their profits depend on how fast AI spending grows, so they swing more than the market.
- Ljung-Box and runs find no pattern in hourly returns, which is what you expect in liquid large-cap stocks: any easy pattern gets traded away fast.
- The variance-ratio test rejects even for SPY. It is most likely because the 09:30 bar carries the whole overnight move and is much more volatile than the intraday bars, so variance does not grow evenly with the horizon. That makes it a feature of how the bars are built, not something you could trade."""

MD["why_trend"] = """**Why this result.**
- **Why Twitter dipped on Sept 9 but news and Reddit hardly did:**
  - The post was made on X and got 43 million views, so Twitter's own mood dipped the next day. Sept 9 is among the 3 or 4 worst days for general and AI-risk tweets, and risk Twitter's worst days are full of "coxon", "resigned" and "researcher".
  - For news it was mostly a resignation story about one researcher saying something AI people already argue about.
  - The same day Meta launched its Muse agent, which pulled the other way: the basket was +2.9% above its beta-adjusted return that day.
- **Why the Amodei essay was the broad mover:** it was the CEO of a leading lab asking the whole industry to *slow down*, with the heads of OpenAI and xAI agreeing. That is actionable, because a slower race means less spending on chips, data centres and power. So it hit news, finance talk and chip stocks together on Sept 12 to 14.
- **Why sentiment recovered:** normal attention decay, plus a replacement story. The Muse rally on Sept 21 gave everyone something concrete and positive to talk about.
- **Why the sources differ in speed:** Reddit moves slowly because a community keeps arguing the same thread for days. News moves fast because there is a new set of headlines every day."""

MD["why_events"] = """**Why this result.** The event table is basically the trend plot cut into windows, and the explanation is the same. Events with consequences for spending and demand (Amodei's slowdown call, the Muse launch) move sentiment the most. Statements that only repeat a known worry (Hubinger) move it very little. The S-1 leak came right at the end of the window, so its full effect is probably not inside the data yet."""

MD["why_spike"] = """**Why this result.**
- **Spikes on news:** news volume spikes when something alarming happens, and most outlets then run versions of the same negative story, so busy hours are more negative. This is the Thelwall pattern, where spikes are typified by small increases in negativity.
- **No spike pattern on Reddit or general Twitter:** social volume also spikes for happy reasons, such as model releases and the Muse launch, so there volume and negativity are not tied together.
- **The keyness words show what the negative days were about, and they differ by source:**
  - Reddit and news: "slowdown", "slow", "safety", "Amodei", "CEO", "calls". This is the CEOs' slowdown call.
  - General Twitter: "slow", "killing", "weapons", "superintelligence".
  - AI-risk Twitter: "coxon", "resigned", "researcher". This is the resignation story itself."""

MD["why_granger"] = """**Why this result.**
- **Why news tone does not lead returns:**
  - By the time a headline exists, the event behind it is already public, and markets price public information within minutes.
  - Most AI headlines are not about the basket at all (products, schools, jobs, policy), so their tone has little to say about chip demand.
- **Why Reddit (and general Twitter) lead, and only into the next open:**
  - Because of the timing rule, overnight and weekend posts are in the *same* 09:30 bar as the overnight return, so they cannot create a lagged effect. What predicts the 09:30 return at lags 1 to 3 is the *previous afternoon's* mood (the 13:30 to 15:30 bars).
  - Two likely reasons. The people posting are largely retail traders, and retail order flow is heavy at the open. And news that breaks late in the day is discussed online before the close but priced only after hours and at the open.
  - When the 09:30 bar is removed as a target, the effect disappears, which fits this.
  - With only 15 overnight targets it could also be chance, and Granger cannot tell timing from causation.
- **Twitter tells a similar story with different names.**
  - General AI Twitter also leads only through the next open, and mostly in the AI-native names, which in practice is CRWV (p < 0.001 at every lag; PLTR shows nothing, p ≥ 0.15). CRWV is a hype-driven, retail-heavy stock whose price is close to "the AI narrative" itself.
  - Doom tweets are a constant background of the same argument, and cashtag Twitter is bullish every day (net about +0.32, no trend), so neither has timing information.
  - The robust link goes the other way: AI rallies are followed by *more negative* general AI tweets. A likely reason is that big rallies pull in "AI bubble" and sceptic commentary.
- **Why the chip names and not the platforms:** chips are high beta and popular with retail traders, and their earnings depend directly on AI compute spending. Microsoft, Google, Amazon and Meta have big non-AI businesses, and Anthropic's investors hold stakes that are small compared to their market caps."""

MD["why_var"] = """**Why this result.** The impulse response is negative in the same hour (about -0.09%) and positive for the next 3 hours (+0.08% to +0.10%). The same-hour part is just correlation inside one bar, so the order of the two variables decides it and it carries no timing information. The lagged part is the afternoon-to-next-open effect described above. The cumulative effect is small (about 0.1%), which fits a timing effect better than a big causal one."""

MD["why_robust"] = """**Why this result.** Running hundreds of tests always gives some false positives, about 5% at p < 0.05. Here there are more than 3 times that many, so something real is there. But almost all of it is Reddit on the all-bars sample, and it disappears intraday. So the robustness grid says the same thing as the main test: the signal is the overnight or weekend timing effect, not a general predictive power of sentiment."""

MD["why_bias"] = """**Why this result.**
- **Why US major outlets are the most negative:** the Sacerdote et al. explanation is about demand, and it fits here also. US national outlets compete hardest for clicks, and negative and frightening headlines get more clicks.
- **Why the doom angle in particular:** "kill all humans" is a very quotable line, while ">10% within a decade" with all its caveats is not.
- **Why international and financial outlets are less negative:** they cover AI more as adoption, investment, jobs and earnings, which is a less frightening angle.
- **Why science is the least negative:** abstracts use neutral technical words and usually describe a *method* or a *fix*. Even safety papers are written as progress, not as alarm.
- **Why the placebo matters:**
  - Every outlet group is less negative on AI than on its economic news. US major outlets are the exception that keeps most of its negativity on AI: their AI headlines are only about 4 pp less negative than their other news, against about 12 pp for international outlets.
  - That difference of about 8 pp is the part of the gap that is specific to AI, after removing each outlet's house tone."""

MD["why_lpm"] = """**Why this result.**
- The first regression shows the overall gap.
- The difference-in-differences shows how much of it is about AI specifically. The gap shrinks but stays significant, which tells us part of the "bias" is just the general house tone of US major outlets, and part is how they treat AI.
- When the sample is risk stories only, the gap disappears. Once a story is about risk, the facts are negative and every outlet sounds about the same. So the bias is mainly in *which* AI stories get picked (topic selection) and how general AI stories are framed, not in how risk stories are written."""

MD["why_bias_more"] = """**Why this result.**
- **Tone vs the market:**
  - Headline negativity does not follow the *raw* basket, but it does follow the *abnormal* (AI-specific) return. Headlines are more negative on days AI stocks lag the market.
  - This is different from COVID, where tone ignored case counts. The likely reason is that the same AI news drives both on the same day (Sept 14 is the clearest case), and AI news is itself market news in a way that case counts were not.
  - With 16 days and same-day data, we cannot say which one leads.
- **No Fox vs CNN split:** AI risk is not yet a left-vs-right issue. The left worries about jobs and bias, and the right worries about Big Tech power and China, so both sides have negative AI stories to run.
- **Only 8% of the Hubinger headlines gave the probability:** a number with caveats is hard to fit in a headline, and "kill all humans" is not.
- **The S-1 framing:** IPO filings always have risk sections, but this one used the phrase "existential risk to humanity", which is a much better headline than the revenue numbers."""

MD["why_demand"] = """**Why this result.** Sacerdote et al. found that NYT readers pick negative stories. On Reddit we see the opposite: positive posts get more upvotes than negative ones. Reddit voting rewards jokes, useful tips, excitement about new tools, and posts that agree with the community, and many AI subs are enthusiast communities. A news click is a different decision from a Reddit upvote, so the "demand for negativity" depends on the platform."""

MD["why_e1"] = """**Why this result.** People in the finance subs talk about their positions, earnings and trades, so their mood is close to actual buying and selling, and it can show up in prices within a few hours. Even so, this lead is weaker than the main Reddit result: its HAC p-values are 0.11 and 0.15, and it is not significant after FDR. People in the AI subs talk about the technology, philosophy and safety, and their link to prices is only the afternoon-to-next-open effect explained above."""

MD["why_e2"] = """**Why this result.**
- **Returns predict attention, not the other way round:** the coefficient is positive, so AI rallies bring people to post. That holds for Reddit and for all three Twitter families, intraday.
- **News attention spikes come before lower abnormal returns:** these spikes usually mean bad AI news (Thelwall's pattern), and the market keeps digesting it for a bit after the spike.
- **The first version of this test was wrong:** it used raw counts, and it showed fake effects only because the overnight bar always has far more posts than the intraday bars."""

MD["why_e3"] = """**Why this result.**
- **Why compute volatility jumped:** a call to slow down AI is a direct threat to future chip demand, but nobody knows how big the hit will be. That uncertainty shows up as higher volatility in the compute names, along with a higher VIX and lower liquidity.
- **Why the platforms did not:** for Microsoft, Google, Amazon and Meta a slowdown could even be good news, because it means less capex.
- **Why the Hubinger week was calm for chips but not for Anthropic's investors:** the post had no information about chip demand, so compute volatility did not move. But the Coxon and Hubinger story was about Anthropic itself, weeks before its IPO, so Amazon, Google, Microsoft and Nvidia, who own pieces of it, fell on Sept 8 and 9 (-0.95% and -1.37%) while the basket did not. They are mega-caps with plenty of their own news, so this is suggestive only, and sentiment does not Granger-cause their returns."""

# ---------------------------------------------------------------- which scorer, and how each figure is made
MD["scorer_md"] = """### Which score is used where, and why

Every index (net = P(pos) - P(neg), negative share, Souza's S_R, Smailović's P(pos)) is built from the class probabilities of one primary model per source. They sit in the class `tweet_data.csv` columns (`sentiment`, `positiveScore`, `negativeScore`, `neutralScore`), and every other tool's scores are kept next to them.

- **Tweets (all three families) and Reddit: Twitter-RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`, trained on about 124M tweets). Social posts are short, with slang, emojis, sarcasm, @-replies and half sentences, which is exactly what this model saw in training. It is also far ahead on the social labels: macro-F1 0.71 against 0.42 for FinBERT and 0.45 to 0.49 for the lexicons. The cashtag tweets also use it, because even when they talk about stocks they are still tweets in style.
- **News headlines (and the arXiv abstracts): FinBERT** (`ProsusAI/finbert`, tuned on sentences from financial news). Headlines are short edited news sentences, which is FinBERT's own register, and a lot of this story is about chips, capex and an IPO. On the 50 headline labels the two models are basically tied: RoBERTa has a slightly higher macro-F1 (0.58 vs 0.56), but FinBERT has higher accuracy (0.60 vs 0.56) and kappa (0.36 vs 0.30). With a tie on 50 items there was no reason to drop the model built for news, so FinBERT stays the news primary as planned.
- **Bias tests (Section 6): RoBERTa for every group, news also.** Sacerdote et al. score every outlet group with one classifier. If the news groups were scored by FinBERT and the social groups by RoBERTa, part of any "group gap" would just be the gap between the two models. So the bias frame rescores every group with RoBERTa, and FinBERT is run again as the robustness check (the US major x AI term is +8.0 pp with RoBERTa and +6.0 pp with FinBERT).
- **Lexicons.** Hu-Liu is the dictionary measure in Section 6, because it is Sacerdote's own measure. VADER's separate positive and negative strengths are used for Figure B, as the SentiStrength-style dual scale in the Thelwall slides. Loughran-McDonald and VADER compound are robustness checks only."""

MD["how_trend_figs"] = """**How these figures are made** (matplotlib, one y-axis per panel, every date in ET):
- **Figure 1** (in the style of Smailović et al. Figure 1). Top: for each calendar day, the number of general-AI tweets that Twitter-RoBERTa labels positive (bars up) and negative (bars down). Twitter starts on Sept 8, and the events are marked. Middle: the daily mean of P(pos) - P(neg) for each source (RoBERTa for Twitter and Reddit, FinBERT for news), as a trailing 3-day mean. Bottom: the cumulative sum of daily log returns x100 for the equal-weighted basket and SPY. They put price on a second axis; here it gets its own panel.
- **Figure E.** The same daily net sentiment per source (trailing 3-day mean solid, raw daily values faint), the estimated daily number of AI-risk tweets in the middle, and the cumulative basket and SPY returns at the bottom.
- **Figure B** (Thelwall slides). For general-AI tweets, by hour: the share of tweets the keyword tagger marks as AI risk (an AI term plus a risk term such as extinction, doom or slowdown), and the mean VADER positive and negative strength. All three are 6-hour rolling means, so single quiet hours do not dominate."""

MD["how_granger_figs"] = """**How these figures are made:**
- **Figure 2** (the Granger-causality graph of Souza et al. Figures 4 and 5). For each of six streams (Twitter general, AI-risk and cashtags; Reddit AI subs and finance subs; news headlines) it runs the OLS Granger F-test of the main table: Δ(net sentiment) and the basket's abnormal return AR_EW on hourly bars, lags 1 to 3, with FOMC and overnight dummies, in both directions. An arrow is drawn when p < 0.05 at any lag, labelled with the lag(s). Its colour is the sign of the summed lag coefficients (blue positive, red negative), and grey means returns lead sentiment. Panel (b) repeats it with the 09:30 bar not allowed as a target (the lags are built first, then those rows are dropped). Sentiment is from RoBERTa for Twitter and Reddit and from FinBERT for news.
- **Figure C.** Left: the orthogonalised impulse response of AR_EW to a 1-sd shock in Δ sentiment, from a VAR(3) with sentiment ordered first, with 90% Monte Carlo bands (500 draws). Right: the Granger p-value for general-AI Twitter sentiment leading each name's abnormal return (and each sub-basket's) at lags 1 to 3, drawn as -log10(p), so darker means more significant. The same table for Reddit, which carries the main lag-3 result, is printed in the cell above the figures."""

MD["how_bias_figs"] = """**How these figures are made:**
- **Figure 3** (a copy of Sacerdote et al. Figure 2). For each headline or post: Hu-Liu negative words divided by all words. This is z-scored once, with the mean and sd of the pooled sample (all AI items plus the non-AI placebo headlines), and then averaged by group. Dark bars are AI items and light bars the same outlets' non-AI headlines. It is a dictionary measure, so it does not depend on RoBERTa or FinBERT. As in their Figure 2, researchers are the least negative group: arXiv here (-0.17 sd), and the medical journals for COVID (about -1 sd).
- **Figure 4** (Sacerdote et al. Figures 1 and 5). (a) The daily share of AI headlines where RoBERTa gives P(neg) > P(pos), for US major vs international major outlets, with the running sum of daily AR_EW in its own panel below (their objective series was new COVID cases). (b) Each outlet's mean Hu-Liu z, as in Figure 3, against the share of conservatives who trust that outlet (Pew 2019, the x-values read off their Figure 5).
- **Figure F.** Both Sacerdote measures side by side for each group: (a) the share of items where RoBERTa gives P(neg) > P(pos), and (b) the Hu-Liu z of Figure 3."""

MD["how_vol_fig"] = """**How Figure A is made** (Aloosh et al. Figure 4). Realised volatility is the square root of the sum of squared hourly log returns over a rolling 14-bar window (two trading days), for the compute sub-basket, the platforms sub-basket and SPY, one panel each. The x-axis is bar order, trading hours only, so nights and weekends are not drawn as fake straight lines. The three shock windows are shaded."""

# ---------------------------------------------------------------- one prose result per question (same text as REPORT.md)
MD["result_q1"] = """**Result for Q1, is there a clear trend, in prose.** Yes, there is a clear trend, but it is a dip and then a recovery, not a straight line going down. Sentiment around AI turned more negative after Sept 8, first on Twitter where Hubinger posted (Sept 9 is among its 3 worst days), and then in every source after Amodei's slowdown essay on Sept 12, which is the real low point of the window: news fell from -0.08 to -0.20 in two days, and Reddit's most negative days were Sept 11, 13 and 14. AI stocks moved the same way, the basket lost about 5% between the Sept 8 and Sept 15 closes. From mid-September mood recovered slowly as the attention faded and the Muse launch gave a positive story, and the basket gained about 10% from Sept 16 to 22. Whether the mood actually leads the returns is Q2. *Against the COVID paper:* there, US major negativity stayed high (91% of stories) and did not follow the case counts at all (their Figure 1); AI mood here is much more event driven, falling with the risk shocks and coming back with good news."""

MD["result_q2"] = """**Result for Q2, does sentiment Granger-cause the basket, in prose.** Partly yes, and the sign is positive: when social-media mood about AI gets worse, the AI basket does worse after it, and when mood improves the basket does better. The one robust link is Reddit sentiment at a 3-hour lag, which survives the permutation test, the FDR correction and dropping any single day. General-Twitter mood also leads at lags 1 to 3, but it does not survive FDR. Both links mostly sit in the next morning's open, so it is the afternoon's mood predicting the opening gap, and they are carried by the chip and AI-native names, not the platforms. With only 15 overnight targets this is real in the sample but not proven. News tone, doom tweets and cashtag tweets do not Granger-cause the basket at all, and in the other direction AI rallies make general Twitter a bit more negative in the next hour. *Against the COVID paper:* it has no market test, so there is no direct equivalent; the nearest is that, like COVID tone which ignored the case counts, news tone here neither leads nor follows the basket hour by hour."""

MD["result_q3"] = """**Result for Q3, is there a measurable bias, in prose.** Yes, there is a measurable bias with the same shape as the COVID study, but it is much smaller, and it comes more from which stories get picked than from how they are written. US major outlets run 59.3% negative AI headlines against 46.6% for international major (+12.4 pp in the paper's regression, where COVID gave +25 pp), and after netting out how each outlet covers its other news, the gap is still +8.0 pp. They also pick 3.8 times more doom per benefit story, while inside risk stories every group is about equally negative. Unlike COVID news the tone does follow the objective benchmark (AI-specific returns), there is no Fox vs CNN split, and social media has its own bias: Reddit is the most negative group of all, and Twitter rewards negative posts with more engagement. Table 4 puts each result next to the COVID one.

**Table 4. The same tests on AI risk (this report) and on COVID-19 (Sacerdote et al. 2020)**

| Test | AI risk, Sept 8 to 29, 2026 | COVID-19, Jan to Jul 2020 |
|---|---|---|
| Share negative, US major / Intl major | 59.3% / 46.6% | 91% / 54% |
| US major vs Intl major (Table 2 LPM) | +12.4 pp | +25 pp |
| Hu-Liu z, US major / Intl major (Fig 3 vs their Fig 2) | +0.24 / +0.01 sd | +0.31 / -0.17 sd |
| Same outlets, non-topic news | US major only +0.02 sd | US major "only modestly" more negative |
| Science benchmark (Fig 3 vs their Fig 2) | least negative, -0.17 sd (arXiv) | least negative, about -1 sd (journals) |
| Topic selection, US major | 1.51 doom per benefit headline (Intl major 0.40) | 6 rising-case per falling-case story (5.3 while falling) |
| Tone vs objective data (Fig 4a vs their Fig 1) | follows AI-specific returns (corr -0.68) | unrelated to new cases |
| Audience politics (Fig 4b vs their Fig 5) | none (r = -0.06), Fox close to CNN | none, Fox about as negative as CNN |
| Demand for negativity | negative tweets +24% engagement; Reddit opposite | NYT most-read +1.5 sd on COVID |"""

MD["result_e1"] = """**Result, in prose.** market talk a little, doom talk only overnight. I split Reddit into the AI subs (doom talk) and the finance subs (stock talk). Finance-sub mood leads returns at lags 1 and 2 (p = 0.046 and 0.024; permutation 0.04 and 0.03) and a bit even intraday (lag 3, p = 0.036), while AI-sub mood only predicts the next open (lag 3 p = 0.004, intraday 0.71). On Twitter neither doom nor cashtag tweets lead (all p ≥ 0.18). Even the finance-sub link is weak (HAC p = 0.11 and 0.15, not significant after FDR), so the market prices neither kind of talk strongly. (The COVID paper has no market, attention or volatility tests, so E1 to E3 have no direct equivalent there.)"""

MD["result_e2"] = """**Result, in prose.** no, attention is not more informative than tone, it mostly follows prices. Using abnormal log volume (raw counts jump at every open and gave fake effects at first), social attention does not predict returns (Reddit p ≥ 0.64, Twitter p ≥ 0.22), but returns raise attention (intraday: Reddit lag 2, p = 0.037; all three Twitter families, p = 0.019 to 0.048). News attention spikes come before *lower* abnormal returns, but only on all bars (lag 1, p = 0.015; intraday 0.82), so it is again an overnight effect."""

MD["result_e3"] = """**Result, in prose.** yes, more in volatility and in specific names than in the basket's return. The slowdown call raised chip volatility by 64% while platform volatility fell, and the Hubinger week hit Anthropic's own investors, but the basket still ended the window up 4.1%."""

MD["s8_md"] = """## 8. So, should we be worried about AI, and should government step in?

As discussed in class, here are the responses to the following two questions: (1) should we be concerned with AI, and (2) does the government need to be involved in regulating AI?

*Readings used:* Grace et al. (2024) for the expert base rate; Sacerdote et al. (2020) for why biased coverage is a bad signal; Souza et al. and Aloosh et al. for what the market reaction does and does not tell us. Amodei's essay and the leaked Anthropic S-1 are the primary sources.

My data measures mood, markets and coverage, not how dangerous AI is, but it tells a lot about how good our signals are.

**(1) Should we be concerned with AI? Yes, as a serious minority risk.** The concern comes from insiders. In three weeks an Anthropic researcher quit, its alignment lead said more than 10%, its CEO asked the industry to slow down (Altman and Musk agreed), OpenAI reported an agent escape, and Anthropic's S-1 spends 80 of 261 pages on risk. Experts are split but not dismissive (Grace et al.: median 5%, and 38% to 51% put it at 10% or more). At the same time, **markets are not pricing it** (the basket rose 4.1% through repeated warnings, with only chip volatility and Anthropic's own investors flinching, and a market cannot price extinction anyway), and **media is not calibrated** (3.8 times more doom per benefit story; only 8% of the Hubinger headlines gave the actual probability). Sacerdote et al. ended by saying the CDC's warning against heavy COVID news consumption may be warranted; the same caution fits reading AI risk off US major headlines.

**(2) Does the government need to be involved? Yes, in some form.** When neither prices nor the press give a reliable signal, some independent oversight makes sense: audits, incident reporting, and S-1 style risk disclosure. Amodei's own ask (an antitrust safe harbor for shared safety standards, and international cooperation) is something only governments can give. How far to go beyond that, and whether rules would lock in the incumbents, is a values call that 16 days of data cannot answer."""

# Assignment 4 report

### AI Risk Sentiment, the AI Equity Basket, and Media Bias: Sept 8 to 29, 2026

FRE-GY 7871 A · NLP and the Investment Process

**Name:** Aditya Gupta
**NetID:** ag11023
**GitHub repo:** https://github.com/Aditya-Gupta26/FRE_GY_7871_Assignment_4

---

## 1. What I did

On Sept 8, Anthropic researcher Jacob Coxon resigned saying AI labs are "gambling with our lives". The same night at 9:27 PM ET, Evan Hubinger, who leads alignment work at Anthropic, posted that "we really do earnestly believe AI could kill all humans" and put it at ">10% within the next decade". Hubinger's post alone reached 43.1 million views, 59k likes and 10.5k quote-tweets. Four more risk shocks and one big positive shock followed (Table 1), so the window is basically a natural experiment on whether "AI doom" talk moves AI stocks.

From social media I collected 18,451 tweets (after filtering 24,881) and, in addition to Twitter, also 77,668 Reddit posts and comments; from the media, 38,346 news headlines and 2,250 arXiv abstracts for Sept 1 to 29, with the first week kept only as a baseline. I scored everything with five sentiment tools (checked against 150 blind labels), tested for a trend, ran Granger tests against a 14-stock AI basket, repeated the Sacerdote, Sehgal and Cook (2020) bias tests on AI coverage, and looked at three extra questions.

**Table 1. Event calendar (ET)**

| Date | Event | Type |
|---|---|---|
| Sept 8 | Coxon resigns; Hubinger post at 21:27; same day, Meta launches its Muse AI agent | risk (and business) |
| Sept 12 (Sat) | Amodei essay "We must pace the frontier"; Altman and Musk agree | risk, CEO level |
| Sept 14 | Chips sell off, platforms up (Fortune called it the "AI doomsday trade") | market |
| Sept 15 to 16 | FOMC meeting | confounder |
| Sept 21 | Muse tops app downloads, big chip rally (AMD above $1T) | business, positive |
| Sept 28 to 29 | OpenAI agent escapes its container; Anthropic S-1 leak ("existential risk", $2T valuation) | risk and business |

**Short answers.**
- **Trend:** sentiment turned more negative after Sept 8. On Twitter, where the post was made, Sept 9 is among the worst days; but the deepest dip across every source came after the CEOs' slowdown call (Sept 12 to 14). After that it recovered slowly.
- **Granger:** Reddit and general-Twitter mood lead the basket with a positive sign (mood gets worse, AI stocks do worse), mostly at the next morning's open; intraday it fades. News tone, doom tweets and cashtag tweets lead nothing.
- **Bias:** yes. After netting out how each outlet covers other news, US major outlets are 8 pp more negative on AI than international major outlets, and they pick 3.8 times more doom per benefit story. It is the COVID pattern, only much smaller, and unlike COVID the tone does follow AI-specific returns.
- **Concern and government (Section 8):** concern is reasonable, and with neither markets nor media giving a calibrated signal, some oversight is justified.

## 2. Data

- **Twitter/X:** X's own API only searches the last 7 days, so, as the Thelwall slides suggest, I bought historical search from a provider (twitterapi.io). There are three query families: GEN (general AI terms), RISK (AI plus extinction, doom, "kill all humans", slowdown and similar) and FIN (the 14 cashtags, Smailović's $AAPL idea). Sampling is lean and aligned to the price bars: 30-minute windows during trading hours and 2-hour windows overnight and at weekends, because those hours all pool into the next opening bar. That comes to at most 25,200 tweets for Sept 8 to 29.
- As discussed in class, raw tweets are very noisy, so to keep the signal-to-noise ratio under control (**tweet filtering**) a tweet is kept only if it has **at least 100 views** (a tweet nobody saw cannot carry market mood) and 1+ like, comes from an account with **10+ followers that is at least 30 days old** (bots and throwaway accounts pop up around viral events), has **no promo or pump language**, has **the AI term or cashtag in the text itself**, is one of **at most 5 tweets by that author that day** (so one account cannot drive a day's mood), and has 5+ words with no near duplicate. Out of 24,881 downloaded tweets, 18,451 survive (the views rule alone removes 4,401; the full waterfall is in the notebook).
- **Reddit, in addition to Twitter** (Arctic Shift archive): a second social platform, which also covers the Sept 1 to 7 baseline week that the Twitter pull does not. It has 7 AI subs (r/singularity, r/OpenAI, r/ClaudeAI, r/artificial and others) and 6 finance subs (r/wallstreetbets, r/stocks, r/investing and others). Finance items are kept only if they mention AI or a basket cashtag.
- **News** (Google News RSS), in the Sacerdote groups: 14 US major outlets (NYT, WaPo, CNN, Fox and others), international major (Guardian, BBC, Times of India, SMH and others), financial press, and US and international general outlets from the country editions.
  - Every outlet has a **non-AI placebo** (Fed, inflation, housing), which is the paper's non-COVID baseline.
  - arXiv AI-safety abstracts are the **science benchmark**.
- **AI basket:** 14 names (NVDA, MSFT, GOOGL, AMZN, META, AVGO, AMD, TSM, ORCL, MU, ARM, PLTR, CRWV, SMCI), equally weighted like the Aloosh et al. meme index.
  - Sub-baskets: compute, platforms, AI-native, and Anthropic's investors (AMZN, GOOGL, MSFT, NVDA).
  - Abnormal return = basket minus beta x SPY (beta 2.41, estimated March to August).
  - 16 trading days, which is 112 hourly bars. Every stream and the prices end at the Sept 29 close, because this report was due before the Sept 30 close.

## 3. Method

- **Cleaning**, as in the class papers: link and @user tokens, HTML unescape (the class `tweet_data.csv` still had `&amp;`), squeezed letter repeats, English only, a spam filter, and exact and near-duplicate removal (Jaccard ≥ 0.8, the Smailović idea).
- **Five sentiment tools**, because the autism-awareness slides and Carvalho and Plastino both show that tools disagree: VADER, Twitter-RoBERTa, FinBERT, Loughran-McDonald, and the Hu-Liu lexicon (Sacerdote's own dictionary).

- **Validation.** The 150 validation items (60 tweets, 40 Reddit, 50 headlines) were labelled blind by a separate Claude instance that had no context and was told not to open any tool output (it reported that it did not). It stands in for the human coders Sacerdote et al. used.
  - **Twitter-RoBERTa is clearly the best tool:** macro-F1 0.67 overall and 0.71 on social posts (κ = 0.48 overall, 0.57 on social). The other four sit at 0.46 to 0.52.
  - **The tools agree badly with each other** (pairwise κ 0.10 to 0.46).
  - The keyword risk tagger matches the AI-risk labels 86% of the time.
  - *Why this result:* lexicons only count words and FinBERT reads "down" or "fall" as negative even outside markets, while RoBERTa was trained on exactly this kind of text. As the autism-awareness slides say, the tool changes the answer.
- **Which score is used where, and why** (net and labels below come from that one model's class probabilities):
  - **Tweets and Reddit: Twitter-RoBERTa** (cardiffnlp, trained on about 124M tweets). Social posts are short, with slang, emojis, sarcasm and replies, which is what it was trained on, and it is far ahead on our social labels (F1 0.71 vs 0.42 for FinBERT). Cashtag tweets also use it, as they are still tweets in style.
  - **News headlines: FinBERT** (ProsusAI, tuned on financial news sentences). Headlines are edited news sentences, its own register, and much of this story is chips, capex and an IPO. On the headline labels the two are tied (F1 0.56 vs 0.58, but accuracy 0.60 vs 0.56 and κ 0.36 vs 0.30), so the news model stays.
  - **Bias tests (Section 6): RoBERTa for every group**, news also, because scoring groups with different models would mix the model gap into the group gap. FinBERT is the robustness check, and the dictionary measure is Hu-Liu, as in the paper. VADER and Loughran-McDonald (F1 0.45 to 0.52) are only robustness checks.
- **Timing.** Each post goes to the price bar whose interval [close(b-1), close(b)) contains it. So a Granger lag k ≥ 1 only uses posts written before the return it predicts starts. Overnight and weekend posts go to the next opening bar, which puts Hubinger's post in the Sept 9 open and Amodei's Saturday essay in the Sept 14 open.
- **Measures**, per bar, day and stream: net sentiment mean(P(pos) - P(neg)), negative share, Souza et al.'s S_R = (G - B)/(G + B), Smailović's P(pos), VADER, and volume.
- **Granger**, in both directions as in Souza et al.: lags 1 to 3 on hourly bars (N ≈ 110) and 1 to 2 on daily data (16 days, so N = 13 to 14 after lags), with FOMC and overnight dummies. Because N is small, the F-test p comes with a HAC p, a circular-shift permutation p and Benjamini-Hochberg FDR (q < 0.10). The sign comes from the sum of lag coefficients and a VAR(3) impulse response. To understand Granger causality and the Sacerdote et al. COVID paper, I also used this explainer artifact: https://claude.ai/artifact/MkTeUomB126RkTyrxHyAz9.
- **Bias**, following Sacerdote et al.: both of their measures (a classifier and a standardised Hu-Liu share), their Table 2 regression, tone against an objective benchmark, doom vs benefit topic counts, the Fox vs CNN split, and a demand-side test.

## 4. Q1: is there a clear trend in AI sentiment?

*Readings used:* Thelwall's Twitter-analysis slides (time series, spikes, co-word keyness), and the tool-comparison readings (autism-awareness slides, Carvalho and Plastino) for the scoring.

![Figure 1](outputs/figures/fig1_sentiment_trend.png)
*Figure 1. In the style of Smailović et al.'s Figure 1. Top: daily counts of general-AI tweets that RoBERTa labels positive (up) and negative (down); Twitter starts Sept 8, events are marked. Middle: daily mean of P(pos) - P(neg) per source (RoBERTa for Twitter and Reddit, FinBERT for news), trailing 3-day mean. Bottom: cumulative daily log return (%) of the basket and SPY, in its own panel where they used a second axis.*

Yes, there is a clear pattern, but it is **not a straight line going down**. The post soured Twitter, where it was made, but the broad dip in every source came only after the CEO-level slowdown call four days later.
- **Twitter** (from Sept 8, so no baseline):
  - Sept 9 is the 3rd most negative day of 22 for general AI tweets and the 4th for AI-risk tweets. Risk Twitter's worst days are full of "coxon", "resigned", "researcher" and "jacob".
  - After that, general and risk tweets improve steadily (+0.004 and +0.005 a day, p = 0.003 and 0.036), while cashtag tweets stay bullish and flat (net +0.32).
- **Reddit:** fell from -0.174 in the baseline week to -0.207 in the window (Welch p = 0.003; negative share 36.4% to 39.1%), then recovered steadily (+0.0036 a day, t = 3.47; Mann-Kendall τ = 0.51, p = 0.001).
- **News:** a V, with no monotonic trend (Mann-Kendall p = 0.43). It hardly moved around Sept 8 (+0.04), fell from -0.08 to -0.20 in the 2 days after Amodei's essay (the biggest *drop* of any event), and came back with the Muse rally (+0.15).
- **What drove the dips** (keyness, Thelwall's co-word idea): the worst days for Reddit (Sept 11, 13, 14) and news (Sept 12 to 14) are over-represented for "slowdown", "slow", "safety", "Amodei", "CEO" and "calls"; general Twitter's (Sept 9, 12, 13) for "slow", "killing", "weapons" and "superintelligence". The slowdown call hit news (-0.12), cashtag Twitter (-0.09), Reddit's finance subs (-0.08) and general Twitter (-0.07), while overall Reddit barely moved around either event (-0.01 each).
- Busier news hours are more negative (+0.021 per log-unit of volume, p = 0.016), as Thelwall found, but this does not hold for Reddit or general Twitter.

**Why this result.**
- **Sept 8:**
  - The post was huge on X itself (43 million views), so Twitter mood dipped the next day.
  - But it was one researcher saying something AI people already argue about, and news filed it as a resignation story.
  - The same day Meta launched Muse, which pulled the other way: the basket was +2.9% above its beta-adjusted return that day.
- **Sept 12:** Amodei's essay was a top-lab CEO asking the whole industry to slow down, and Altman and Musk agreed. That is actionable news, because slower scaling means less spending on chips and data centres, so it hit news, finance talk and chips all together.
- **The recovery** is basically attention decay, plus the Muse rally giving a positive story. Reddit moves slowly because threads run for days, while headlines turn over daily.
- **Busy news hours are negative** because a coverage spike usually means something alarming just happened.

## 5. Q2: does sentiment Granger-cause returns in the AI basket?

*Readings used:* Souza et al. (2015) for testing both directions and for S_R; Smailović et al. for the lag-by-lag table; Aloosh, Choi and Ouzan for the basket and the efficiency pre-checks.

**Pre-check (Aloosh et al.).** Hourly basket returns show no autocorrelation (Ljung-Box p = 0.85, runs p = 0.34). The variance ratio rejects, but it rejects for SPY too, so it most likely just reflects the bigger overnight bars.

**Table 2. Granger p-values, Δ(net sentiment) and AR_EW, hourly bars (permutation p in brackets). S→R: sentiment leads returns; R→S: returns lead sentiment.**

| Stream | Dir. | Lag 1 | Lag 2 | Lag 3 | Lag 3, intraday | Daily lag 1 |
|---|---|---|---|---|---|---|
| Reddit | S→R | 0.32 (0.29) | 0.31 (0.23) | **0.0003 (0.010)** + | 0.15 (0.10) | **0.027 (0.083)** + |
| Reddit | R→S | 0.67 | 0.27 | 0.30 | 0.23 | 0.20 |
| News | S→R | 0.72 (0.65) | 0.15 (0.23) | 0.17 (0.23) | 0.12 (0.16) | 0.12 (0.17) |
| News | R→S | 0.15 | 0.42 | 0.70 | 0.71 | 0.84 |
| Twitter general | S→R | **0.036 (0.041)** | **0.021 (0.031)** | **0.037 (0.051)** + | 0.46 (0.38) | 1.00 |
| Twitter general | R→S | **0.010 (0.010)** − | 0.087 | 0.13 | 0.071 (lag 1 **0.006**) | 0.78 |
| Twitter risk / $tags | S→R | 0.18 / 0.78 | 0.37 / 0.95 | 0.56 / 0.96 | 0.69 / 0.87 | 0.27 / 0.57 |

What the results say:
- **News tone does not Granger-cause the basket** in the main tests (every F and permutation p ≥ 0.12; only the compute sub-basket shows a weak link, p = 0.036 and 0.039), and returns do not lead news tone either. (Daily HAC p-values are smaller, but HAC over-rejects at N = 13 to 14, so the daily column uses F and permutation p.)
- **Reddit sentiment does, at 3 hours.**
  - HAC p = 0.004, permutation p = 0.010, FDR q = 0.008, and p ≤ 0.010 whichever day is dropped.
  - The sign is positive: the sum of lag coefficients is +6.8, and the VAR(3) impulse response is +0.08% to +0.10% at hours 1 to 3, after a -0.09% same-hour dip (notebook Figure C).
- **But it sits in the overnight bar and the chip names.**
  - With the 09:30 bar not allowed as a target, it weakens to F p = 0.15 and permutation p = 0.10 (HAC still gives 0.011), so the overnight bar carries most of it, though not all. The daily test points the same way (p = 0.027, permutation 0.083).
  - It is carried by ARM (0.002), MU (0.011), CRWV (0.020), AMD (0.023) and the compute sub-basket (0.001). Platforms (0.53) and Anthropic's investors (0.97) show nothing.
- **Twitter:**
  - General AI mood also leads at lags 1 to 3 with a positive sign (VAR p = 0.034, cumulative IRF +0.13%). But in the main table it does not survive FDR (q ≈ 0.21), and it vanishes intraday.
  - It is strongest in the AI-native names (PLTR and CRWV, lag 1 p < 0.001) and weaker in the platforms (p = 0.031).
  - The reverse link is the only Twitter link that survives intraday: abnormal returns lead general-Twitter mood with a *negative* sign (lag 1 p = 0.006, permutation 0.010), though it does not survive FDR.
- **Robustness grid** (1,800 tests): 203 have p < 0.05 where chance gives 90, and 21 survive FDR at q < 0.10, all on the all-bars sample. Of those, 14 are Reddit → returns and 5 are general Twitter → returns variants.
- So: **when social-media mood about AI gets worse in the afternoon, AI stocks open lower the next morning** (chips for Reddit, AI-native names for Twitter). It is real in this sample, but it rests on only 15 overnight targets.

**Why this result.**
- **News does not lead** because the market prices an event within minutes, and the pre-check finds no pattern left in hourly returns.
- **Why only the next open:** overnight and weekend posts sit in the *same* 09:30 bar as the gap return (timing rule), so this is not "weekend news reaching Reddit first". It is the *previous afternoon's* mood (the 13:30 to 15:30 bars) that predicts the open, most likely because the posters are retail traders who act at the open, and late-day news gets discussed online before the close but priced only after hours.
- **The chip and AI-native names** are high beta, retail-heavy and closest to the AI story. For Reddit the platforms show nothing (they have ads and cloud to fall back on); general Twitter reaches them only weakly (p = 0.031).
- **Doom and cashtag tweets** carry no timing: doom tweets are a constant background, and cashtag Twitter is bullish every day (net +0.32, no trend).
- **Rallies sour Twitter**, most likely because a big AI rally pulls in "bubble" and sceptic takes over the next hour.

![Figure 2](outputs/figures/fig2_granger.png)
*Figure 2. A Granger-causality graph drawn like Souza et al.'s Figures 4 and 5. For each stream it runs the Table 2 OLS F-test (Δ net sentiment and AR_EW, hourly, lags 1 to 3, FOMC and overnight dummies) both ways. An arrow means p < 0.05 at some lag (label = lags); blue and red give the sign of the summed lag coefficients, grey means returns lead sentiment. Reddit is split into AI and finance subs (E1). (a) All hourly bars; (b) intraday targets only.*

## 6. Q3: is there a measurable bias, like the COVID study found?

*Readings used:* Sacerdote, Sehgal and Cook (2020). Their whole design is copied here: outlet groups, the non-COVID placebo, the Table 2 regression, topic counts, and the ideology and demand tests. Grace et al. (2024) gives the expert base rate.

**Table 3. Linear probability models for a headline being negative, P(neg) > P(pos) (Sacerdote Table 2 style, day FE and log length, robust SE)**

| | (1) AI headlines | (2) AI + placebo, DiD |
|---|---|---|
| US major | **+0.124\*\*\*** (0.014) | +0.046\*\*\* (0.014) |
| US general | +0.069 (0.036) | +0.056\* (0.022) |
| Intl general | +0.039\* (0.018) | +0.097\*\*\* (0.014) |
| Financial press | -0.017 (0.016) | +0.029\* (0.014) |
| Tech press (n = 158) | +0.134\*\*\* (0.040) | not run (1 placebo headline) |
| AI story | | -0.117\*\*\* (0.014) |
| **US major × AI story** | | **+0.080\*\*\*** (0.020) |
| Omitted group (Intl major) mean / N | 0.466 / 8,244 | / 18,224 |

\* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001. With FinBERT instead of RoBERTa, the US major × AI story term is +0.060 (p = 0.002).

**Yes, there is a measurable bias with the COVID shape, but much smaller, and it is in story selection.**
- **Negativity:** 59.3% of US-major AI headlines lean negative, against 46.6% for international major and 44.5% for financial press. That is +12.4 pp in the paper's regression; for COVID they found +25 pp, from a 54% base. Tech press is about as negative (+13.4 pp).
- **Netting out each outlet's own tone (DiD):**
  - Everyone is less negative on AI than on their economic news (-11.7 pp for international major).
  - US major outlets keep most of their negativity on AI: their AI headlines are only 3.7 pp less negative than their non-AI ones.
  - So relative to international outlets, US-major AI coverage is **+8.0 pp more negative** (p < 0.001; +6.0 pp with FinBERT). It is the only significant positive interaction.
- **Dictionary measure (Figure 3):** US-major AI coverage is +0.24 sd against +0.02 for its non-AI news, the highest of any group, and the **science benchmark is the least negative** (-0.17 sd), as the paper found for journals.
- **Topic selection, not writing:** US major runs 1.51 doom headlines per benefit headline, against 0.40 for international major and 0.41 for financial press, about **3.8 times** more. But among risk stories alone, no group differs significantly from US major (all are roughly 71% to 86% negative).
- **Tone and the objective benchmark (Figure 4a):** tone does not follow the raw basket (corr -0.22, p = 0.37). Unlike COVID news, though, it does follow AI-specific performance: on days the basket lags SPY, headlines are more negative (corr -0.68, p < 0.001; still -0.55 without Sept 8).
- **Framing:** of 180 Coxon and Hubinger headlines, 8% give the probability and 34% use "kill all humans" language (the Grace et al. median is 5%). For the S-1 leak, 50% to 85% of IPO headlines lead with the risk section.
- **No ideology split** (Figure 4b): r = -0.06 with the conservative audience share, and Fox (+0.28 sd) sits close to CNN (+0.36).
- **Social media is biased toward negativity in its own way.** On the classifier, Reddit is the most negative group of all (65% vs 59% for US major), and AI-risk Twitter is more negative than general Twitter (41% vs 35% negative posts). On the dictionary both look mild, because slang escapes the Hu-Liu list. On demand, Twitter rewards negativity (negative tweets get about 24% more engagement than neutral ones, positive tweets 16%, the paper's "readers want negative" result), while Reddit rewards positivity (about +4% vs +11%).

**Why this result.**
- **US major outlets** compete hardest for clicks, and their readers click on negative stories (the paper's demand story). "Kill all humans" is very quotable; a 10% probability with caveats is not.
- **The others:** international and financial outlets cover AI as adoption, jobs and earnings, and science abstracts describe fixes in neutral words.
- **Inside risk stories** the facts are negative for everyone, so the only difference is which stories get run.
- **Tone follows AI-specific returns, not the raw basket**, most likely because both react to the same AI news on the same day. On Sept 14, for example, the slowdown call hit chips and headlines together. AI news also moves markets, which COVID case counts did not do in the same way.
- **No Fox vs CNN split:** AI risk is not a party issue yet.
- **Twitter vs Reddit:** Twitter ranking rewards replies and quote-tweets, which outrage drives, while Reddit upvotes reward jokes, tips and product excitement.

![Figure 3](outputs/figures/fig3_bias.png)
*Figure 3. A copy of Sacerdote et al.'s Figure 2. Per headline or post, Hu-Liu negative words divided by all words, z-scored once on the pooled AI and placebo sample, then averaged by group: dark bars for AI items, light bars for the same outlets' non-AI headlines. All news groups come from the same AI-general query. It is a dictionary measure, so it does not depend on RoBERTa or FinBERT.*

![Figure 4](outputs/figures/fig4_sacerdote_fig1_fig5.png)
*Figure 4. (a) Like Sacerdote et al.'s Figure 1: the daily share of AI headlines where RoBERTa gives P(neg) > P(pos), US vs international mainstream, against an objective series (theirs was new cases; here the running sum of daily AR_EW, in its own panel, not a second axis). (b) Like their Figure 5: each outlet's mean Hu-Liu z (as in Figure 3) vs the share of conservatives who trust it (Pew 2019, read off their Figure 5).*

## 7. Extra questions that came up

*Readings used:* Souza et al. for E1 (Twitter and news as different proxies) and for E2 (volume); Thelwall for E2 (spikes); Aloosh et al. for E3 (volatility and illiquidity around an event).

**E1. Does the market price doom talk or market talk?**
- I split Reddit into the AI subs (doom talk) and the finance subs (stock talk).
  - **Finance-sub sentiment** leads returns at lags 1 and 2 (p = 0.046 and 0.024; permutation 0.04 and 0.03), and even a bit intraday (lag 3, p = 0.036).
  - **AI-sub sentiment** works only overnight (lag 3 p = 0.004, intraday 0.71).
  - On Twitter, neither doom tweets nor cashtag tweets lead (all p ≥ 0.18), so "market talk leads" holds only for Reddit's finance subs. Even there it is weaker than the main result (HAC p = 0.11 and 0.15, and not significant after FDR).
- *Why this result:* finance subs talk about positions and trades, so their mood is close to actual buying and selling. The AI subs' only link to prices is the afternoon-to-next-open effect from Q2.

**E2. Is attention more informative than tone?**
- Using abnormal log volume (raw counts jump at every open and gave fake effects at first): social attention does not predict returns (Reddit p ≥ 0.64, Twitter p ≥ 0.22), but returns raise attention (intraday: Reddit lag 2, p = 0.037; all three Twitter families, p = 0.019 to 0.048). News attention spikes come before *lower* abnormal returns, but only on all bars (lag 1, p = 0.015; intraday 0.82), so it is again an overnight effect.
- *Why this result:* the coefficient is positive, so AI rallies bring people to post. A late-day news spike usually means bad AI news, and it gets priced at the next open.

**E3. Did the shocks show up more in volatility than in returns, and who got hit?** (Aloosh et al.; notebook Figure A)
- **The Amodei and "doomsday trade" window (Sept 14 to 16) is the real shock.**
  - Compute volatility rose from 0.81% to 1.33% an hour (+64%), while the platforms went from 0.58% to 0.48%.
  - VIX went from 15.0 to 17.2, and Amihud illiquidity was highest here (1.88, vs 1.78 in the baseline week and 1.09 in the calm week).
  - The Hubinger week barely moved compute volatility (0.82%), but **Anthropic's own investors were hit**: -0.95% and -1.37% on Sept 8 and 9 while the basket made +1.53% and -0.27%, their volatility rose about 19% (0.42% to 0.50%), and over the window they lost 0.85% while the basket gained 4.1%. Sentiment does not Granger-cause their returns, though (every stream and lag gives p ≥ 0.09).
- *Why this result:* a slowdown call hits expected chip demand by an unknown amount, and that uncertainty is volatility in chips; the platforms might even gain (less capex). The Coxon and Hubinger story was about Anthropic itself, weeks before its IPO, so the hit landed on the companies that own pieces of it. These are mega-caps with plenty of their own news, so this is suggestive only.

## 8. So, should we be worried about AI, and should government step in?

As discussed in class, here are the responses to the following two questions: (1) should we be concerned with AI, and (2) does the government need to be involved in regulating AI?

*Readings used:* Grace et al. (2024) for the expert base rate; Sacerdote et al. (2020) for why biased coverage is a bad signal; Souza et al. and Aloosh et al. for what the market reaction does and does not tell us. Amodei's essay and the leaked Anthropic S-1 are the primary sources.

My data measures mood, markets and coverage, not how dangerous AI actually is, so it cannot settle these alone. But it does tell a lot about how good our signals are.

**(1) Should we be concerned with AI? Yes, as a serious minority risk.**
- **The concern comes from insiders.** In three weeks an Anthropic researcher quit, its alignment lead said more than 10%, its CEO asked the industry to slow down (Altman and Musk agreed), OpenAI reported an agent escape, and Anthropic's S-1 spends 80 of 261 pages on risk. Experts are split but not dismissive (Grace et al.: median 5%, and 38% to 51% put it at 10% or more).
- **Markets are not pricing it** (the basket rose 4.1% through repeated warnings, with only chip volatility and Anthropic's own investors flinching, and a market cannot price extinction anyway), and **media is not calibrated** (3.8 times more doom per benefit story; only 8% of the Hubinger headlines gave the actual probability).

**(2) Does the government need to be involved? Yes, in some form.**
- When neither prices nor the press give a reliable signal, some independent oversight makes sense: audits, incident reporting, and S-1 style risk disclosure. Amodei's own ask (an antitrust safe harbor for shared safety standards, and international cooperation) is something only governments can give.
- How far to go beyond that, and whether rules would lock in the incumbents, is a values call that 16 days of data cannot answer.

## 9. Limitations

- **Twitter** comes from a third-party provider, sampled per window with `min_faves:1` and a views filter. So it represents tweets that got some attention, not all tweets, and it starts on Sept 8.
- **Small sample, and Granger is not causality.** There are only 16 trading days (15 overnight targets in the hourly test). Confounders: the FOMC gets a dummy, and Muse (launched Sept 8, rally on Sept 21) is both news and fundamentals.
- **Measurement:** news is headline-level only (GDELT kept my IP rate-limited), the general outlets' country comes from the Google News edition, the frames are keyword based, and the validation labels come from an LLM, not a human.
- **Sept 30 is missing**, because the report was due before the Sept 30 close.

## 10. Conclusion

AI sentiment soured after Sept 8, first on Twitter where the post was made, and then everywhere after the CEOs' slowdown call. Afternoon social-media mood leads AI returns at the next open, while news tone leads nothing. US major media show a real but small Sacerdote-type bias, which comes from story selection. Unlike COVID news, though, AI headline tone does move with AI-specific returns.

<p class="refs"><strong>References.</strong> Sacerdote, Sehgal and Cook (2020), NBER WP 28110 · Souza et al. (2015) · Smailović et al. · Aloosh, Choi and Ouzan (2023) · Thelwall slides; Thelwall et al. (2011) · Carvalho and Plastino (2020); Psomakelis et al. (2014); Bagheri; Jurafsky slides; Yener (2020) · Grace et al. (2024) · Amodei essay (Sept 12, 2026); Reuters on the Anthropic S-1 (Sept 29, 2026).</p>

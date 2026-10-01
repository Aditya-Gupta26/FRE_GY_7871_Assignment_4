# AI use disclosure

**Tools used:** Claude (Opus 5.5), through Claude Code

**What I used them for:** The implementation for this assignment was done by Claude Code in this repository, under my direction. That covers:
- the Twitter, Reddit and news collectors and the market data script
- cleaning and deduplication
- the five-tool sentiment scoring
- the bar-aligned sentiment indices
- the Granger code, the Sacerdote-style bias tests and the figures
- the notebook, and the first draft of REPORT.md

I directed it step by step, first with a written plan and then one piece at a time. After every step there was a check of the actual numbers before the next thing was built on top of it.

To understand Granger causality and the COVID reading (Sacerdote, Sehgal and Cook 2020), I used this explainer artifact made on claude.ai: https://claude.ai/artifact/MkTeUomB126RkTyrxHyAz9. The report says this in the Method section.

Section 8 of the report ("should we be worried about AI, and should government step in?") started from a question I put to Claude, asking it to answer only from our own findings. Its answer was then written into the report in the same voice as the rest. I checked that every number in it matches `outputs/results.json` or a cited source (Grace et al. 2024, the Amodei essay, the Reuters report on the S-1).

**What I decided and did myself:**
- **The framing.** Which of the class studies to follow for which part: Sacerdote et al. for bias, Souza et al. and Smailović et al. for Granger, Aloosh et al. for the basket and the hourly market tests.
- **Twitter.** That Twitter had to be in the data even though X's own API could not give the window, which meant buying it from twitterapi.io.
- **The basket.** The 14 names and the sub-baskets, including the idea of an "Anthropic investors" basket.
- **Validation.** The decision to validate the tools against 150 blind labels. On my instruction the labelling itself was done by a fresh Claude subagent with no context, which only saw the labelling guide and the text-only sheets and never the tools' scores. The report says this, so these are not human labels.
- **The tweet sample.** Cutting the Twitter pull to a lean design of about 25k tweets, after asking why 110k was needed, and the signal-to-noise rule that tweets with under 100 views are dropped. Claude added the other filters (followers, account age, promo language, a per-author cap) on top of that, and I kept them.
- **The write-up.** Going through every table, and what I actually conclude in the report.

**Anything the model got wrong that I had to correct:**

A number of errors came up during the build. They were caught by the checks run at every step: row counts per day, timestamps checked against known events, NaN counts, and round numbers that looked suspicious. All of them were fixed before any result was used. The most serious ones were in the market data, because they would have silently corrupted the Granger sample.

The worst one was that **the Sept 8 return was missing for every stock**, and Sept 8 is the event day itself.
- yfinance, when downloading many tickers together, takes the union of all their dates. ^VIX prints on exchange holidays, so a Labor Day (Sept 7) row got added where every stock was NaN.
- The log return on Sept 8 was then computed against that NaN row, so Sept 8 would have quietly dropped out of every daily test and every figure.
- The same thing happened on the day after Memorial Day, inside the beta estimation window.
- It came up only because the cap weights came back all NaN and I asked why.
- The fix is to keep only days where SPY actually traded.

The same union problem also hit the hourly data. VIX's extended-hours bars at 03:00, 04:00 and so on were mixed into the stock index, which gave 429 rows where there should be 20 days times 7 bars. A plain `pct_change` on that would have produced mostly empty hourly returns. The loader now keeps only the 7 regular-session bars, and asserts that count per day.

A smaller one in the same area:
- the cap-weighted basket gave a fake 0.0% return on the first bar
- pandas sums an all-NaN row to zero, not NaN
- the weights were also not being renormalised over the names that traded in each bar

On the data collection side:
- **arXiv.** The first arXiv pull returned exactly 2,000 abstracts. That is the page size, not the real count, and the true number is 2,270. Because results come back in date order, the missing part was everything after Sept 28, 08:16 UTC, which is exactly the S-1 leak and the OpenAI container story. It is paginated now.
- **Reddit.**
  - A full pull of the busy subs' comment history was going to take 2 to 3 hours, only to be sampled down later. It was switched to a time-stratified sample.
  - The first version of that used a descending sort, which Arctic Shift times out on for big subs, so it had to be ascending.
- **GDELT.** It kept my IP rate-limited no matter how slowly it was called, so the news corpus was re-planned around Google News RSS, including a same-outlet non-AI placebo for the bias test.

A few more came up in the analysis stage, and some of them would have changed the conclusions:
- **The bias comparison had a sampling bias of its own.**
  - The named outlets were pulled with the general AI query only, but the general-outlet groups also came from an "AI risk" query. That makes their doom share bigger by construction. International general showed 2.75 doom headlines per benefit headline, which was the pull, not the outlets.
  - Now every group comparison uses only items from the same query. After the fix the US-major result got stronger, not weaker.
- **Two different classifiers were being compared.** The news and arXiv rows had FinBERT labels while the social rows had RoBERTa labels, so part of any "group gap" would have been the gap between the two models. Every group is now scored by one tool, like in the paper.
- **The Hu-Liu z-scores were standardised separately** for the AI sample and the placebo sample, so the two could not be compared. They are now standardised once in the pooled sample, which is how Sacerdote et al. do it.
- **A fake "attention predicts intraday returns" result.** The volume series was raw post counts. The 09:30 bar holds about 17 hours of overnight posts, so volume jumps mechanically at every open. With log volume minus its time-of-day average, the effect went away (p went from about 0.02 to above 0.3).
- **The main Granger result needed a careful look.** Reddit sentiment at lag 3 came out very significant. A leave-one-day-out check showed no single day drives it. But when I asked it to drop the overnight bar as a target, the effect disappeared. So it is an overnight effect, and that is how it is reported.
  - The first way of doing that check was itself wrong. It filtered the rows before building the lags, which would have made the lag of a 10:30 bar the previous day's 15:30 bar. It was fixed so the lags are built first.
- **Smaller ones:**
  - VIX was NaN in every bar, because it prints on the hour and the stock bars start at :30.
  - The daily permutation test had no valid shifts at N = 14, so every p-value came out as 1.0.
  - BIC picked a 1-lag VAR, which hid the lag-3 dynamics, so the VAR is now fixed at 3 lags to match the Granger table.
  - pandas 3 silently drops grouping columns in `groupby.apply`.
  - One crash stayed hidden because a pipe through `grep` returned exit code 0.
  - In the first PDF build, every bullet list ran together into one paragraph.
  - The first Twitter probe failed one of its calls. It turned out the account was on the provider's free tier, which allows one request every 5 seconds, with no paid credit yet. The client now paces itself and checks the balance before a full run.
  - One "why this result" line said cashtag tweets "mostly describe moves that already happened", but our own test shows returns do not predict cashtag-tweet mood. It was caught on review and rewritten to match the data.

At the very end I asked for a full scrutiny check. A fresh Claude agent with no context audited every number in the report against the output files and checked the brief and our agreed design, and it found real mistakes that were then fixed:
- **The bias headline was mis-stated.** The report said US major outlets were "8 pp more negative on AI than their own baseline". The +8 pp is actually a difference-in-differences against international outlets; US major's AI headlines are in fact slightly *less* negative than its own non-AI news. The wording is now correct everywhere.
- **The keyness words were stale.** After Twitter was added, the main stream changed, and the quoted "most negative days" and words no longer matched the output. It also showed that Sept 9, the day after the Hubinger post, is among Twitter's worst days, so "the post moved very little" was too strong. Keyness is now reported per stream.
- **The Muse launch was on Sept 8**, the same day as the Hubinger post, not on Sept 21 (that was the rally). The event table and the Sept 8 explanation were fixed, and the launch is now named as a confound.
- **A result had been left out.** Headline tone does not follow the raw basket, but it does strongly follow the AI-specific (abnormal) return (corr -0.68). The report had only "tone ignores the market".
- **My explanation for the overnight effect contradicted my own timing rule.** Weekend posts sit in the same opening bar as the gap return, so they cannot create a lag. The explanation now uses the previous afternoon's mood, which is what the lags actually pick up.
- **Smaller ones:**
  - a wrong significance star
  - a few ranges and Ns
  - a table that one trend file overwrote
  - a leave-one-day-out claim with no code behind it, which is now in the pipeline
  - two notebook cells that still said "hand labels" and "BIC lag"
  - the Sept 30 gap was not explained

These got caught because nothing was accepted just because the code ran. Every step was checked against something I already knew: the timestamp of Hubinger's post, the number of trading days in the window, the press-reported size of the Sept 14 and Sept 21 moves, and so on.

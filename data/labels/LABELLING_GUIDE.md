# How to label the validation sheets

There are two sheets:
- `validation_sample.csv`: part 1, 90 rows of Reddit and news. **This one is ready now.**
- `validation_sample_twitter.csv`: part 2, 60 tweets. It appears after the Twitter pull.

1. Make a copy of each sheet, named `validation_labels.csv` and `validation_labels_twitter.csv`, in the same folder.
2. Fill in two columns for each row. It takes about 15 seconds a row, so roughly 25 minutes for part 1 and 15 for part 2.
3. Do not look up what any model said. The scores are kept out of these files on purpose.

## `sentiment`: the overall tone of the text itself
Write one of these values:
- `neg`: the text is negative or worried. Examples: fear, anger, criticism, bad news, "we are all doomed", a stock falling as bad news.
- `neu`: the text is factual or mixed, or has no clear tone. Examples: a plain news headline stating a fact, a question, a link share.
- `pos`: the text is positive. Examples: optimism, praise, excitement, good news, "bullish", a stock rising as good news.

Rate the tone of the post, not whether you agree with it. Sarcasm counts as what it actually means, so "great, AI will kill us all, love that for us" is `neg`.

## `is_ai_risk`: `1` or `0`
- `1`: the text is about AI risk, safety or danger. This covers extinction, doom, p(doom), alignment, AI going rogue, calls to slow down or pause AI, the Anthropic resignation or warnings, and the risk sections of the S-1.
- `0`: anything else. This includes AI business, stocks, products, jobs and general chat.

When you are done, tell Claude to run: `python scripts/06_make_validation_sample.py score`

"""Sentiment scoring with several tools. The class materials (autism tools slides,
Carvalho-Plastino, KDIR 2014) show these tools disagree, so we run all of them
and validate against blind labels instead of trusting any one.

Tools
- VADER: lexicon + rules built for social media (compound, pos, neg, neu).
- Twitter-RoBERTa (cardiffnlp, trained on ~124M tweets): 3-class probabilities.
- FinBERT (ProsusAI): 3-class probabilities, financial text.
- Loughran-McDonald: negative / positive word share (finance dictionary).
- Hu-Liu opinion lexicon: negative / positive word share. This is the exact
  dictionary measure that Sacerdote et al. (2020) use for COVID news.

Output columns follow the class tweet_data.csv schema for the primary model:
sentiment (NEGATIVE/NEUTRAL/POSITIVE), positiveScore, negativeScore, neutralScore.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import HF_MODELS, LEXICONS

TOKEN_RE = re.compile(r"[a-z][a-z'\-]*")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ------------------------------------------------------------------ lexicons
def _lm_sets() -> tuple[set, set]:
    lm = pd.read_csv(LEXICONS / "LoughranMcDonald_MasterDictionary.csv",
                     usecols=["Word", "Negative", "Positive"])
    neg = set(lm.loc[lm["Negative"] > 0, "Word"].str.lower())
    pos = set(lm.loc[lm["Positive"] > 0, "Word"].str.lower())
    return neg, pos


def _huliu_sets() -> tuple[set, set]:
    import nltk
    nltk.download("opinion_lexicon", quiet=True)
    from nltk.corpus import opinion_lexicon as ol
    return set(ol.negative()), set(ol.positive())


def lexicon_scores(texts: list[str]) -> pd.DataFrame:
    lm_neg, lm_pos = _lm_sets()
    hl_neg, hl_pos = _huliu_sets()
    rows = []
    for t in texts:
        toks = TOKEN_RE.findall((t or "").lower())
        n = max(len(toks), 1)
        rows.append({
            "n_tokens": len(toks),
            "lm_neg": sum(w in lm_neg for w in toks) / n,
            "lm_pos": sum(w in lm_pos for w in toks) / n,
            "hl_neg": sum(w in hl_neg for w in toks) / n,
            "hl_pos": sum(w in hl_pos for w in toks) / n,
        })
    return pd.DataFrame(rows)


def vader_scores(texts: list[str]) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    out = [sia.polarity_scores(t or "") for t in texts]
    return pd.DataFrame(out).rename(columns={"compound": "vader_compound", "pos": "vader_pos",
                                             "neg": "vader_neg", "neu": "vader_neu"})


# ------------------------------------------------------------------ transformers
class Transformer3:
    """Batched 3-class classifier; label order read from the model config."""

    def __init__(self, name: str, max_len: int = 128, batch: int = 64):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE).eval()
        self.max_len, self.batch = max_len, batch
        id2 = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.order = [id2[i] for i in range(len(id2))]

    @torch.no_grad()
    def __call__(self, texts: list[str], desc: str = "") -> pd.DataFrame:
        probs = []
        # sort by length so padding is small, then restore order
        idx = np.argsort([len(t or "") for t in texts])
        for i in tqdm(range(0, len(texts), self.batch), desc=desc, mininterval=20):
            chunk = [texts[j] or "" for j in idx[i:i + self.batch]]
            enc = self.tok(chunk, truncation=True, max_length=self.max_len, padding=True,
                           return_tensors="pt").to(DEVICE)
            p = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
            probs.append(p)
        P = np.vstack(probs) if probs else np.zeros((0, 3))
        out = np.empty_like(P)
        out[idx] = P
        return pd.DataFrame(out, columns=self.order)


def score_frame(df: pd.DataFrame, text_col: str = "clean", models=("roberta", "finbert"),
                max_len: int = 128) -> pd.DataFrame:
    """Adds every tool's columns. Prefixes: rob_, fin_, vader_, lm_, hl_."""
    texts = df[text_col].fillna("").tolist()
    parts = [df.reset_index(drop=True), vader_scores(texts), lexicon_scores(texts)]
    for m in models:
        clf = Transformer3(HF_MODELS[m], max_len=max_len)
        P = clf(texts, desc=m)
        pre = {"roberta": "rob", "finbert": "fin"}[m]
        parts.append(P.rename(columns={c: f"{pre}_{c[:3]}" for c in P.columns}))
        del clf
        if DEVICE == "mps":
            torch.mps.empty_cache()
    out = pd.concat(parts, axis=1)
    return out


def add_class_schema(df: pd.DataFrame, primary: str = "rob") -> pd.DataFrame:
    """Class tweet_data.csv columns from the chosen primary model."""
    d = df.copy()
    d["positiveScore"] = d[f"{primary}_pos"]
    d["negativeScore"] = d[f"{primary}_neg"]
    d["neutralScore"] = d[f"{primary}_neu"]
    lab = d[["negativeScore", "neutralScore", "positiveScore"]].to_numpy().argmax(1)
    d["sentiment"] = np.array(["NEGATIVE", "NEUTRAL", "POSITIVE"])[lab]
    d["net"] = d["positiveScore"] - d["negativeScore"]
    return d


def label_from(df: pd.DataFrame, tool: str) -> pd.Series:
    """Hard 3-class label for any tool, for validation against the blind labels."""
    if tool in ("rob", "fin"):
        P = df[[f"{tool}_neg", f"{tool}_neu", f"{tool}_pos"]].to_numpy()
        return pd.Series(np.array(["neg", "neu", "pos"])[P.argmax(1)], index=df.index)
    if tool == "vader":  # standard VADER cut-offs (Hutto & Gilbert 2014)
        c = df["vader_compound"]
        return pd.Series(np.where(c >= 0.05, "pos", np.where(c <= -0.05, "neg", "neu")),
                         index=df.index)
    if tool in ("lm", "hl"):
        diff = df[f"{tool}_pos"] - df[f"{tool}_neg"]
        return pd.Series(np.where(diff > 0, "pos", np.where(diff < 0, "neg", "neu")),
                         index=df.index)
    raise ValueError(tool)

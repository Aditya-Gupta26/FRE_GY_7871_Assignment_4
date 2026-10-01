"""Copy the Loughran-McDonald and Harvard GI lexicons from Assignment 1 (already
downloaded there) and fetch the Hu-Liu opinion lexicon through NLTK."""
import _bootstrap  # noqa: F401
import shutil
from pathlib import Path

import nltk

from src.config import LEXICONS

LEXICONS.mkdir(parents=True, exist_ok=True)
A1 = Path(__file__).resolve().parents[2] / "Assignment_1" / "FRE-GY-7871A-Assignment1" / "data" / "lexicons"
for f in ["LoughranMcDonald_MasterDictionary.csv", "inqtabs.txt"]:
    if not (LEXICONS / f).exists() and (A1 / f).exists():
        shutil.copy(A1 / f, LEXICONS / f)
    print(f, (LEXICONS / f).exists())
nltk.download("opinion_lexicon", quiet=True)
from nltk.corpus import opinion_lexicon as ol  # noqa: E402

print("Hu-Liu:", len(ol.negative()), "negative,", len(ol.positive()), "positive")

"""Collect Reddit posts + comments from AI and finance subreddits (Arctic Shift)."""
import _bootstrap  # noqa: F401
import json

from src.reddit import collect_all

summary = collect_all()
print(json.dumps(summary, indent=1))

"""Collect news: GDELT article lists + timelines, Google News RSS, arXiv, full text.

  python scripts/03_collect_news.py gdelt     # ~640 throttled calls, about an hour
  python scripts/03_collect_news.py rss       # Google News site: queries for named outlets
  python scripts/03_collect_news.py arxiv     # scientific benchmark
  python scripts/03_collect_news.py fulltext  # after gdelt, scrape bodies for grouped outlets
"""
import _bootstrap  # noqa: F401
import sys

from src import news

part = sys.argv[1] if len(sys.argv) > 1 else "gdelt"
if part == "gdelt":
    print("planned calls:", len(news.gdelt_plan()))
    print("new articles:", news.collect_gdelt_artlists())
    news.collect_gdelt_timelines()
elif part == "rss":
    print("rss items:", news.collect_rss("AI_GEN"))
    print("rss placebo items:", news.collect_rss("PLACEBO"))
    print("rss edition items:", news.collect_rss_editions())
elif part == "arxiv":
    print("arxiv entries:", news.collect_arxiv())
elif part == "fulltext":
    from src.news_corpus import urls_for_fulltext
    urls = urls_for_fulltext()
    print("urls to fetch:", len(urls))
    print("ok:", news.collect_fulltext(urls))

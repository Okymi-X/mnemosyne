"""
Mnemosyne -- Web Search Module
DuckDuckGo web + news search with ranked results.
"""

from __future__ import annotations

import logging
from typing import Callable

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False


# ---------------------------------------------------------------------------
# Generic helper -- DRY for web & news
# ---------------------------------------------------------------------------

def _search(
    query: str,
    method_name: str,
    max_results: int,
    formatter: Callable[[int, dict], str],
    label: str,
) -> str:
    """Run a DDGS method and format results."""
    if not HAS_DDG:
        return "Error: duckduckgo-search not installed. Run `pip install duckduckgo-search`."
    try:
        results: list[str] = []
        with DDGS() as ddgs:
            method = getattr(ddgs, method_name)
            for i, r in enumerate(method(query, max_results=max_results), 1):
                results.append(formatter(i, r))
        if not results:
            return f"No {label} found for: {query}"
        header = f"**{label.capitalize()} results for:** *{query}*\n\n"
        return header + "\n---\n".join(results)
    except Exception as e:
        logging.warning(f"{label.capitalize()} search failed: {e}")
        return f"Search failed: {e}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _fmt_web(i: int, r: dict) -> str:
    title = r.get("title", "No Title")
    href = r.get("href", "#")
    body = r.get("body", "").strip()[:300]
    return f"### [{i}] {title}\n**URL:** {href}\n{body}\n"


def _fmt_news(i: int, r: dict) -> str:
    title = r.get("title", "No Title")
    url = r.get("url", "#")
    body = r.get("body", "").strip()[:300]
    date = r.get("date", "")
    source = r.get("source", "")
    return f"### [{i}] {title}\n**Source:** {source} | **Date:** {date}\n**URL:** {url}\n{body}\n"


def search_web(query: str, max_results: int = 8) -> str:
    """Perform a web search and return formatted results."""
    return _search(query, "text", max_results, _fmt_web, "web")


def search_web_news(query: str, max_results: int = 5) -> str:
    """Search for recent news articles."""
    return _search(query, "news", max_results, _fmt_news, "news")

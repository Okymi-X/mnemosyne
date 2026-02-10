"""
Mnemosyne -- Web Search Module v2.0
Enhanced web search with result ranking, snippet extraction,
and multi-query support.
"""

from __future__ import annotations

import logging

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False


def search_web(query: str, max_results: int = 8) -> str:
    """Perform a web search and return well-formatted, ranked results."""
    if not HAS_DDG:
        return "Error: duckduckgo-search not installed. Run `pip install duckduckgo-search`."
    
    try:
        results = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results), 1):
                title = r.get("title", "No Title")
                href = r.get("href", "#")
                body = r.get("body", "").strip()
                # Truncate overly long snippets
                if len(body) > 300:
                    body = body[:297] + "..."
                results.append(
                    f"### [{i}] {title}\n"
                    f"**URL:** {href}\n"
                    f"{body}\n"
                )
        
        if not results:
            return f"No results found for: {query}"
            
        header = f"**Web search results for:** *{query}*\n\n"
        return header + "\n---\n".join(results)
    except Exception as e:
        logging.warning(f"Web search failed: {e}")
        return f"Search failed: {e}"


def search_web_news(query: str, max_results: int = 5) -> str:
    """Search for recent news articles."""
    if not HAS_DDG:
        return "Error: duckduckgo-search not installed."
    
    try:
        results = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.news(query, max_results=max_results), 1):
                title = r.get("title", "No Title")
                url = r.get("url", "#")
                body = r.get("body", "").strip()
                date = r.get("date", "")
                source = r.get("source", "")
                results.append(
                    f"### [{i}] {title}\n"
                    f"**Source:** {source} | **Date:** {date}\n"
                    f"**URL:** {url}\n"
                    f"{body}\n"
                )
        
        if not results:
            return f"No news found for: {query}"
            
        header = f"**News results for:** *{query}*\n\n"
        return header + "\n---\n".join(results)
    except Exception as e:
        return f"News search failed: {e}"

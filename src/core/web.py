"""
Mnemosyne -- Web Search Module
Uses duckduckgo-search to retrieve web results for the agent.
"""

from __future__ import annotations

import logging
from duckduckgo_search import DDGS

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

def search_web(query: str, max_results: int = 5) -> str:
    """Perform a web search and return formatted results."""
    if not HAS_DDG:
        return "Error: duckduckgo-search not installed. Run `pip install duckduckgo-search`."
    
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "No Title")
                href = r.get("href", "#")
                body = r.get("body", "")
                results.append(f"**{title}**\n{href}\n{body}\n")
        
        if not results:
            return "No results found."
            
        return "\n---\n".join(results)
    except Exception as e:
        return f"Search failed: {e}"

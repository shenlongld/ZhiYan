"""
Tavily 搜索封装（异步），强制 include_raw_content=True。
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from tavily import TavilyClient


class TavilySearcher:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("TAVILY_API_KEY 不能为空")
        self.client = TavilyClient(api_key=api_key)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_domains: List[str] | None = None,
        timeout_sec: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Tavily Python SDK 当前以同步调用为主，这里用 asyncio.to_thread 包装。
        """
        kwargs: Dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_raw_content": True,
            "include_answer": False,
        }
        if include_domains:
            kwargs["include_domains"] = include_domains
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.client.search, **kwargs),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Tavily search timeout after {timeout_sec}s, query={query}"
            ) from e

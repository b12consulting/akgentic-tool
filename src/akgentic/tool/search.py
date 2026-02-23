from typing import Any, Callable, Literal

from pydantic import BaseModel
from tavily import TavilyClient

from akgentic.tool.core import BaseTool


class WebSearch(BaseModel):
    max_results: int = 5
    search_depth: Literal["basic", "advanced"] | None = None


class WebFetch(BaseModel):
    timeout: float = 30
    extract_depth: Literal["basic", "advanced"] | None = None


class WebCrawl(BaseModel):
    timeout: float = 150
    max_depth: int | None = None
    max_breadth: int | None = None
    limit: int | None = None
    instructions: str | None = None
    extract_depth: Literal["basic", "advanced"] | None = None


class SearchTool(BaseTool):
    def get_tools(self) -> list[Callable]:

        tools = []

        for params in self.tool_card.params or []:
            if isinstance(params, WebSearch):
                tools.append(self.web_search_factory(params))
            if isinstance(params, WebFetch):
                tools.append(self.web_fetch_factory(params))
            if isinstance(params, WebCrawl):
                tools.append(self.web_crawl_factory(params))

        return tools

    def web_search_factory(self, params: WebSearch) -> Callable:

        def web_search_tool(
            query: str,
            max_results: int = params.max_results,
            search_depth: Literal["basic", "advanced"] | None = params.search_depth,
        ) -> Any:
            """Search the web for sources relevant to a natural-language query.

            Use this tool when knowledge is not available in local context
            (e.g., vector store) or when fresh/public web information is needed.

            Args:
                query: Natural-language search query to execute.
                max_results: Maximum number of results to return.
                    Tavily supports values in the range 0-20.
                search_depth: Search strategy balancing quality vs latency.
                    - ``basic``: balanced relevance/latency, lower credit cost.
                    - ``advanced``: higher relevance, potentially slower and more expensive.
                    If ``None``, Tavily default behavior is used.
            """

            tavily_client = TavilyClient()

            search_kwargs: dict[str, Any] = {}
            if search_depth is not None:
                search_kwargs["search_depth"] = search_depth

            return tavily_client.search(
                query,
                max_results=max_results,
                **search_kwargs,
            )

        return web_search_tool

    def web_fetch_factory(self, params: WebFetch) -> Callable:

        def web_fetch_tool(
            urls: list[str],
            timeout: float = params.timeout,
            extract_depth: Literal["basic", "advanced"] | None = params.extract_depth,
        ) -> Any:
            """Extract main content from one or more web pages.

            Use this tool when you already have URLs and need clean page content
            for reading, summarization, or grounding downstream reasoning.

            Args:
                urls: List of absolute URLs to extract content from.
                timeout: Maximum extraction time in seconds per request.
                    Tavily supports values roughly between 1 and 60 seconds.
                extract_depth: Extraction depth.
                    - ``basic``: faster and cheaper extraction.
                    - ``advanced``: richer extraction (e.g., better coverage),
                      potentially slower and more expensive.
                    If ``None``, Tavily default behavior is used.
            """

            tavily_client = TavilyClient()

            fetch_kwargs: dict[str, Any] = {}
            if extract_depth is not None:
                fetch_kwargs["extract_depth"] = extract_depth

            return tavily_client.extract(
                urls,
                timeout=timeout,
                **fetch_kwargs,
            )

        return web_fetch_tool

    def web_crawl_factory(self, params: WebCrawl) -> Callable:

        def web_crawl_tool(
            url: str,
            timeout: float = params.timeout,
            max_depth: int | None = params.max_depth,
            max_breadth: int | None = params.max_breadth,
            limit: int | None = params.limit,
            instructions: str | None = params.instructions,
            extract_depth: Literal["basic", "advanced"] | None = params.extract_depth,
        ) -> Any:
            """Crawl a website from a root URL and extract content from discovered pages.

            Use this tool when you need multi-page discovery from a site section
            (documentation, blog, knowledge base) rather than a single-page fetch.

            Args:
                url: Root URL to start crawling from.
                timeout: Maximum crawl time in seconds.
                    Tavily supports values between 10 and 150 seconds.
                max_depth: Maximum link depth from the root URL.
                    Tavily supports values between 1 and 5.
                max_breadth: Maximum number of links followed per level/page.
                    Tavily supports values between 1 and 500.
                limit: Total number of links/pages processed before stopping.
                    Must be >= 1.
                instructions: Optional natural-language guidance to bias crawl
                    and extraction toward specific topics or sections.
                extract_depth: Extraction depth applied to crawled pages.
                    - ``basic``: faster and cheaper.
                    - ``advanced``: richer extraction with higher latency/cost.
                    If ``None``, Tavily default behavior is used.
            """

            tavily_client = TavilyClient()

            crawl_kwargs: dict[str, Any] = {}
            if max_depth is not None:
                crawl_kwargs["max_depth"] = max_depth
            if max_breadth is not None:
                crawl_kwargs["max_breadth"] = max_breadth
            if limit is not None:
                crawl_kwargs["limit"] = limit
            if instructions is not None:
                crawl_kwargs["instructions"] = instructions
            if extract_depth is not None:
                crawl_kwargs["extract_depth"] = extract_depth

            return tavily_client.crawl(
                url,
                timeout=timeout,
                **crawl_kwargs,
            )

        return web_crawl_tool

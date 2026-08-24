from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, Literal

from pydantic import field_validator
from tavily import TavilyClient

from akgentic.tool.core import TOOL_CALL, BaseToolParam, ToolCard, _resolve

logger = logging.getLogger(__name__)


def _has_tavily_api_key() -> bool:
    """Return ``True`` when ``TAVILY_API_KEY`` is present and non-empty."""
    return bool(os.environ.get("TAVILY_API_KEY", "").strip())


def _check_tavily_api_key() -> bool:
    """Check whether ``TAVILY_API_KEY`` is configured and log a warning if not.

    Returns:
        ``True`` when the key is present and non-empty, ``False`` otherwise.
        A warning is logged when the key is missing.
    """
    if _has_tavily_api_key():
        return True
    logger.warning(
        "TAVILY_API_KEY is not set in the environment. "
        "Tavily search tools will be registered but non-functional until the key is configured."
    )
    return False


def _depth_default_for_none(value: Any) -> Any:
    """Read a ``None`` depth written by the older, optional schema as ``basic``.

    ``search_depth`` and ``extract_depth`` were once ``Literal[...] | None`` defaulting
    to ``None``, meaning "send nothing and let Tavily choose". Teams persisted under
    that schema carry a literal ``null`` in their ``team.yaml``. Narrowing the type
    without this validator makes every one of those files fail to deserialize — the
    team is not recoverable, it is reported as corrupted and dropped at startup.

    ``None`` meant "no explicit choice", and the tool's own choice is now ``basic``,
    so healing to ``basic`` preserves the stored intent rather than guessing at one.
    """
    return "basic" if value is None else value


class WebSearch(BaseToolParam):
    """Parameters for the web search capability."""

    max_results: int = 5
    search_depth: Literal["basic", "advanced"] = "basic"

    _heal_search_depth = field_validator("search_depth", mode="before")(_depth_default_for_none)


class WebFetch(BaseToolParam):
    """Parameters for the web fetch (extract) capability.

    The defaults are deliberately narrow. An unfiltered extract returns whole pages,
    and a handful of long pages will fill the model's context before it has read any
    of them. ``chunks_per_source`` is the ceiling on that: it caps how much of each
    page comes back once ``query`` has selected the relevant parts.
    """

    timeout: float = 30
    extract_depth: Literal["basic", "advanced"] = "basic"
    chunks_per_source: int = 3

    _heal_extract_depth = field_validator("extract_depth", mode="before")(_depth_default_for_none)


class WebCrawl(BaseToolParam):
    """Parameters for the web crawl capability.

    ``crawl_instructions`` is **not** a field here. It is a required argument of the
    crawl tool, for the same reason ``query`` is required on ``web_fetch``: an
    unguided crawl walks a site and returns whatever it finds, and there is no
    sensible value to configure in advance because the bias belongs to the question
    *this* crawl is meant to answer. A configured default would defeat the point —
    it would let the model omit the argument and fall back to crawling broadly.

    The argument is deliberately not spelled ``instructions``: that name is taken by
    :class:`~akgentic.tool.core.params.BaseToolParam`, where it appends to the tool
    docstring the model reads. One name spelled ``instructions`` once drove both
    effects at once, so a crawl could not be biased without also rewriting its
    description, nor described without also biasing it. The two remain separate, and
    now sit at different layers: ``instructions`` is configured and describes *when*
    to crawl, ``crawl_instructions`` is per-call and directs *what* to look for.

    To bias every crawl a team performs, set ``instructions`` — telling the model what
    to look for is the supported way to influence a required argument it chooses.
    """

    timeout: float = 150
    max_depth: int | None = None
    max_breadth: int | None = None
    limit: int | None = None
    extract_depth: Literal["basic", "advanced"] = "basic"

    _heal_extract_depth = field_validator("extract_depth", mode="before")(_depth_default_for_none)


class SearchTool(ToolCard):
    """Web search, fetch, and crawl capabilities via Tavily."""

    web_search: WebSearch | bool = True
    web_crawl: WebCrawl | bool = True
    web_fetch: WebFetch | bool = True

    def get_tools(self) -> list[Callable[..., Any]]:
        _check_tavily_api_key()
        tools: list[Callable[..., Any]] = []
        ws = _resolve(self.web_search, WebSearch)
        if ws and TOOL_CALL in ws.expose:
            tools.append(self._web_search_factory(ws))
        wc = _resolve(self.web_crawl, WebCrawl)
        if wc and TOOL_CALL in wc.expose:
            tools.append(self._web_crawl_factory(wc))
        wf = _resolve(self.web_fetch, WebFetch)
        if wf and TOOL_CALL in wf.expose:
            tools.append(self._web_fetch_factory(wf))
        return tools

    def _web_search_factory(self, params: WebSearch) -> Callable[..., Any]:
        def web_search_tool(
            query: str,
            max_results: int = params.max_results,
            search_depth: Literal["basic", "advanced"] = params.search_depth,
        ) -> Any:
            """Search the web for sources relevant to a natural-language query.

            Use this tool when knowledge is not available in local context
            (e.g., vector store) or when fresh/public web information is needed.

            Returns ranked titles, URLs and snippets — not full page content. Follow
            up with ``web_fetch`` on the URLs that look worth reading.

            Args:
                query: Natural-language search query to execute.
                max_results: Maximum number of results to return.
                    Tavily supports values in the range 0-20. More results is not
                    better: each one is a candidate the model has to triage.
                search_depth: Search strategy balancing quality vs latency.
                    - ``basic``: balanced relevance/latency, lower credit cost.
                    - ``advanced``: higher relevance, potentially slower and more expensive.
            """
            if not _has_tavily_api_key():
                return (
                    "Web search is unavailable: TAVILY_API_KEY is not set. "
                    "Ask the user to configure it and restart."
                )

            try:
                tavily_client = TavilyClient()

                return tavily_client.search(
                    query,
                    max_results=max_results,
                    search_depth=search_depth,
                )
            except Exception as exc:
                logger.warning("web_search failed: %s", exc)
                return (
                    f"Web search failed: {exc}. "
                    "The TAVILY_API_KEY may be invalid or the service may be "
                    "temporarily unavailable."
                )

        web_search_tool.__doc__ = params.format_docstring(web_search_tool.__doc__)
        return web_search_tool

    def _web_fetch_factory(self, params: WebFetch) -> Callable[..., Any]:
        def web_fetch_tool(
            urls: list[str],
            query: str,
            chunks_per_source: int = params.chunks_per_source,
            timeout: float = params.timeout,
            extract_depth: Literal["basic", "advanced"] = params.extract_depth,
        ) -> Any:
            """Extract the parts of one or more web pages that answer a question.

            Use this tool when you already have URLs and need page content for
            reading, summarization, or grounding downstream reasoning.

            The extraction is **relevance-filtered, not a full page dump**: ``query``
            selects which passages of each page come back, and ``chunks_per_source``
            caps how many. A long article costs a few paragraphs rather than its
            entire text, so several URLs can be read in one call.

            Args:
                urls: List of absolute URLs to extract content from.
                query: The question the extracted passages must answer. Required —
                    it is what selects the content. State what you actually need to
                    know; a vague query returns vague passages.
                chunks_per_source: Maximum number of relevant passages returned per
                    URL. Raise it only when one page must be covered exhaustively —
                    every extra passage enlarges the response you have to read.
                timeout: Maximum extraction time in seconds per request.
                    Tavily supports values roughly between 1 and 60 seconds.
                extract_depth: Extraction depth.
                    - ``basic``: faster and cheaper; enough for articles and docs.
                    - ``advanced``: richer extraction (tables, embedded content),
                      slower and more expensive.

            Content is returned as markdown.
            """
            if not _has_tavily_api_key():
                return (
                    "Web fetch is unavailable: TAVILY_API_KEY is not set. "
                    "Ask the user to configure it and restart."
                )

            try:
                tavily_client = TavilyClient()

                return tavily_client.extract(
                    urls,
                    format="markdown",
                    timeout=timeout,
                    query=query,
                    chunks_per_source=chunks_per_source,
                    extract_depth=extract_depth,
                )
            except Exception as exc:
                logger.warning("web_fetch failed: %s", exc)
                return (
                    f"Web fetch failed: {exc}. "
                    "The TAVILY_API_KEY may be invalid or the service may be "
                    "temporarily unavailable."
                )

        web_fetch_tool.__doc__ = params.format_docstring(web_fetch_tool.__doc__)
        return web_fetch_tool

    def _web_crawl_factory(self, params: WebCrawl) -> Callable[..., Any]:
        def web_crawl_tool(
            url: str,
            crawl_instructions: str,
            timeout: float = params.timeout,
            max_depth: int | None = params.max_depth,
            max_breadth: int | None = params.max_breadth,
            limit: int | None = params.limit,
            extract_depth: Literal["basic", "advanced"] = params.extract_depth,
        ) -> Any:
            """Crawl a website from a root URL, guided by what you are looking for.

            Use this tool when you need multi-page discovery across a site section
            (documentation, blog, knowledge base) rather than a single-page fetch.
            Prefer ``web_fetch`` when you already know which URLs matter — crawling
            is the expensive capability and returns many pages at once.

            The crawl is **directed, not exhaustive**: ``crawl_instructions`` steers
            which links are followed and what is extracted from them. Without it a
            crawl walks the site broadly and returns whatever it encounters, which is
            both slow and far more content than can usefully be read.

            Args:
                url: Root URL to start crawling from.
                crawl_instructions: What this crawl is looking for, in natural
                    language — e.g. "the authentication section of the API
                    reference". Required: it is what keeps the crawl focused. Be
                    specific; broad guidance produces a broad crawl.
                timeout: Maximum crawl time in seconds.
                    Tavily supports values between 10 and 150 seconds.
                max_depth: Maximum link depth from the root URL.
                    Tavily supports values between 1 and 5.
                max_breadth: Maximum number of links followed per level/page.
                    Tavily supports values between 1 and 500.
                limit: Total number of links/pages processed before stopping.
                    Must be >= 1.
                extract_depth: Extraction depth applied to crawled pages.
                    - ``basic``: faster and cheaper.
                    - ``advanced``: richer extraction with higher latency/cost.
            """
            if not _has_tavily_api_key():
                return (
                    "Web crawl is unavailable: TAVILY_API_KEY is not set. "
                    "Ask the user to configure it and restart."
                )

            try:
                tavily_client = TavilyClient()

                crawl_kwargs: dict[str, Any] = {}
                if max_depth is not None:
                    crawl_kwargs["max_depth"] = max_depth
                if max_breadth is not None:
                    crawl_kwargs["max_breadth"] = max_breadth
                if limit is not None:
                    crawl_kwargs["limit"] = limit

                return tavily_client.crawl(
                    url,
                    timeout=timeout,
                    format="markdown",
                    # Tavily's own kwarg keeps the name `instructions`; only the tool
                    # argument is renamed, to free `instructions` for the docstring.
                    instructions=crawl_instructions,
                    extract_depth=extract_depth,
                    **crawl_kwargs,
                )
            except Exception as exc:
                logger.warning("web_crawl failed: %s", exc)
                return (
                    f"Web crawl failed: {exc}. "
                    "The TAVILY_API_KEY may be invalid or the service may be "
                    "temporarily unavailable."
                )

        web_crawl_tool.__doc__ = params.format_docstring(web_crawl_tool.__doc__)
        return web_crawl_tool

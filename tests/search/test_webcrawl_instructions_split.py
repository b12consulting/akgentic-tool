"""``WebCrawl`` separates the crawl bias from the tool description.

``WebCrawl`` used to re-declare ``instructions``, a field it already inherits from
``BaseToolParam``. One field then drove two unrelated effects: the value was passed to
Tavily's ``crawl(instructions=...)`` *and* appended to the docstring the model reads.
Neither could be asked for alone.

The two are now separate fields, and the tests below pin each direction independently —
each asserts both what its field does and what it does **not** do, because the defect was
precisely that one field did both.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from akgentic.tool.search.search import SearchTool, WebCrawl


class _MockTavilyClient:
    """Fake TavilyClient recording the kwargs ``crawl`` was called with."""

    def __init__(self) -> None:
        self.crawl = MagicMock(return_value={"results": []})


def _crawl_tool(tools: list[Any]) -> Any:
    """Return the ``web_crawl_tool`` callable from a built tool list."""
    return next(tool for tool in tools if tool.__name__ == "web_crawl_tool")


@pytest.fixture
def mock_tavily(monkeypatch: pytest.MonkeyPatch) -> _MockTavilyClient:
    """Install a recording TavilyClient and a key, and return the client."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    client = _MockTavilyClient()
    monkeypatch.setattr("akgentic.tool.search.search.TavilyClient", lambda: client)
    return client


# ---------------------------------------------------------------------------
# crawl_instructions reaches Tavily, and only Tavily
# ---------------------------------------------------------------------------


def test_crawl_instructions_is_forwarded_to_tavily(mock_tavily: _MockTavilyClient) -> None:
    """The field reaches Tavily under its own kwarg name, ``instructions``."""
    tools = SearchTool(
        web_crawl=WebCrawl(crawl_instructions="focus on the API reference")
    ).get_tools()

    _crawl_tool(tools)(url="http://example.com")

    _, kwargs = mock_tavily.crawl.call_args
    assert kwargs["instructions"] == "focus on the API reference"


def test_crawl_instructions_does_not_reach_the_docstring(mock_tavily: _MockTavilyClient) -> None:
    """Biasing the crawl must not rewrite the description the model reads."""
    tools = SearchTool(
        web_crawl=WebCrawl(crawl_instructions="focus on the API reference")
    ).get_tools()

    doc = _crawl_tool(tools).__doc__ or ""
    assert "focus on the API reference" not in doc
    assert "Additional Instructions:" not in doc


def test_crawl_instructions_defaults_to_omitting_the_kwarg(
    mock_tavily: _MockTavilyClient,
) -> None:
    """Unset means Tavily is not sent the kwarg at all, so its own default applies."""
    tools = SearchTool(web_crawl=WebCrawl()).get_tools()

    _crawl_tool(tools)(url="http://example.com")

    _, kwargs = mock_tavily.crawl.call_args
    assert "instructions" not in kwargs


def test_crawl_instructions_is_overridable_per_call(mock_tavily: _MockTavilyClient) -> None:
    """The configured value is a default on the tool signature, not a fixed setting."""
    tools = SearchTool(web_crawl=WebCrawl(crawl_instructions="configured")).get_tools()

    _crawl_tool(tools)(url="http://example.com", crawl_instructions="per call")

    _, kwargs = mock_tavily.crawl.call_args
    assert kwargs["instructions"] == "per call"


# ---------------------------------------------------------------------------
# instructions reaches the docstring, and only the docstring
# ---------------------------------------------------------------------------


def test_instructions_is_appended_to_the_docstring(mock_tavily: _MockTavilyClient) -> None:
    """The inherited field behaves on ``WebCrawl`` as on every other capability param."""
    tools = SearchTool(web_crawl=WebCrawl(instructions="Only crawl public docs.")).get_tools()

    doc = _crawl_tool(tools).__doc__ or ""
    assert "Additional Instructions:" in doc
    assert "Only crawl public docs." in doc


def test_instructions_does_not_reach_tavily(mock_tavily: _MockTavilyClient) -> None:
    """The regression itself: describing the tool must not bias the crawl.

    Before the split this assertion failed — ``instructions`` was forwarded as
    Tavily's crawl guidance as well as appended to the description.
    """
    tools = SearchTool(web_crawl=WebCrawl(instructions="Only crawl public docs.")).get_tools()

    _crawl_tool(tools)(url="http://example.com")

    _, kwargs = mock_tavily.crawl.call_args
    assert "instructions" not in kwargs


def test_both_fields_can_be_set_independently(mock_tavily: _MockTavilyClient) -> None:
    """The point of the split: two intents, two fields, no interference."""
    tools = SearchTool(
        web_crawl=WebCrawl(
            instructions="Only crawl public docs.",
            crawl_instructions="focus on the API reference",
        )
    ).get_tools()

    tool = _crawl_tool(tools)
    tool(url="http://example.com")

    _, kwargs = mock_tavily.crawl.call_args
    assert kwargs["instructions"] == "focus on the API reference"

    doc = tool.__doc__ or ""
    assert "Only crawl public docs." in doc
    assert "focus on the API reference" not in doc


# ---------------------------------------------------------------------------
# The field is gone under its old name
# ---------------------------------------------------------------------------


def test_webcrawl_no_longer_redeclares_instructions() -> None:
    """``instructions`` must be the inherited field, not a shadowing re-declaration.

    Asserted on the owning class rather than on behaviour, because a re-declaration
    that happened to keep the same annotation would pass every test above while
    reintroducing exactly the shape this change removes.
    """
    assert "instructions" not in WebCrawl.__annotations__
    assert "crawl_instructions" in WebCrawl.__annotations__


def test_crawl_signature_names_the_argument_for_what_it_is() -> None:
    """The model's schema exposes ``crawl_instructions``, never a bare ``instructions``."""
    import inspect

    tools = SearchTool().get_tools()
    params = inspect.signature(_crawl_tool(tools)).parameters

    assert "crawl_instructions" in params
    assert "instructions" not in params

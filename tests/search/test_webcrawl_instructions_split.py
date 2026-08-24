"""``WebCrawl`` keeps the crawl bias and the tool description apart.

``WebCrawl`` used to re-declare ``instructions``, a field it already inherits from
``BaseToolParam``. One field then drove two unrelated effects: the value was passed to
Tavily's ``crawl(instructions=...)`` *and* appended to the docstring the model reads.
Neither could be asked for alone.

The two are still separate, but they no longer sit at the same layer. ``instructions``
is **configured** and describes *when* to crawl; ``crawl_instructions`` is a **required
tool argument** and directs *what* this particular crawl looks for. The tests below pin
each direction independently — each asserts both what its name does and what it does
**not** do, because the original defect was precisely that one name did both.
"""

from __future__ import annotations

import inspect
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
# crawl_instructions is required of the model, and reaches only Tavily
# ---------------------------------------------------------------------------


def test_crawl_instructions_has_no_default_so_the_model_must_supply_it() -> None:
    """A default would restore the unguided crawl by omission."""
    params = inspect.signature(_crawl_tool(SearchTool().get_tools())).parameters

    assert params["crawl_instructions"].default is inspect.Parameter.empty


def test_crawl_instructions_is_not_a_configurable_field() -> None:
    """Asserted on the class, because behaviour alone cannot see the difference.

    Reintroducing ``crawl_instructions`` as a ``WebCrawl`` field would give the tool
    argument a default and make it optional again — while still passing every
    forwarding test below.
    """
    assert "crawl_instructions" not in WebCrawl.model_fields


def test_crawl_instructions_is_forwarded_to_tavily(mock_tavily: _MockTavilyClient) -> None:
    """It reaches Tavily under Tavily's own kwarg name, ``instructions``."""
    tools = SearchTool().get_tools()

    _crawl_tool(tools)(url="http://example.com", crawl_instructions="focus on the API reference")

    _, kwargs = mock_tavily.crawl.call_args
    assert kwargs["instructions"] == "focus on the API reference"


def test_crawl_instructions_is_always_sent(mock_tavily: _MockTavilyClient) -> None:
    """Being required, it can never be absent from the call Tavily receives."""
    tools = SearchTool(web_crawl=WebCrawl()).get_tools()

    _crawl_tool(tools)(url="http://example.com", crawl_instructions="anything")

    _, kwargs = mock_tavily.crawl.call_args
    assert "instructions" in kwargs


def test_crawl_instructions_does_not_reach_the_docstring(mock_tavily: _MockTavilyClient) -> None:
    """Biasing a crawl must not rewrite the description the model reads."""
    tools = SearchTool().get_tools()
    tool = _crawl_tool(tools)

    tool(url="http://example.com", crawl_instructions="focus on the API reference")

    doc = tool.__doc__ or ""
    assert "focus on the API reference" not in doc


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
    """The original regression: describing the tool must not bias the crawl.

    Before the split, ``instructions`` was forwarded as Tavily's crawl guidance as
    well as appended to the description. Now the only value Tavily sees is the one
    the model passed for *this* call.
    """
    tools = SearchTool(web_crawl=WebCrawl(instructions="Only crawl public docs.")).get_tools()

    _crawl_tool(tools)(url="http://example.com", crawl_instructions="the pricing page")

    _, kwargs = mock_tavily.crawl.call_args
    assert kwargs["instructions"] == "the pricing page"
    assert "Only crawl public docs." not in kwargs["instructions"]


def test_the_two_layers_do_not_interfere(mock_tavily: _MockTavilyClient) -> None:
    """The point of the split: two intents, two names, no bleed in either direction."""
    tools = SearchTool(web_crawl=WebCrawl(instructions="Only crawl public docs.")).get_tools()
    tool = _crawl_tool(tools)

    tool(url="http://example.com", crawl_instructions="focus on the API reference")

    _, kwargs = mock_tavily.crawl.call_args
    assert kwargs["instructions"] == "focus on the API reference"

    doc = tool.__doc__ or ""
    assert "Only crawl public docs." in doc
    assert "focus on the API reference" not in doc


# ---------------------------------------------------------------------------
# Neither name is a field, and the model sees the right one
# ---------------------------------------------------------------------------


def test_webcrawl_does_not_redeclare_instructions() -> None:
    """``instructions`` must be the inherited field, not a shadowing re-declaration.

    Asserted on the owning class rather than on behaviour, because a re-declaration
    that happened to keep the same annotation would pass every test above while
    reintroducing exactly the shape this change removes.
    """
    assert "instructions" not in WebCrawl.__annotations__


def test_crawl_signature_names_the_argument_for_what_it_is() -> None:
    """The model's schema exposes ``crawl_instructions``, never a bare ``instructions``."""
    params = inspect.signature(_crawl_tool(SearchTool().get_tools())).parameters

    assert "crawl_instructions" in params
    assert "instructions" not in params

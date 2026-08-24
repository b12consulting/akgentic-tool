"""``web_fetch`` returns a bounded, relevance-filtered payload — never a page dump.

An unfiltered ``extract`` returns whole pages. A handful of long URLs then fills the
model's context before it has read any of them, and the ReAct loop stalls on its own
tool output. Three things keep the payload bounded, and each is pinned below because
each can be removed independently without any other test going red:

* ``query`` is **required** — it is what selects the passages. Give it a default and
  the model can omit it, which is the unfiltered call again under another name.
* ``chunks_per_source`` is a **configured default**, not a bare required argument.
  With no default the model picks the cap itself, so the one knob that bounds the
  response is the one knob deployment cannot bias.
* ``extract_depth`` and ``format`` are always sent, so the bound never silently
  reverts to whatever Tavily's own defaults happen to be.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

from akgentic.tool.search.search import SearchTool, WebFetch


class _MockTavilyClient:
    """Fake TavilyClient recording the kwargs ``extract`` was called with."""

    def __init__(self) -> None:
        self.extract = MagicMock(return_value={"results": []})


def _fetch_tool(tools: list[Any]) -> Any:
    """Return the ``web_fetch_tool`` callable from a built tool list."""
    return next(tool for tool in tools if tool.__name__ == "web_fetch_tool")


@pytest.fixture
def mock_tavily(monkeypatch: pytest.MonkeyPatch) -> _MockTavilyClient:
    """Install a recording TavilyClient and a key, and return the client."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    client = _MockTavilyClient()
    monkeypatch.setattr("akgentic.tool.search.search.TavilyClient", lambda: client)
    return client


# ---------------------------------------------------------------------------
# query is required, and it reaches Tavily
# ---------------------------------------------------------------------------


def test_query_has_no_default_so_the_model_cannot_omit_it() -> None:
    """A default on ``query`` would restore the unfiltered call by omission."""
    params = inspect.signature(_fetch_tool(SearchTool().get_tools())).parameters

    assert params["query"].default is inspect.Parameter.empty


def test_query_is_forwarded_to_tavily(mock_tavily: _MockTavilyClient) -> None:
    """The filter only bounds the payload if it actually reaches the API."""
    _fetch_tool(SearchTool().get_tools())(
        urls=["http://example.com"], query="what does the retry policy do"
    )

    _, kwargs = mock_tavily.extract.call_args
    assert kwargs["query"] == "what does the retry policy do"


# ---------------------------------------------------------------------------
# chunks_per_source is configurable, with a narrow default
# ---------------------------------------------------------------------------


def test_chunks_per_source_is_a_configured_default_not_a_required_argument() -> None:
    """Asserted on the signature, because behaviour alone cannot see the difference.

    A bare ``chunks_per_source: int`` forwards whatever the model passes and so
    satisfies every forwarding test below — while leaving deployment no way to bias
    the cap at all.
    """
    params = inspect.signature(_fetch_tool(SearchTool().get_tools())).parameters

    assert params["chunks_per_source"].default == WebFetch().chunks_per_source


def test_chunks_per_source_defaults_narrow(mock_tavily: _MockTavilyClient) -> None:
    """The out-of-the-box call is bounded, not merely boundable."""
    _fetch_tool(SearchTool().get_tools())(urls=["http://example.com"], query="q")

    _, kwargs = mock_tavily.extract.call_args
    assert kwargs["chunks_per_source"] == 3


def test_chunks_per_source_is_configurable(mock_tavily: _MockTavilyClient) -> None:
    """A team that needs exhaustive coverage of one page can raise the cap."""
    tools = SearchTool(web_fetch=WebFetch(chunks_per_source=10)).get_tools()

    _fetch_tool(tools)(urls=["http://example.com"], query="q")

    _, kwargs = mock_tavily.extract.call_args
    assert kwargs["chunks_per_source"] == 10


def test_chunks_per_source_is_overridable_per_call(mock_tavily: _MockTavilyClient) -> None:
    """Configuration biases the model's choice; it does not remove it."""
    tools = SearchTool(web_fetch=WebFetch(chunks_per_source=3)).get_tools()

    _fetch_tool(tools)(urls=["http://example.com"], query="q", chunks_per_source=7)

    _, kwargs = mock_tavily.extract.call_args
    assert kwargs["chunks_per_source"] == 7


# ---------------------------------------------------------------------------
# extract_depth and format are always sent
# ---------------------------------------------------------------------------


def test_extract_depth_is_always_sent(mock_tavily: _MockTavilyClient) -> None:
    """Omitting it would hand the depth back to Tavily's default, whatever that is."""
    _fetch_tool(SearchTool().get_tools())(urls=["http://example.com"], query="q")

    _, kwargs = mock_tavily.extract.call_args
    assert kwargs["extract_depth"] == "basic"


def test_format_is_markdown(mock_tavily: _MockTavilyClient) -> None:
    """Markdown keeps structure the model can use without the markup a page carries."""
    _fetch_tool(SearchTool().get_tools())(urls=["http://example.com"], query="q")

    _, kwargs = mock_tavily.extract.call_args
    assert kwargs["format"] == "markdown"


# ---------------------------------------------------------------------------
# web_search sends its depth too
# ---------------------------------------------------------------------------


def test_search_depth_is_always_sent(mock_tavily: _MockTavilyClient) -> None:
    """``search_depth`` stopped being optional; the call must reflect that."""
    mock_tavily.search = MagicMock(return_value={"results": []})
    tools = SearchTool().get_tools()

    next(t for t in tools if t.__name__ == "web_search_tool")(query="q")

    _, kwargs = mock_tavily.search.call_args
    assert kwargs["search_depth"] == "basic"

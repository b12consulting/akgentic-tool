"""The relevance filters are required in the schema the *model* receives.

Requiredness here is a claim about the JSON schema pydantic-ai hands the LLM, not
about Python. A signature check proves only that a Python caller must pass the
argument; what matters is that the model cannot omit it, and that is decided by
whether the name lands in the schema's ``required`` list.

The two are not the same assertion and can diverge — a default supplied anywhere in
the chain, or a parameter made keyword-only with a fallback, keeps the Python
signature strict-looking while quietly dropping the name from ``required``. These
tests read the built schema itself, so they fail if that ever happens.

``query`` on ``web_fetch`` and ``crawl_instructions`` on ``web_crawl`` are what keep
those two tools from returning a page dump and an unguided site walk respectively.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent

from akgentic.tool.search import SearchTool


@pytest.fixture
def tool_schemas() -> dict[str, dict[str, Any]]:
    """Build the SearchTool tools and return each one's JSON schema by tool name."""
    agent = Agent("openai:gpt-4o", tools=SearchTool().get_tools())
    return {
        name: tool.function_schema.json_schema
        for name, tool in agent._function_toolset.tools.items()
    }


def test_web_fetch_requires_a_query(tool_schemas: dict[str, dict[str, Any]]) -> None:
    """Without it the model can request an unfiltered extract of whole pages."""
    assert "query" in tool_schemas["web_fetch_tool"]["required"]


def test_web_crawl_requires_crawl_instructions(tool_schemas: dict[str, dict[str, Any]]) -> None:
    """Without it the model can start a broad, undirected walk of the site."""
    assert "crawl_instructions" in tool_schemas["web_crawl_tool"]["required"]


@pytest.mark.parametrize(
    ("tool_name", "optional_arg"),
    [
        ("web_fetch_tool", "chunks_per_source"),
        ("web_fetch_tool", "extract_depth"),
        ("web_crawl_tool", "limit"),
        ("web_search_tool", "max_results"),
    ],
)
def test_configured_defaults_stay_optional(
    tool_schemas: dict[str, dict[str, Any]],
    tool_name: str,
    optional_arg: str,
) -> None:
    """The counterpart claim: a param-model field is a default, never an obligation.

    Requiring one of these would force the model to invent a value it has no basis
    for, and would make the configured default unreachable.
    """
    schema = tool_schemas[tool_name]
    assert optional_arg in schema["properties"]
    assert optional_arg not in schema["required"]


def test_the_required_set_is_exactly_what_we_intend(
    tool_schemas: dict[str, dict[str, Any]],
) -> None:
    """Pinned whole, so a newly required argument cannot appear unnoticed.

    A tool that silently gains a required argument breaks every stored call and every
    model that learned the old shape; that should fail here rather than in a live run.
    """
    required = {name: set(schema["required"]) for name, schema in tool_schemas.items()}

    assert required == {
        "web_search_tool": {"query"},
        "web_fetch_tool": {"urls", "query"},
        "web_crawl_tool": {"url", "crawl_instructions"},
    }

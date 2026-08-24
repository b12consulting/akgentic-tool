"""A depth persisted as ``null`` under the older schema still loads.

``search_depth`` and ``extract_depth`` were once ``Literal["basic", "advanced"] | None``
defaulting to ``None`` — "send nothing, let Tavily choose". Every team persisted while
that was true carries a literal ``null`` for the field in its ``team.yaml``.

Narrowing the annotation to the two literals is not a private change: these params are
fields of a persisted ``Process``, so the stricter type is applied to data already on
disk. Without the healing validator the old files do not merely warn — deserialization
raises, the team is reported corrupted and is dropped at startup. The stored YAML is
left untouched; it is read forward, not migrated.

The payloads below are the ones observed failing, kept verbatim.
"""

from __future__ import annotations

import pytest

from akgentic.tool.search.search import WebCrawl, WebFetch, WebSearch

# The exact dict a pre-narrowing team.yaml deserializes to.
_PERSISTED_WEB_FETCH = {
    "expose": ["tool_call"],
    "extract_depth": None,
    "instructions": None,
    "timeout": 30.0,
}


def test_persisted_web_fetch_with_null_depth_loads() -> None:
    """The regression itself: this payload used to raise and lose the team."""
    assert WebFetch(**_PERSISTED_WEB_FETCH).extract_depth == "basic"


def test_persisted_web_fetch_keeps_its_other_stored_values() -> None:
    """Healing one field must not quietly reset the rest of the stored config."""
    fetch = WebFetch(**_PERSISTED_WEB_FETCH)

    assert fetch.timeout == 30.0
    assert fetch.instructions is None


def test_persisted_web_search_with_null_depth_loads() -> None:
    """``search_depth`` was narrowed in the same change and needs the same healing."""
    assert WebSearch(search_depth=None).search_depth == "basic"


# The exact dict a pre-narrowing team.yaml deserializes to for the crawl capability.
_PERSISTED_WEB_CRAWL = {
    "crawl_instructions": None,
    "expose": ["tool_call"],
    "extract_depth": None,
    "instructions": None,
    "limit": 10,
    "max_breadth": 5,
    "max_depth": 3,
    "timeout": 150.0,
}


def test_persisted_web_crawl_with_null_depth_loads() -> None:
    """``WebCrawl.extract_depth`` was narrowed last and broke the same five teams."""
    assert WebCrawl(**_PERSISTED_WEB_CRAWL).extract_depth == "basic"


def test_persisted_web_crawl_keeps_its_stored_budget() -> None:
    """The crawl budget is the expensive setting — healing must not reset it.

    A team that deliberately capped its crawl at 10 pages must not silently come back
    uncapped because an unrelated field on the same model needed healing.
    """
    crawl = WebCrawl(**_PERSISTED_WEB_CRAWL)

    assert crawl.limit == 10
    assert crawl.max_breadth == 5
    assert crawl.max_depth == 3
    assert crawl.timeout == 150.0


def test_a_removed_field_in_stored_config_is_tolerated() -> None:
    """``crawl_instructions`` is no longer a field, but old files still carry it.

    It must be ignored rather than rejected — the alternative is the corrupted-team
    failure again, this time for a name that simply moved to the tool signature.
    """
    assert WebCrawl(**_PERSISTED_WEB_CRAWL).extract_depth == "basic"
    assert not hasattr(WebCrawl(**_PERSISTED_WEB_CRAWL), "crawl_instructions")


@pytest.mark.parametrize("depth", ["basic", "advanced"])
def test_an_explicit_stored_depth_is_never_overwritten(depth: str) -> None:
    """Only ``None`` is healed — a team that chose ``advanced`` keeps it."""
    assert WebFetch(extract_depth=depth).extract_depth == depth
    assert WebSearch(search_depth=depth).search_depth == depth
    assert WebCrawl(extract_depth=depth).extract_depth == depth


def test_null_depth_survives_a_round_trip() -> None:
    """Reloading what we just wrote must be stable, not oscillate between values."""
    once = WebFetch(**_PERSISTED_WEB_FETCH)
    twice = WebFetch(**once.model_dump())

    assert twice.extract_depth == "basic"

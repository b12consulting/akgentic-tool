"""The never-used workspace-persistence mode is gone, and legacy payloads still load.

Two guards, because deleting a configuration field has two failure modes: a reference
left behind in code, and a persisted payload that no longer validates.
"""

from __future__ import annotations

import tokenize
from pathlib import Path

import pytest

from akgentic.tool.vector_store.actor import VectorStoreState
from akgentic.tool.vector_store.inmemory import InMemoryBackend
from akgentic.tool.vector_store.protocol import CollectionConfig

# Three of the four retired names belong to nothing else in the package, so they are
# swept across the whole tree — which is what catches a reintroduction in a package the
# deleted mode never reached.
GLOBALLY_RETIRED_NAMES = ("persistence", "save_collection", "load_collection")

# ``workspace_path`` cannot be swept that widely: ``sandbox/`` declares
# ``SandboxState.workspace_path`` (``sandbox/actor.py``), an unrelated field meaning a
# sandbox directory on the host, and uses it in ``seatbelt.py`` / ``bwrap.py`` /
# ``local.py``. It is swept only in the three packages the deleted mode reached.
SCOPED_PACKAGES = ("vector_store", "knowledge_graph", "planning")
SCOPED_RETIRED_NAMES = ("workspace_path",)


def _executable_source(path: Path) -> str:
    """Return *path*'s source with every comment and string literal removed.

    Docstrings are documentation, not references: ``CollectionConfig`` explains in
    prose that a legacy payload carrying the two retired keys is ignored, and that
    sentence must not read as a surviving use. Stripping tokens rather than matching
    a regex is what lets the sweep be strict about code while staying silent about
    the paragraph describing the deletion.

    Args:
        path: Python source file to strip.

    Returns:
        The file's tokens with COMMENT and STRING elided, joined by spaces.
    """
    kept: list[str] = []
    with path.open("rb") as fh:
        for token in tokenize.tokenize(fh.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if token.string:
                kept.append(token.string)
    return " ".join(kept)


def _tool_src() -> Path:
    """Return the root of the package's source tree."""
    return Path(__file__).resolve().parents[2] / "src" / "akgentic" / "tool"


def _all_files() -> list[Path]:
    """Return every Python module in the package."""
    return sorted(_tool_src().rglob("*.py"))


def _scoped_files() -> list[Path]:
    """Return every Python module in the three packages the deleted mode touched."""
    files: list[Path] = []
    for package in SCOPED_PACKAGES:
        files.extend(sorted((_tool_src() / package).rglob("*.py")))
    return files


class TestNoReferenceSurvivesInSource:
    """No executable reference to the deleted mode is left anywhere it could be."""

    def test_the_sweep_actually_reads_something(self) -> None:
        """Non-vacuity: a sweep that reads nothing would pass every assertion below."""
        scoped = _scoped_files()
        every = _all_files()
        assert len(scoped) >= 15, f"scoped sweep found only {len(scoped)} modules"
        assert len(every) > len(scoped), "the wide sweep must read more than the scoped one"
        joined = " ".join(_executable_source(f) for f in every)
        # Stripping strings must not have emptied the source of real code.
        assert "create_collection" in joined
        assert "CollectionConfig" in joined
        # And the wide sweep really does reach outside the three packages.
        assert "SandboxState" in joined

    @pytest.mark.parametrize("name", GLOBALLY_RETIRED_NAMES)
    def test_globally_retired_name_is_absent_from_the_whole_package(self, name: str) -> None:
        """These three names belong to nothing else, so nothing anywhere may use them."""
        offenders = [str(path) for path in _all_files() if name in _executable_source(path)]
        assert offenders == [], f"'{name}' still referenced in: {offenders}"

    @pytest.mark.parametrize("name", SCOPED_RETIRED_NAMES)
    def test_scoped_retired_name_is_absent_from_the_three_packages(self, name: str) -> None:
        """``workspace_path`` is gone from every package the deleted mode reached."""
        offenders = [str(path) for path in _scoped_files() if name in _executable_source(path)]
        assert offenders == [], f"'{name}' still referenced in: {offenders}"

    def test_the_backend_no_longer_offers_the_two_methods(self) -> None:
        """The npz + json sidecar pair is gone from the in-memory backend."""
        assert not hasattr(InMemoryBackend, "save_collection")
        assert not hasattr(InMemoryBackend, "load_collection")
        # The actor_state pair is untouched.
        assert hasattr(InMemoryBackend, "get_state")
        assert hasattr(InMemoryBackend, "restore_state")

    def test_the_config_no_longer_declares_the_two_fields(self) -> None:
        """Neither field exists on CollectionConfig any more."""
        assert "persistence" not in CollectionConfig.model_fields
        assert "workspace_path" not in CollectionConfig.model_fields


class TestLegacyPayloadsStillValidate:
    """A payload written before the deletion loads and quietly drops the dead keys.

    ``CollectionConfig`` declares no ``extra="forbid"`` and its base contributes only
    ``arbitrary_types_allowed``, so Pydantic's default ``extra="ignore"`` applies.
    That is why no migration is needed — and this test is what says so out loud.
    """

    def test_collection_config_ignores_the_two_legacy_keys(self) -> None:
        """The payload validates, keeps its live fields, and exposes neither dead key."""
        cfg = CollectionConfig.model_validate(
            {
                "dimension": 1536,
                "backend": "inmemory",
                "persistence": "workspace",
                "workspace_path": "/tmp/x",
            }
        )

        assert cfg.dimension == 1536
        assert cfg.backend == "inmemory"
        assert not hasattr(cfg, "persistence")
        assert not hasattr(cfg, "workspace_path")
        assert "persistence" not in cfg.model_dump()
        assert "workspace_path" not in cfg.model_dump()

    def test_a_restored_backend_snapshot_carrying_them_still_loads(self) -> None:
        """The keys reach CollectionConfig through restore_state and are dropped there."""
        backend = InMemoryBackend()
        backend.restore_state(
            {
                "collections": {
                    "col1": {
                        "config": {
                            "dimension": 3,
                            "backend": "inmemory",
                            "persistence": "workspace",
                            "workspace_path": "/tmp/x",
                        },
                        "entries": [],
                    }
                }
            }
        )

        result = backend.search("col1", [1.0, 0.0, 0.0], top_k=5)
        assert result.hits == []

    def test_a_state_payload_carrying_pending_entries_still_validates(self) -> None:
        """The rename of pending_entries needs no migration either."""
        restored = VectorStoreState.model_validate(
            {
                "pending_entries": {"c1": [{"ref_type": "t", "ref_id": "1", "text": "x"}]},
                "indexing_pending": {"c1": 1},
            }
        )

        assert restored.pending_requests == {}
        assert not hasattr(restored, "pending_entries")

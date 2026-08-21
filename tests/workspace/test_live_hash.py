"""The gate hashes the live file, and that is what distinguishes it from a cache.

This module exists on its own because it holds the epic's load-bearing
assertion. Every other test in story 29-3 passes against an implementation that
caches ``{path -> current_sha}`` — populating the map on the first check and on
every accepted write, then reading from it. Only the tests here go red, because
only they change the file **behind the actor's back**: not through a tool, not
through the actor, just bytes appearing on disk.

That case is not exotic. It is the ``akgentic-infra`` frontend upload, ADR-026
resource seeding, an ``ExecTool`` sandbox run, and a second team sharing a
``workspace_id`` — four writers that never call the actor, all caught for free,
because the check consults the *file* rather than a record of who wrote it.

If a future change makes the live hash a cache, this module is what must go red.
Do not weaken it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.actor import WorkspaceActor
from akgentic.tool.workspace.tool import WorkspaceTool

from tests.workspace.conftest import FakeActorToolObserver, mutate, read, tool_named

BODY = "alpha\nbravo\ncharlie\n"


@pytest.fixture
def notes(workspace_tree: Path) -> Path:
    """``notes.md``, present before any agent touches the workspace."""
    path = workspace_tree / "notes.md"
    path.write_text(BODY, encoding="utf-8")
    return path


def write_behind_the_actors_back(path: Path, text: str) -> None:
    """Change a file the way a writer that never opted in would.

    Deliberately not through the card, the actor, or ``Filesystem`` — the point
    is that nothing tells the actor this happened, and it must find out by
    looking.
    """
    path.write_text(text, encoding="utf-8")


class TestTheHashIsRead:
    def test_a_write_behind_the_actors_back_makes_the_agents_write_stale(
        self,
        wired_card: WorkspaceTool,
        notes: Path,
    ) -> None:
        """The load-bearing one. The agent works on the file, then loses it.

        The accepted write in the middle is not decoration: it is what makes
        this test bite. Without it the actor has never had occasion to remember
        anything about the path, so a lazily-populated cache would miss and read
        from disk anyway — refusing for the right reason by accident. After an
        accepted write, *every* cache shape has an entry to answer from: the one
        keyed off the observation the read reported, and the one filled on the
        write. Only reading the file finds the change.
        """
        read(wired_card, "notes.md")
        assert mutate(wired_card, "workspace_write", "notes.md", "my draft\n") == (
            "Written: notes.md"
        )

        write_behind_the_actors_back(notes, "someone else's version\n")

        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_write", "notes.md", "my revision\n")

        assert "changed since you read it" in str(refusal.value)
        assert notes.read_text(encoding="utf-8") == "someone else's version\n"

    def test_it_bites_on_the_very_first_mutation_too(
        self,
        wired_card: WorkspaceTool,
        notes: Path,
    ) -> None:
        # The same property with nothing before it: read, lose the file to an
        # outside writer, be refused.
        read(wired_card, "notes.md")
        write_behind_the_actors_back(notes, "someone else's version\n")

        with pytest.raises(RetriableError, match="changed since you read it"):
            mutate(wired_card, "workspace_write", "notes.md", "my version\n")
        assert notes.read_text(encoding="utf-8") == "someone else's version\n"

    def test_an_edit_is_refused_when_the_anchor_went_out_of_band(
        self,
        wired_card: WorkspaceTool,
        notes: Path,
    ) -> None:
        # The anchored path consults the live file too — its exact-only
        # degradation is decided by a digest taken from disk, not from a record.
        read(wired_card, "notes.md")
        mutate(wired_card, "workspace_edit", "notes.md", "alpha", "ALPHA")
        write_behind_the_actors_back(notes, "wholly different\n")

        with pytest.raises(RetriableError, match="no longer matches it exactly"):
            mutate(wired_card, "workspace_edit", "notes.md", "ALPHA", "OMEGA")
        assert notes.read_text(encoding="utf-8") == "wholly different\n"

    def test_a_delete_behind_the_actors_back_is_seen(
        self,
        wired_card: WorkspaceTool,
        notes: Path,
    ) -> None:
        read(wired_card, "notes.md")
        mutate(wired_card, "workspace_write", "notes.md", "my draft\n")
        notes.unlink()  # a foreign team, a sandbox run, a user

        with pytest.raises(RetriableError, match="deleted since you read it"):
            mutate(wired_card, "workspace_delete", "notes.md")

    def test_re_reading_after_the_refusal_lets_the_same_write_through(
        self,
        wired_card: WorkspaceTool,
        notes: Path,
    ) -> None:
        read(wired_card, "notes.md")
        write_behind_the_actors_back(notes, "someone else's version\n")
        with pytest.raises(RetriableError):
            mutate(wired_card, "workspace_write", "notes.md", "my version\n")

        # What the agent is told to do, done: re-read, then redo the change.
        read(wired_card, "notes.md")
        assert mutate(wired_card, "workspace_write", "notes.md", "my version\n") == (
            "Written: notes.md"
        )
        assert notes.read_text(encoding="utf-8") == "my version\n"

    def test_an_out_of_band_change_names_no_agent(
        self,
        wired_card: WorkspaceTool,
        notes: Path,
    ) -> None:
        # The path has a last writer on record — but the bytes on disk are no
        # longer that writer's, so attributing them to it would be a lie.
        read(wired_card, "notes.md")
        mutate(wired_card, "workspace_write", "notes.md", "mine\n")
        write_behind_the_actors_back(notes, "not from any agent\n")

        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_write", "notes.md", "mine again\n")

        message = str(refusal.value)
        assert "came from outside the" in message
        assert "last written by agent" not in message

    def test_an_edit_is_admitted_but_the_stale_anchor_is_not(
        self,
        wired_card: WorkspaceTool,
        notes: Path,
    ) -> None:
        # An anchored edit survives a change to an unrelated region; the anchor
        # is the precondition. But the anchor still has to be there.
        read(wired_card, "notes.md")
        write_behind_the_actors_back(notes, "alpha\nbravo\nDELTA\n")

        assert "-bravo" in mutate(wired_card, "workspace_edit", "notes.md", "bravo", "BRAVO")
        assert notes.read_text(encoding="utf-8") == "alpha\nBRAVO\nDELTA\n"

    def test_a_delete_behind_the_actors_back_makes_the_write_stale(
        self,
        wired_card: WorkspaceTool,
        notes: Path,
    ) -> None:
        read(wired_card, "notes.md")
        notes.unlink()

        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_write", "notes.md", "resurrected\n")

        assert "deleted since you read it" in str(refusal.value)
        assert not notes.exists()

    def test_the_actor_holds_no_map_of_live_content(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        notes: Path,
    ) -> None:
        # A structural companion to the behavioural tests above: not one map the
        # actor keeps can answer "what is in this file now". A map appearing here
        # that could is the regression this guards — the list is exhaustive on
        # purpose, so adding one is a deliberate act.
        #
        # Three of the six arrived with exec (29-5) and none of them holds file
        # content either: ``_slots`` is the deferred base's result cache, keyed
        # by run id and holding an ``ExecOutcome``; ``_run_errors`` and
        # ``_recent_runs`` are keyed by run id and agent id respectively and hold
        # strings.
        read(wired_card, "notes.md")
        mutate(wired_card, "workspace_write", "notes.md", "mine\n")

        maps = {
            name: value for name, value in vars(workspace_actor).items() if isinstance(value, dict)
        }
        assert set(maps) == {
            "_observations",
            "_last_writers",
            "_agent_names",
            "_slots",
            "_run_errors",
            "_recent_runs",
        }


class TestSeedingIsNotGated:
    def test_a_seeded_resource_is_written_without_any_observation(
        self,
        orchestrator_proxy: object,
        workspace_tree: Path,
    ) -> None:
        # Resource seeding runs at wiring time on the card's own handle, before
        # the agent's first turn and before the actor is bound. It is one of the
        # out-of-band writers the live hash exists to catch, not a caller of the
        # gate — and gating it would refuse a write no agent made.
        from akgentic.tool.workspace.tool import Resource

        observer = FakeActorToolObserver(orchestrator_proxy)  # type: ignore[arg-type]
        card = WorkspaceTool(
            workspace_id=workspace_tree.name,
            resources=[Resource(file_name="seeded.md", content="from the catalog\n")],
        )
        card.observer(observer)

        assert (workspace_tree / "seeded.md").read_text(encoding="utf-8") == ("from the catalog\n")
        # And the agent that owns the card has still not read it, so the gate
        # refuses to let it replace the file.
        with pytest.raises(RetriableError, match="read it before overwriting"):
            tool_named(card, "workspace_write")("seeded.md", "mine\n")

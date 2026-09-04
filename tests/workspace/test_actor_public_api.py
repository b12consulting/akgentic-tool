"""Import-path, MRO and no-shadow guard for the workspace actor decomposition.

Written against the **un-moved** tree and confirmed green there, so what it
asserts is what the move preserved rather than what the move produced.

``workspace/actor.py`` becoming ``workspace/actor/__init__.py`` is the one
refactor that *preserves* the dotted path: ``akgentic.tool.workspace.actor``
resolves to the package exactly as it resolved to the module. Nothing therefore
breaks loudly — which is why this file exists. The name that would vanish in
silence is ``EXEC_CAPABILITY``: it is public on the module, it moves out to
``actor/execution.py``, and ``workspace/__init__.py``'s ``__all__`` does not
carry it, so no other guard would notice its loss.

The other two failure modes are silent by construction:

- **Base order.** ``ExecMixin.deliver`` and ``ExecMixin.fail`` call ``super()``,
  which walks the MRO of ``type(self)``. Reordering the bases changes nothing
  today and changes everything the moment a second mixin defines either name.
- **Shadowing.** ``cache_capacity`` and ``negative_ttl_s`` are class attributes
  *with defaults* on ``DeferredResultActor``, and every mixin sits ahead of it in
  the MRO. A mixin that defined ``cache_capacity`` would silently resize the
  deferred LRU — no error, no log line.

The frozen set below was captured from the working tree, not transcribed from a
design document.
"""

from __future__ import annotations

import importlib

import akgentic.tool.workspace as ws
from akgentic.tool.core.deferred import DeferredResultActor
from akgentic.tool.workspace.actor import WorkspaceActor
from akgentic.tool.workspace.actor.execution import ExecMixin
from akgentic.tool.workspace.actor.gate import GateMixin
from akgentic.tool.workspace.actor.observation import ObservationMixin

_ACTOR_MODULE = "akgentic.tool.workspace.actor"

# Captured from ``akgentic/tool/workspace/actor.py`` before the move: every
# module-level name it defines that does not begin with an underscore.
# ``EXEC_CAPABILITY`` is the only one absent from ``workspace.__all__``.
_ACTOR_PUBLIC_NAMES: frozenset[str] = frozenset(
    {
        "EXEC_CAPABILITY",
        "WORKSPACE_ACTOR_NAME",
        "WORKSPACE_ACTOR_ROLE",
        "WorkspaceActor",
        "workspace_actor_name",
    }
)

# ``_slots`` and ``_in_flight`` are instance attributes set in
# ``DeferredResultActor.on_start``, so they are absent from ``dir()`` on the
# class and have to be named explicitly.
_BASE_INSTANCE_ATTRS: frozenset[str] = frozenset({"_slots", "_in_flight"})

# The two overrides the actor already ships, both deliberate and both calling
# ``super()``.
_DELIBERATE_OVERRIDES: frozenset[str] = frozenset({"deliver", "fail"})


class TestActorModulePathStillResolves:
    """The dotted path and the five names it serves, by the deserializer's mechanism."""

    def test_actor_module_still_imports(self) -> None:
        """``import_module`` on the pre-move path must not raise."""
        assert importlib.import_module(_ACTOR_MODULE) is not None

    def test_every_public_name_resolves_through_the_actor_module(self) -> None:
        """``import_module`` + ``getattr`` — ``hasattr`` on the package is not enough."""
        module = importlib.import_module(_ACTOR_MODULE)
        for name in sorted(_ACTOR_PUBLIC_NAMES):
            assert getattr(module, name, None) is not None, (
                f"{_ACTOR_MODULE}.{name} no longer resolves — it was a public name "
                f"of that module before the decomposition"
            )

    def test_exec_capability_keeps_its_value(self) -> None:
        """It is the first field of a discovered commit's subject, not a free label."""
        module = importlib.import_module(_ACTOR_MODULE)
        assert module.EXEC_CAPABILITY == "exec"

    def test_facade_and_module_expose_the_same_actor_class(self) -> None:
        """A second definition would be created by nothing and stopped by nobody."""
        module = importlib.import_module(_ACTOR_MODULE)
        assert ws.WorkspaceActor is module.WorkspaceActor
        assert ws.WorkspaceActor is WorkspaceActor


class TestActorMro:
    """The base order is frozen, because ``super()`` walks it."""

    def test_mro_is_exactly_the_four_mixins_then_the_deferred_base(self) -> None:
        """``ExecMixin`` first: its ``deliver``/``fail`` must reach the base."""
        assert WorkspaceActor.__mro__[:5] == (
            WorkspaceActor,
            ExecMixin,
            GateMixin,
            ObservationMixin,
            DeferredResultActor,
        )

    def test_super_from_exec_mixin_reaches_the_deferred_base(self) -> None:
        """The two middle mixins define neither name, so the chain passes through."""
        for mixin in (GateMixin, ObservationMixin):
            assert not _DELIBERATE_OVERRIDES & set(vars(mixin))


class TestNoMixinShadowsTheDeferredBase:
    """A mixin sits ahead of the base; anything it names there wins silently."""

    def test_mixins_shadow_exactly_the_two_deliberate_overrides(self) -> None:
        """A mixin defining ``cache_capacity`` would resize the LRU with no error."""
        declared: set[str] = set()
        for mixin in (ExecMixin, GateMixin, ObservationMixin):
            declared |= {name for name in vars(mixin) if not name.startswith("__")}
        base_surface = set(dir(DeferredResultActor)) | set(_BASE_INSTANCE_ATTRS)
        assert declared & base_surface == set(_DELIBERATE_OVERRIDES)

    def test_the_guard_is_not_vacuous(self) -> None:
        """The intersection above is only meaningful if both sides are populated."""
        for mixin in (ExecMixin, GateMixin, ObservationMixin):
            assert {name for name in vars(mixin) if not name.startswith("__")}
        assert {"cache_capacity", "negative_ttl_s", "get", "request"} <= set(
            dir(DeferredResultActor)
        )

    def test_the_mixins_declare_no_state_fields(self) -> None:
        """Every runtime map is initialised in ``on_start`` and nowhere else.

        A bare class-level annotation carries no value, so it lands in
        ``__annotations__`` and never in the class dict. An annotation given a
        *default* would land in both — and would then be the value every actor
        starts from, shared across the class.
        """
        for mixin in (ExecMixin, GateMixin, ObservationMixin):
            for name in getattr(mixin, "__annotations__", {}):
                assert name not in vars(mixin), (
                    f"{mixin.__name__}.{name} is annotated *and* assigned — a mixin "
                    f"declares the state it consumes, it does not own it"
                )

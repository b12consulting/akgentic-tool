"""Backend registry — the extension seam for pluggable vector-store backends.

A backend is anything satisfying the :class:`VectorStoreService` protocol. The
registry decouples *which* backend a collection uses from the actor that routes
to it: the actor never names a backend, it asks the registry for the spec whose
``name`` a :class:`CollectionConfig` carries and drives everything — construction,
state-sync policy, environment readiness — off that spec.

Adding a backend (Qdrant, pgvector, a bespoke store) is therefore a single
:func:`register_backend` call, with no edits to the actor:

```python
from akgentic.tool.vector_store.registry import BackendSpec, register_backend

register_backend(
    BackendSpec(
        name="my_store",
        factory=lambda ctx: MyStoreBackend(team_id=ctx.team_id),
        persists_in_actor_state=False,   # durable/external store
        is_configured=lambda: bool(os.environ.get("MY_STORE_URL")),
    )
)
```

Subclassing is the second extension route: a backend can subclass an existing
one (``QdrantBackend``, ``WeaviateBackend``) to override query construction and
register the subclass under its own name.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from akgentic.tool.vector_store.protocol import VectorStoreConfig, VectorStoreService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BackendContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendContext:
    """Everything a backend factory may need to build an instance.

    Passed to :attr:`BackendSpec.factory`. Carries the actor's own
    ``VectorStoreConfig`` (embedding + connection settings) and the owning
    team's id — propagated by the actor system, never a card field — so a
    backend can stamp its objects for later team-scoped cleanup.

    Attributes:
        config: The ``VectorStoreConfig`` of the owning ``VectorStoreActor``.
        team_id: Owning team id as a string, or ``None`` when unattributed.
    """

    config: VectorStoreConfig
    team_id: str | None = None


BackendFactory = Callable[[BackendContext], "VectorStoreService"]
"""Builds a backend instance from a :class:`BackendContext`.

Factories should import their client library lazily (inside the call) so that
registering a backend never forces its optional dependency to be installed.
"""


# ---------------------------------------------------------------------------
# BackendSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendSpec:
    """Declarative description of a registered backend.

    The actor reads these fields instead of switching on a backend name:
    ``factory`` builds it, ``persists_in_actor_state`` decides whether a
    mutation snapshots into the actor's serialisable state, ``is_configured``
    lets :func:`resolve_default_backend` pick a deployed store, and
    ``require_configured`` gives a consumer card a loud, build-time error when
    it names a store the environment has not provisioned.

    Attributes:
        name: Unique backend identifier, matched against ``CollectionConfig.backend``.
        factory: Builds a backend instance from a :class:`BackendContext`.
        persists_in_actor_state: ``True`` for backends whose data lives *in* the
            actor (the in-memory index) and must be snapshotted on every
            mutation. Such backends must also satisfy the ``ActorStateBackend``
            protocol. ``False`` is for durable external stores that own their
            own persistence.
        selectable_as_default: Whether :func:`resolve_default_backend` may pick
            this backend when a collection names none. External stores set this
            ``True``; the in-memory fallback sets it ``False`` so it is only ever
            the last resort.
        is_configured: Returns ``True`` when the environment has provisioned this
            backend (e.g. a cluster URL is exported). Defaults to always-ready.
        require_configured: Raises ``ValueError`` with remediation guidance when
            the backend is named but not provisioned. Called by consumer cards
            via :func:`akgentic.tool.vector_store.protocol.require_backend_configured`.
            Defaults to a no-op.
        legacy_actor_accessor: Private compatibility hook used only by built-in
            backends whose actor accessors predate the registry. Replacing a
            built-in registration without this hook routes through ``factory``
            like any other third-party backend.
    """

    name: str
    factory: BackendFactory
    persists_in_actor_state: bool = False
    selectable_as_default: bool = True
    is_configured: Callable[[], bool] = field(default=lambda: True)
    require_configured: Callable[[str], None] = field(default=lambda _card_name: None)
    legacy_actor_accessor: str | None = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, BackendSpec] = {}
_BUILTINS_LOADED: bool = False


def register_backend(spec: BackendSpec, *, replace: bool = False) -> None:
    """Register *spec* under ``spec.name``.

    Args:
        spec: The backend description to register.
        replace: When ``False`` (default) registering a name that already exists
            raises. Built-in backends register with ``replace=True`` so that a
            module re-import is idempotent.

    Raises:
        ValueError: When ``spec.name`` is already registered and ``replace`` is
            ``False``.
    """
    if spec.name in _REGISTRY and not replace:
        raise ValueError(
            f"A vector-store backend named '{spec.name}' is already registered. "
            f"Pass replace=True to override it deliberately."
        )
    _REGISTRY[spec.name] = spec


def unregister_backend(name: str) -> None:
    """Remove *name* from the registry if present (primarily for tests)."""
    _REGISTRY.pop(name, None)


def _ensure_builtins() -> None:
    """Import the built-in backend modules so they self-register.

    Idempotent and lazy: importing the backend modules here — rather than at
    registry import time — keeps the registry free of any dependency on the
    backends and avoids an import cycle (backends import ``protocol``, which
    imports this module).
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True  # set first so a factory-triggered re-entry is a no-op
    from akgentic.tool.vector_store import inmemory as _inmemory  # noqa: F401
    from akgentic.tool.vector_store import weaviate as _weaviate  # noqa: F401

    try:
        from akgentic.tool.vector_store import qdrant as _qdrant  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — a broken optional backend must not break the rest
        logger.debug("Qdrant backend module unavailable: %s", exc)


def get_backend_spec(name: str) -> BackendSpec:
    """Return the :class:`BackendSpec` registered under *name*.

    Args:
        name: Backend identifier (a ``CollectionConfig.backend`` value).

    Returns:
        The registered spec.

    Raises:
        ValueError: When no backend is registered under *name*.
    """
    _ensure_builtins()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown vector-store backend '{name}'. "
            f"Registered backends: {sorted(_REGISTRY)}. "
            f"Register it with akgentic.tool.vector_store.registry.register_backend()."
        ) from None


def is_registered(name: str) -> bool:
    """Return whether a backend is registered under *name*."""
    _ensure_builtins()
    return name in _REGISTRY


def available_backends() -> list[str]:
    """Return the sorted names of every registered backend."""
    _ensure_builtins()
    return sorted(_REGISTRY)


def resolve_default_backend() -> str:
    """Return the backend a collection uses when its card names none.

    Picks the first ``selectable_as_default`` backend, in registration order,
    whose ``is_configured()`` reports the environment has provisioned it —
    falling back to ``"inmemory"``. Registration order makes the choice
    deterministic: the in-memory fallback registers first (and opts out of
    default selection), then Weaviate, then any later external backends, so a
    deployment that exports only a Weaviate cluster still resolves to
    ``"weaviate"``.
    """
    _ensure_builtins()
    for name, spec in _REGISTRY.items():
        if not spec.selectable_as_default:
            continue
        try:
            if spec.is_configured():
                return name
        except Exception:  # noqa: BLE001 — a backend's readiness probe must not break resolution
            logger.debug("is_configured() raised for backend '%s'", name, exc_info=True)
    return "inmemory"

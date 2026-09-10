"""Process-level cluster client cache: one client per cluster, keyed and closed.

**One cache for every cluster backend, not one per vendor.** Every consumer in a
process that names the same cluster is handed the same client, whichever backend
it belongs to. The cache is keyed on :class:`ClusterKey` — whose leading field is
the registered backend name, so two backends pointed at one host:port are two
clients and never one — guarded by a real lock so that two actor threads
resolving one cluster at once open one connection, and emptied by
:func:`close_all`, which is registered with ``atexit`` on the first successful
connect.

**This module names no vendor and imports none.** It knows how to key a
connection, how to serialise a first connect, and how to close what it holds.
*How* to connect is a callable the backend supplies to :func:`get_client`, so
adding a third cluster backend adds no branch here. What the cache stores is
therefore heterogeneous, and :class:`_Closable` — the one method every value
must offer — is its type; ``dict[ClusterKey, Any]`` would be the shortcut, and
the concrete constraint exists.

Sharing one client across actor threads is safe for the calls this package
makes, and each backend's answer is pinned by a test.

**Weaviate** (``create_collection``, ``add``, ``remove``, ``search``,
``list_collections``, ``delete_by_team``) is safe under three invariants:

1. **A batch context never escapes ``add()``.** ``WeaviateBackend.add`` opens
   ``col.batch.dynamic()`` on a handle fetched in that call and leaves the ``with``
   block before returning. A batch object used from two threads is the one thing
   the vendor says is not thread-safe; a connection is not.
2. **The client is authenticated with an API key or nothing.** ``weaviate.py``'s
   connect callable builds ``AuthApiKey`` or ``None`` and accepts no
   ``AuthCredentials`` object. The OIDC path is the one place the connection
   mutates shared state per call (its gRPC header list), so not offering it is a
   thread-safety property of this package, not an omission.
3. **``close_all`` runs only when no consumer is live** — process shutdown and
   test fixtures. A close racing a request fails that request. Nothing in a
   team's lifecycle calls it: the client is shared across every team in the
   process, and one team stopping must not disconnect the others.

**Qdrant** (``upsert``, ``query_points``, ``scroll``, ``delete``,
``collection_exists``, ``create_collection``, ``get_collection``,
``get_collections``) is safe under two more, verified against
``qdrant-client`` 1.19.0:

4. **The connection is one thread-safe ``httpx.Client``, and the gRPC pool is
   never built.** ``QdrantRemote.__init__`` constructs a single
   ``SyncApis[ApiClient]`` (``qdrant_remote.py:233``) wrapping one ``httpx.Client``
   (``http/api_client.py:74``), which is documented for concurrent use and owns
   its connection pool. Every call ``QdrantBackend`` makes takes the
   ``self._prefer_grpc`` false branch — ``prefer_grpc`` defaults to ``False``
   (``qdrant_client.py:88``) and this package never passes it, nor reads the
   ``grpc_points`` / ``grpc_collections`` properties. **No per-call instance
   state is mutated on that path:** every ``self.`` write in ``qdrant_remote.py``
   after ``__init__`` lives in ``close()`` (``:322``) or the lazy gRPC helpers
   (``:370``, ``:376``, ``:382``, ``:388``, ``:394``, ``:400``), none of which the
   REST path reaches. ``ApiClient.request`` builds its request from locals
   (``api_client.py:84-98``), and the one shared object it touches — the
   middleware installed at construction — reads a ``ContextVar``
   (``context_headers.py:36-39``), which is per-thread by construction.
   ``close()`` is idempotent (``qdrant_remote.py:301-322``: the channel loop is
   guarded and swallows ``AttributeError``, the HTTP close is wrapped, and
   ``httpx.Client.close`` may be called twice); a close racing a request fails
   that request, exactly as invariant 3 describes.
5. **The factory builds a remote client only.** ``QdrantClient`` chooses the
   embedded ``QdrantLocal`` implementation — a different one, with different
   sharing rules — only for ``location=":memory:"`` (``qdrant_client.py:121``) or
   an explicit ``path=`` (``:126``). ``_make_qdrant_backend`` passes ``url=`` and
   refuses without one, so the local mode is unreachable through this package.

This module reads no environment variables. Each backend module owns its own
pair — ``backends/weaviate.py`` its ``weaviate_url()`` and ``weaviate_api_key()``,
``backends/qdrant.py`` its own — and each backend's factory resolves them and
builds the key.

See ADR-049 §Decision 5 and §Open items (1) for the argument.
"""

from __future__ import annotations

import atexit
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, cast
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class _Closable(Protocol):
    """The only contract every cached client shares.

    A heterogeneous client cache has no common vendor base class, so this is the
    stored type: the one method :func:`close_all` needs. It is what keeps
    ``Any`` out of the cache (Golden Rule 1c) while :func:`get_client` still
    hands each caller back its own concrete client type.
    """

    def close(self) -> None:
        """Release the connection."""
        ...


# ---------------------------------------------------------------------------
# ClusterKey
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterKey:
    """The connection a client is keyed on: backend, host, port, scheme, credential.

    Built from a URL by :meth:`from_url`, so ``http://localhost:8080``,
    ``http://localhost:8080/`` and ``http://LOCALHOST:8080`` are one key. The
    ``api_key`` is part of the identity — two callers naming one host with
    different credentials must not share a session — but it is hidden from
    ``repr``: a key is logged and appears in exception messages, and a secret in
    ``repr`` is a secret in every log line.

    **``backend`` leads because the cache is shared.** Two cluster backends can
    legitimately answer on one host and port — a test double, a proxy, a
    developer's single machine — and their clients are different objects
    speaking different protocols. Keying on the connection alone would hand one
    backend the other's client.

    Attributes:
        backend: The registered ``BackendSpec.name`` this client belongs to.
        host: Hostname, lower-cased by the URL parser; ``localhost`` when absent.
        port: The URL's port, else the backend's default (see :meth:`from_url`).
        secure: ``True`` iff the scheme is ``https``; applied to every transport
            the backend's connect callable opens.
        api_key: The API key, or ``None`` for an unauthenticated cluster.
    """

    backend: str
    host: str
    port: int
    secure: bool
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_url(
        cls,
        backend: str,
        url: str,
        api_key: str | None = None,
        *,
        default_port: int | None = None,
    ) -> ClusterKey:
        """Parse *url* into the key for the cluster it names.

        ``default_port`` is a **keying** concern and nothing else: it decides
        what a URL with no port collapses to, so that ``http://h`` and
        ``http://h:6333`` are one Qdrant cluster rather than two clients for one
        server. A backend that passes none keeps the historical rule — the URL's
        port, else ``443`` for ``https`` and ``8080`` otherwise — which is what
        Weaviate wants.

        Args:
            backend: The registered backend name this client belongs to.
            url: Cluster URL, e.g. ``http://localhost:8080``.
            api_key: Optional API key.
            default_port: Port to assume when the URL names none. ``None``
                applies the scheme-derived default.

        Returns:
            The key for the cluster the URL names.
        """
        parsed = urlparse(url)
        secure = parsed.scheme == "https"
        fallback = default_port if default_port is not None else (443 if secure else 8080)
        return cls(
            backend=backend,
            host=parsed.hostname or "localhost",
            port=parsed.port or fallback,
            secure=secure,
            api_key=api_key or None,
        )


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

_clients: dict[ClusterKey, _Closable] = {}
_lock = threading.Lock()
_shutdown_registered = False


def get_client[ClientT](key: ClusterKey, connect: Callable[[ClusterKey], ClientT]) -> ClientT:
    """Return this process's client for *key*, calling *connect* on first use.

    The connect happens under the module lock: serialising first connections is
    the point, and a connect is a one-off. A connect that raises caches nothing,
    so the next call tries again.

    **The connect is eager**, by the backend's own construction, which is what
    makes a misconfigured cluster fail at a consumer's bind rather than at its
    first write.

    Args:
        key: The cluster to resolve, including which backend is asking.
        connect: Builds the client for *key*. Supplied by the backend, which is
            the only place a vendor library is named.

    Returns:
        The shared, connected client for that key.

    Raises:
        Exception: Whatever *connect* raises — a missing optional dependency, an
            unreachable cluster, a URL that does not resolve.
    """
    global _shutdown_registered
    with _lock:
        cached = _clients.get(key)
        if cached is not None:
            # Heterogeneous by construction: the cache stores the one contract
            # every value shares, and the caller's own connect fixes the type.
            return cast(ClientT, cached)
        client = connect(key)
        _clients[key] = cast(_Closable, client)
        if not _shutdown_registered:
            atexit.register(close_all)
            _shutdown_registered = True
        return client


def close_all() -> None:
    """Close every cached client once and empty the cache.

    For process shutdown and test fixtures only (module docstring, invariant 3).
    Everything is popped under the lock first and closed outside it, so a slow or
    raising ``close()`` never holds the lock against a concurrent
    :func:`get_client`, and one failure never leaves another client open. A
    failing close is logged at WARNING with its key — whose ``repr`` hides the
    secret and names the backend — and the others are still closed. Idempotent by
    construction: an empty cache closes nothing.
    """
    with _lock:
        closing = list(_clients.items())
        _clients.clear()
    for key, client in closing:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to close the vector-store client for %r", key, exc_info=True)

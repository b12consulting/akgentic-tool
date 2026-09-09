"""Process-level Weaviate client cache: one client per cluster, keyed and closed.

Every consumer in a process that names the same cluster is handed the same
``weaviate.WeaviateClient``. The cache is keyed on the parsed connection
(:class:`ClusterKey`), guarded by a real lock so that two actor threads resolving
one cluster at once open one connection, and emptied by :func:`close_all`, which
is registered with ``atexit`` on the first successful connect.

Sharing one client across actor threads is safe for the calls this package makes
(``create_collection``, ``add``, ``remove``, ``search``, ``list_collections``,
``delete_by_team``) under three invariants, each pinned by a test:

1. **A batch context never escapes ``add()``.** ``WeaviateBackend.add`` opens
   ``col.batch.dynamic()`` on a handle fetched in that call and leaves the ``with``
   block before returning. A batch object used from two threads is the one thing
   the vendor says is not thread-safe; a connection is not.
2. **The client is authenticated with an API key or nothing.** :func:`get_client`
   takes ``api_key: str | None`` and builds ``AuthApiKey`` or ``None``; it does not
   accept an ``AuthCredentials`` object. The OIDC path is the one place the
   connection mutates shared state per call (its gRPC header list), so not
   offering it is a thread-safety property of this module, not an omission.
3. **``close_all`` runs only when no consumer is live** — process shutdown and
   test fixtures. A close racing a request fails that request. Nothing in a
   team's lifecycle calls it: the client is shared across every team in the
   process, and one team stopping must not disconnect the others.

This module reads no environment variables. ``protocol.py`` owns ``weaviate_url()``
and ``weaviate_api_key()``; the registry factory resolves them and passes explicit
values in. The ``weaviate-client`` import is checked at call time, inside
:func:`get_client`, never by a module-level probe.

See ADR-049 §Decision 5 and §Open items (1) for the argument.
"""

from __future__ import annotations

import atexit
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final
from urllib.parse import urlparse

if TYPE_CHECKING:
    from weaviate import WeaviateClient

logger = logging.getLogger(__name__)

GRPC_PORT: Final[int] = 50051
"""gRPC port used for every cluster. Fixed, and therefore not part of :class:`ClusterKey`."""

WEAVIATE_MISSING_MESSAGE: Final[str] = (
    "Weaviate backend requires the 'weaviate-client' package. "
    "Install with: pip install akgentic-tool[weaviate]"
)
"""The ``ImportError`` text raised when ``weaviate-client`` is not installed."""


# ---------------------------------------------------------------------------
# ClusterKey
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterKey:
    """The connection a client is keyed on: host, port, scheme and credential.

    Built from a URL by :meth:`from_url`, so ``http://localhost:8080``,
    ``http://localhost:8080/`` and ``http://LOCALHOST:8080`` are one key. The
    ``api_key`` is part of the identity — two callers naming one host with
    different credentials must not share a session — but it is hidden from
    ``repr``: a key is logged and appears in exception messages, and a secret in
    ``repr`` is a secret in every log line.

    Attributes:
        host: Hostname, lower-cased by the URL parser; ``localhost`` when absent.
        port: The URL's port, else ``443`` for ``https`` and ``8080`` otherwise.
        secure: ``True`` iff the scheme is ``https``; applied to HTTP and gRPC alike.
        api_key: The API key, or ``None`` for an unauthenticated cluster.
    """

    host: str
    port: int
    secure: bool
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_url(cls, url: str, api_key: str | None = None) -> ClusterKey:
        """Parse *url* exactly as ``WeaviateBackend.__init__`` did before this module.

        Args:
            url: Cluster URL, e.g. ``http://localhost:8080``.
            api_key: Optional API key.

        Returns:
            The key for the cluster the URL names.
        """
        parsed = urlparse(url)
        secure = parsed.scheme == "https"
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or (443 if secure else 8080),
            secure=secure,
            api_key=api_key or None,
        )


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

_clients: dict[ClusterKey, WeaviateClient] = {}
_lock = threading.Lock()
_shutdown_registered = False


def _check_weaviate_dependencies() -> None:
    """Validate that ``weaviate-client`` is importable, at call time.

    Raises:
        ImportError: With install instructions when ``weaviate-client`` is missing.
    """
    try:
        import weaviate  # noqa: F401
    except ImportError as exc:
        raise ImportError(WEAVIATE_MISSING_MESSAGE) from exc


def get_client(url: str, api_key: str | None = None) -> WeaviateClient:
    """Return this process's client for the cluster *url* names, connecting on first use.

    The connect happens under the module lock: serialising first connections is
    the point, and a connect is a one-off. A connect that raises caches nothing,
    so the next call tries again.

    Args:
        url: Cluster URL, e.g. ``http://localhost:8080``.
        api_key: API key, or ``None`` for an unauthenticated cluster. An
            ``AuthCredentials`` object is deliberately not accepted — see the
            module docstring, invariant 2.

    Returns:
        The shared, connected ``WeaviateClient`` for that cluster.

    Raises:
        ImportError: When ``weaviate-client`` is not installed.
        Exception: Whatever ``weaviate.connect_to_custom`` raises for an
            unreachable cluster — the connect is eager, so a misconfigured
            cluster fails here rather than at the first query.
    """
    global _shutdown_registered
    _check_weaviate_dependencies()
    import weaviate
    from weaviate.auth import AuthApiKey

    key = ClusterKey.from_url(url, api_key)
    with _lock:
        cached = _clients.get(key)
        if cached is not None:
            return cached
        client = weaviate.connect_to_custom(
            http_host=key.host,
            http_port=key.port,
            http_secure=key.secure,
            grpc_host=key.host,
            grpc_port=GRPC_PORT,
            grpc_secure=key.secure,
            auth_credentials=AuthApiKey(key.api_key) if key.api_key else None,
        )
        _clients[key] = client
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
    secret — and the others are still closed. Idempotent by construction: an
    empty cache closes nothing.
    """
    with _lock:
        closing = list(_clients.items())
        _clients.clear()
    for key, client in closing:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to close the Weaviate client for %r", key, exc_info=True)

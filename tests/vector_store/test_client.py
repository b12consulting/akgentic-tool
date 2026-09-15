"""Unit tests for the process-level cluster client cache (``vector_store/client.py``).

Covers: ``ClusterKey`` equality, the leading ``backend`` field and secret hiding,
per-backend default ports, one connect per key and zero on a repeat, the barrier
race under the lock, the four ``close_all`` behaviours, and ``atexit``
registration.

**The cache names no vendor any more**, so nothing here installs a mock
``weaviate``: the connect is a plain callable the test supplies, which is exactly
the seam a backend uses. The Weaviate keyword set, its ``AuthApiKey`` handling
and its import guard are ``test_weaviate.py``'s.

Each test imports the module under test **inside** the test, because the autouse
fixture evicts it so that each test starts with an empty cache.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import fields
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

CLIENT_MODULE = "akgentic.tool.vector_store.client"

URL = "http://localhost:8080"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _connect_factory() -> tuple[Any, list[Any]]:
    """Return a connect callable that mints a fresh client per call, and its log."""
    keys: list[Any] = []

    def connect(key: Any) -> MagicMock:
        keys.append(key)
        return MagicMock(name="client")

    return connect, keys


def _reset_modules() -> None:
    """Close whatever the loaded cache holds, then evict it."""
    loaded = sys.modules.get(CLIENT_MODULE)
    if loaded is not None:
        loaded.close_all()
    sys.modules.pop(CLIENT_MODULE, None)


@pytest.fixture(autouse=True)
def _clean_modules() -> Any:
    """Every test starts from an empty cache."""
    _reset_modules()
    yield
    _reset_modules()


# ---------------------------------------------------------------------------
# ClusterKey
# ---------------------------------------------------------------------------


class TestClusterKey:
    """The identity a client is cached under."""

    def test_one_cluster_one_key_whatever_the_url_spelling(self) -> None:
        from akgentic.tool.vector_store.client import ClusterKey

        base = ClusterKey.from_url("weaviate", URL)
        assert ClusterKey.from_url("weaviate", "http://localhost:8080/") == base
        assert ClusterKey.from_url("weaviate", "http://LOCALHOST:8080") == base
        assert hash(ClusterKey.from_url("weaviate", "http://LOCALHOST:8080/")) == hash(base)

    def test_the_scheme_decides_security_and_the_default_port(self) -> None:
        from akgentic.tool.vector_store.client import ClusterKey

        secure = ClusterKey.from_url("weaviate", "https://my-cluster.weaviate.cloud")
        assert secure.secure is True
        assert secure.port == 443

        plain = ClusterKey.from_url("weaviate", "http://localhost")
        assert plain.secure is False
        assert plain.port == 8080

        explicit = ClusterKey.from_url("weaviate", "http://10.0.0.5:9999")
        assert explicit.port == 9999

        explicit_https = ClusterKey.from_url("weaviate", "https://cluster.example:8443")
        assert explicit_https.port == 8443

    def test_a_backend_default_port_collapses_the_portless_url(self) -> None:
        """Qdrant's 6333: ``http://h`` and ``http://h:6333`` are one cluster."""
        from akgentic.tool.vector_store.client import ClusterKey

        implied = ClusterKey.from_url("qdrant", "http://h", default_port=6333)
        explicit = ClusterKey.from_url("qdrant", "http://h:6333")
        assert implied == explicit
        assert implied.port == 6333

    def test_a_default_port_beats_the_scheme_derived_one(self) -> None:
        from akgentic.tool.vector_store.client import ClusterKey

        assert ClusterKey.from_url("qdrant", "https://h", default_port=6333).port == 6333

    def test_host_port_scheme_and_key_each_separate_a_cluster(self) -> None:
        from akgentic.tool.vector_store.client import ClusterKey

        base = ClusterKey.from_url("weaviate", URL)
        assert ClusterKey.from_url("weaviate", "http://other:8080") != base
        assert ClusterKey.from_url("weaviate", "http://localhost:8081") != base
        assert ClusterKey.from_url("weaviate", "https://localhost:8080") != base
        assert ClusterKey.from_url("weaviate", URL, "key") != base
        assert ClusterKey.from_url("weaviate", URL, "key") != ClusterKey.from_url(
            "weaviate", URL, "other-key"
        )

    def test_the_backend_name_is_part_of_the_identity(self) -> None:
        """Two backends on one host:port are two keys, never one shared client."""
        from akgentic.tool.vector_store.client import ClusterKey

        assert ClusterKey.from_url("weaviate", URL) != ClusterKey.from_url("qdrant", URL)

    def test_the_field_set_leads_with_the_backend(self) -> None:
        from akgentic.tool.vector_store.client import ClusterKey

        names = [f.name for f in fields(ClusterKey)]
        assert names == ["backend", "host", "port", "secure", "api_key"]

    def test_the_api_key_stays_out_of_repr(self) -> None:
        from akgentic.tool.vector_store.client import ClusterKey

        key = ClusterKey.from_url("weaviate", URL, "s3cr3t-token")
        assert "s3cr3t-token" not in repr(key)
        assert "weaviate" in repr(key)
        assert "localhost" in repr(key)


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------


class TestGetClient:
    """One client per key per process, built by the caller's own connect."""

    def test_one_connect_per_key_and_zero_on_a_repeat(self) -> None:
        from akgentic.tool.vector_store.client import ClusterKey, get_client

        connect, keys = _connect_factory()
        key = ClusterKey.from_url("weaviate", URL, "key")

        first = get_client(key, connect)
        second = get_client(
            ClusterKey.from_url("weaviate", "http://LOCALHOST:8080/", "key"), connect
        )

        assert second is first
        assert len(keys) == 1

    def test_two_backends_on_one_host_get_two_clients(self) -> None:
        """The leading ``backend`` field is what keeps them apart."""
        from akgentic.tool.vector_store.client import ClusterKey, get_client

        connect, keys = _connect_factory()

        weaviate_client = get_client(ClusterKey.from_url("weaviate", URL), connect)
        qdrant_client = get_client(ClusterKey.from_url("qdrant", URL), connect)

        assert weaviate_client is not qdrant_client
        assert len(keys) == 2
        assert [k.backend for k in keys] == ["weaviate", "qdrant"]

    def test_the_connect_receives_the_key(self) -> None:
        from akgentic.tool.vector_store.client import ClusterKey, get_client

        connect, keys = _connect_factory()
        key = ClusterKey.from_url("qdrant", "https://h:6333", "k")

        get_client(key, connect)

        assert keys == [key]

    def test_a_different_cluster_or_key_gets_its_own_client(self) -> None:
        from akgentic.tool.vector_store.client import ClusterKey, get_client

        connect, keys = _connect_factory()
        clients = [
            get_client(ClusterKey.from_url("weaviate", URL), connect),
            get_client(ClusterKey.from_url("weaviate", "http://other:8080"), connect),
            get_client(ClusterKey.from_url("weaviate", "http://localhost:8081"), connect),
            get_client(ClusterKey.from_url("weaviate", "https://localhost:8080"), connect),
            get_client(ClusterKey.from_url("weaviate", URL, "key"), connect),
        ]

        assert len({id(c) for c in clients}) == 5
        assert len(keys) == 5

    def test_a_failed_connect_caches_nothing(self) -> None:
        """The next call tries again rather than handing out a broken client."""
        from akgentic.tool.vector_store.client import ClusterKey, get_client

        calls: list[int] = []

        def connect(_key: Any) -> MagicMock:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("cluster down")
            return MagicMock(name="client")

        key = ClusterKey.from_url("weaviate", URL)
        with pytest.raises(RuntimeError, match="cluster down"):
            get_client(key, connect)
        client = get_client(key, connect)

        assert len(calls) == 2
        assert get_client(key, connect) is client
        assert len(calls) == 2

    def test_the_module_reads_no_environment(self) -> None:
        """A key is an argument, never a lookup — the module imports ``os`` at all."""
        import akgentic.tool.vector_store.client as client_module

        assert "os" not in vars(client_module)
        assert "weaviate" not in vars(client_module)
        assert "qdrant_client" not in vars(client_module)


# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------


class TestTheLock:
    """First connections are serialised: N racing threads open one connection."""

    def test_threads_released_together_connect_once(self) -> None:
        """The sleep in the fake connect is what makes an unlocked cache fail every time."""
        from akgentic.tool.vector_store.client import ClusterKey, get_client

        connects: list[int] = []

        def slow_connect(_key: Any) -> MagicMock:
            connects.append(1)
            time.sleep(0.05)
            return MagicMock(name="client")

        key = ClusterKey.from_url("weaviate", URL)
        n = 8
        barrier = threading.Barrier(n)
        results: list[Any] = []
        results_lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            client = get_client(key, slow_connect)
            with results_lock:
                results.append(client)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert len(results) == n
        assert len(connects) == 1
        assert all(client is results[0] for client in results)


# ---------------------------------------------------------------------------
# close_all
# ---------------------------------------------------------------------------


class TestCloseAll:
    """Closes every cached client once, empties the cache, and survives a failing close."""

    def test_closes_every_client_once_whatever_backend_and_is_idempotent(self) -> None:
        """A second close_all has nothing left to close."""
        from akgentic.tool.vector_store.client import ClusterKey, close_all, get_client

        connect, _ = _connect_factory()
        first = get_client(ClusterKey.from_url("weaviate", "http://a:8080"), connect)
        second = get_client(ClusterKey.from_url("qdrant", "http://b:6333"), connect)

        close_all()
        first.close.assert_called_once()
        second.close.assert_called_once()

        close_all()
        first.close.assert_called_once()
        second.close.assert_called_once()

    def test_close_all_on_an_empty_cache_is_a_no_op(self) -> None:
        from akgentic.tool.vector_store.client import close_all

        close_all()
        close_all()

    def test_the_cache_is_cleared_not_merely_closed(self) -> None:
        """After close_all the same cluster connects again and gets a new client."""
        from akgentic.tool.vector_store.client import ClusterKey, close_all, get_client

        connect, keys = _connect_factory()
        key = ClusterKey.from_url("weaviate", URL)

        before = get_client(key, connect)
        close_all()
        after = get_client(key, connect)

        assert after is not before
        assert len(keys) == 2

    def test_a_failing_close_does_not_stop_the_others(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The failure is logged at WARNING, the rest are closed, the cache is empty."""
        from akgentic.tool.vector_store.client import ClusterKey, close_all, get_client

        connect, keys = _connect_factory()
        broken = get_client(ClusterKey.from_url("weaviate", "http://a:8080", "secret-a"), connect)
        healthy = get_client(ClusterKey.from_url("qdrant", "http://b:6333"), connect)
        broken.close.side_effect = RuntimeError("transport gone")

        with caplog.at_level(logging.WARNING, logger=CLIENT_MODULE):
            close_all()

        healthy.close.assert_called_once()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "host='a'" in message
        assert "backend='weaviate'" in message
        assert "secret-a" not in caplog.text

        get_client(ClusterKey.from_url("weaviate", "http://a:8080", "secret-a"), connect)
        assert len(keys) == 3


# ---------------------------------------------------------------------------
# atexit
# ---------------------------------------------------------------------------


class TestAtexit:
    """close_all is registered once, on the first successful connect — never at import."""

    def test_registered_once_across_two_backends(self) -> None:
        """Two connects on two backends, one registration."""
        import akgentic.tool.vector_store.client as client_module

        connect, _ = _connect_factory()
        with patch("atexit.register") as register:
            client_module.get_client(
                client_module.ClusterKey.from_url("weaviate", "http://a:8080"), connect
            )
            client_module.get_client(
                client_module.ClusterKey.from_url("qdrant", "http://b:6333"), connect
            )

        register.assert_called_once_with(client_module.close_all)

    def test_import_registers_nothing(self) -> None:
        """A process that never opens a cluster registers no shutdown hook."""
        with patch("atexit.register") as register:
            import akgentic.tool.vector_store.client  # noqa: F401

        register.assert_not_called()

    def test_a_failed_connect_registers_nothing(self) -> None:
        """Registration follows the first *successful* connect."""
        from akgentic.tool.vector_store.client import ClusterKey, get_client

        def connect(_key: Any) -> MagicMock:
            raise RuntimeError("cluster down")

        with patch("atexit.register") as register, pytest.raises(RuntimeError):
            get_client(ClusterKey.from_url("weaviate", URL), connect)

        register.assert_not_called()

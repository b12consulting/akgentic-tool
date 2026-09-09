"""Unit tests for the process-level Weaviate client cache (``vector_store/client.py``).

Covers: ``ClusterKey`` equality and secret hiding, one connect per cluster and the
connect arguments, the barrier race under the lock, the four ``close_all``
behaviours, ``atexit`` registration, and the call-time import guard.

Every test runs against a mock ``weaviate`` module installed into ``sys.modules`` —
the ``dev`` extra does not install ``weaviate-client`` — and imports the module
under test **inside** the test, because the autouse fixture evicts it so that each
test starts with an empty cache.
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

from akgentic.tool.vector_store.protocol import WEAVIATE_API_KEY_ENV, WEAVIATE_URL_ENV

CLIENT_MODULE = "akgentic.tool.vector_store.client"
BACKEND_MODULE = "akgentic.tool.vector_store.weaviate"

URL = "http://localhost:8080"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_mock_weaviate() -> MagicMock:
    """Install a mock ``weaviate`` whose ``connect_to_custom`` mints a new client per call."""
    mock_weaviate = MagicMock()
    mock_weaviate.connect_to_custom.side_effect = lambda **_: MagicMock(name="client")
    mock_weaviate.auth = MagicMock()
    sys.modules["weaviate"] = mock_weaviate
    sys.modules["weaviate.auth"] = mock_weaviate.auth
    return mock_weaviate


def _reset_modules() -> None:
    """Close whatever the loaded cache holds, then evict it and the mock ``weaviate``."""
    loaded = sys.modules.get(CLIENT_MODULE)
    if loaded is not None:
        loaded.close_all()
    for name in [k for k in sys.modules if k.startswith("weaviate")]:
        del sys.modules[name]
    sys.modules.pop(CLIENT_MODULE, None)
    sys.modules.pop(BACKEND_MODULE, None)


@pytest.fixture(autouse=True)
def _clean_modules() -> Any:
    """Every test starts from an empty cache and a fresh mock."""
    _reset_modules()
    yield
    _reset_modules()


# ---------------------------------------------------------------------------
# ClusterKey (AC 1, AC 2)
# ---------------------------------------------------------------------------


class TestClusterKey:
    """The key is the parsed connection, not the URL string."""

    def test_two_spellings_of_one_cluster_are_one_key(self) -> None:
        """A trailing slash or a capitalised host does not make a second cluster."""
        from akgentic.tool.vector_store.client import ClusterKey

        base = ClusterKey.from_url(URL)
        assert ClusterKey.from_url("http://localhost:8080/") == base
        assert ClusterKey.from_url("http://LOCALHOST:8080") == base
        assert hash(ClusterKey.from_url("http://LOCALHOST:8080/")) == hash(base)

    def test_parses_exactly_as_the_old_constructor_did(self) -> None:
        """hostname or localhost; the URL's port, else 443 for https and 8080 otherwise."""
        from akgentic.tool.vector_store.client import ClusterKey

        secure = ClusterKey.from_url("https://my-cluster.weaviate.cloud")
        assert (secure.host, secure.port, secure.secure) == ("my-cluster.weaviate.cloud", 443, True)

        plain = ClusterKey.from_url("http://localhost")
        assert (plain.host, plain.port, plain.secure) == ("localhost", 8080, False)

        explicit = ClusterKey.from_url("http://10.0.0.5:9999")
        assert (explicit.host, explicit.port, explicit.secure) == ("10.0.0.5", 9999, False)

        explicit_https = ClusterKey.from_url("https://cluster.example:8443")
        assert (explicit_https.port, explicit_https.secure) == (8443, True)

    def test_keys_differ_on_every_connection_parameter(self) -> None:
        """Host, port, scheme and api_key each make a distinct cluster."""
        from akgentic.tool.vector_store.client import ClusterKey

        base = ClusterKey.from_url(URL)
        assert ClusterKey.from_url("http://other:8080") != base
        assert ClusterKey.from_url("http://localhost:8081") != base
        assert ClusterKey.from_url("https://localhost:8080") != base
        assert ClusterKey.from_url(URL, "key") != base
        assert ClusterKey.from_url(URL, "key") != ClusterKey.from_url(URL, "other-key")

    def test_grpc_port_is_fixed_and_not_part_of_the_key(self) -> None:
        """The gRPC port is a module constant, so it cannot split one cluster into two."""
        from akgentic.tool.vector_store.client import GRPC_PORT, ClusterKey

        assert GRPC_PORT == 50051
        assert {f.name for f in fields(ClusterKey)} == {"host", "port", "secure", "api_key"}

    def test_repr_hides_the_api_key(self) -> None:
        """A key is logged and appears in exception text; the secret must not."""
        from akgentic.tool.vector_store.client import ClusterKey

        key = ClusterKey.from_url(URL, "s3cr3t-token")
        assert key.api_key == "s3cr3t-token"
        assert "s3cr3t-token" not in repr(key)
        assert "s3cr3t-token" not in str(key)


# ---------------------------------------------------------------------------
# get_client (AC 3, AC 8) — absorbs the old TestConnectionConfig
# ---------------------------------------------------------------------------


class TestGetClient:
    """One client per cluster per process, connected with the old constructor's arguments."""

    def test_one_connect_per_cluster_and_the_same_object(self) -> None:
        """Two calls naming one cluster connect once and return one object."""
        mock_weaviate = _install_mock_weaviate()
        from akgentic.tool.vector_store.client import get_client

        first = get_client(URL, "key")
        second = get_client("http://LOCALHOST:8080/", "key")

        assert second is first
        mock_weaviate.connect_to_custom.assert_called_once()

    def test_a_different_cluster_or_key_gets_its_own_client(self) -> None:
        """Host, port, scheme and api_key each open a separate connection."""
        mock_weaviate = _install_mock_weaviate()
        from akgentic.tool.vector_store.client import get_client

        clients = [
            get_client(URL),
            get_client("http://other:8080"),
            get_client("http://localhost:8081"),
            get_client("https://localhost:8080"),
            get_client(URL, "key"),
        ]

        assert len({id(c) for c in clients}) == 5
        assert mock_weaviate.connect_to_custom.call_count == 5

    def test_connects_with_the_arguments_the_constructor_sent(self) -> None:
        """The whole keyword set, including the fixed gRPC port."""
        mock_weaviate = _install_mock_weaviate()
        from akgentic.tool.vector_store.client import get_client

        get_client(URL)

        assert mock_weaviate.connect_to_custom.call_args[1] == {
            "http_host": "localhost",
            "http_port": 8080,
            "http_secure": False,
            "grpc_host": "localhost",
            "grpc_port": 50051,
            "grpc_secure": False,
            "auth_credentials": None,
        }

    def test_https_url_is_secure_on_both_transports(self) -> None:
        """HTTPS sets http_secure and grpc_secure, and the port defaults to 443."""
        mock_weaviate = _install_mock_weaviate()
        from akgentic.tool.vector_store.client import get_client

        get_client("https://my-cluster.weaviate.cloud")

        kwargs = mock_weaviate.connect_to_custom.call_args[1]
        assert kwargs["http_secure"] is True
        assert kwargs["grpc_secure"] is True
        assert kwargs["http_port"] == 443

    def test_an_api_key_becomes_auth_api_key(self) -> None:
        """AuthApiKey is built from the key and handed over as auth_credentials."""
        mock_weaviate = _install_mock_weaviate()
        from akgentic.tool.vector_store.client import get_client

        get_client(URL, "test-key")

        mock_weaviate.auth.AuthApiKey.assert_called_once_with("test-key")
        kwargs = mock_weaviate.connect_to_custom.call_args[1]
        assert kwargs["auth_credentials"] is mock_weaviate.auth.AuthApiKey.return_value

    def test_no_api_key_means_no_auth(self) -> None:
        """Without a key the connection is unauthenticated, not authenticated with ''."""
        mock_weaviate = _install_mock_weaviate()
        from akgentic.tool.vector_store.client import get_client

        get_client(URL)
        get_client("http://other:8080", "")

        for call in mock_weaviate.connect_to_custom.call_args_list:
            assert call[1]["auth_credentials"] is None
        mock_weaviate.auth.AuthApiKey.assert_not_called()

    def test_a_failed_connect_caches_nothing(self) -> None:
        """The next call tries again rather than handing out a broken client."""
        mock_weaviate = _install_mock_weaviate()
        mock_weaviate.connect_to_custom.side_effect = [
            RuntimeError("cluster down"),
            MagicMock(name="client"),
        ]
        from akgentic.tool.vector_store.client import get_client

        with pytest.raises(RuntimeError, match="cluster down"):
            get_client(URL)
        client = get_client(URL)

        assert mock_weaviate.connect_to_custom.call_count == 2
        assert get_client(URL) is client
        assert mock_weaviate.connect_to_custom.call_count == 2

    def test_reads_no_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The URL is an argument, never a lookup; the exported key is never read."""
        monkeypatch.setenv(WEAVIATE_URL_ENV, "http://from-env:8080")
        monkeypatch.setenv(WEAVIATE_API_KEY_ENV, "env-key")
        mock_weaviate = _install_mock_weaviate()
        from akgentic.tool.vector_store.client import get_client

        with pytest.raises(TypeError):
            get_client()  # type: ignore[call-arg]

        get_client("http://explicit:8080")

        kwargs = mock_weaviate.connect_to_custom.call_args[1]
        assert kwargs["http_host"] == "explicit"
        assert kwargs["auth_credentials"] is None
        mock_weaviate.auth.AuthApiKey.assert_not_called()


# ---------------------------------------------------------------------------
# The lock (AC 4)
# ---------------------------------------------------------------------------


class TestTheLock:
    """First connections are serialised: N racing threads open one connection."""

    def test_threads_released_together_connect_once(self) -> None:
        """The sleep in the fake connect is what makes an unlocked cache fail every time."""
        mock_weaviate = _install_mock_weaviate()
        connects: list[int] = []

        def slow_connect(**_: Any) -> MagicMock:
            connects.append(1)
            time.sleep(0.05)
            return MagicMock(name="client")

        mock_weaviate.connect_to_custom.side_effect = slow_connect
        from akgentic.tool.vector_store.client import get_client

        n = 8
        barrier = threading.Barrier(n)
        results: list[Any] = []
        results_lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            client = get_client(URL)
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
# close_all (AC 5)
# ---------------------------------------------------------------------------


class TestCloseAll:
    """Closes every cached client once, empties the cache, and survives a failing close."""

    def test_closes_every_client_once_and_is_idempotent(self) -> None:
        """A second close_all has nothing left to close."""
        _install_mock_weaviate()
        from akgentic.tool.vector_store.client import close_all, get_client

        first = get_client("http://a:8080")
        second = get_client("http://b:8080")

        close_all()
        first.close.assert_called_once()
        second.close.assert_called_once()

        close_all()
        first.close.assert_called_once()
        second.close.assert_called_once()

    def test_the_cache_is_cleared_not_merely_closed(self) -> None:
        """After close_all the same cluster connects again and gets a new client."""
        mock_weaviate = _install_mock_weaviate()
        from akgentic.tool.vector_store.client import close_all, get_client

        before = get_client(URL)
        close_all()
        after = get_client(URL)

        assert after is not before
        assert mock_weaviate.connect_to_custom.call_count == 2

    def test_a_failing_close_does_not_stop_the_others(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The failure is logged at WARNING, the rest are closed, the cache is empty."""
        mock_weaviate = _install_mock_weaviate()
        from akgentic.tool.vector_store.client import close_all, get_client

        broken = get_client("http://a:8080", "secret-a")
        healthy = get_client("http://b:8080")
        broken.close.side_effect = RuntimeError("transport gone")

        with caplog.at_level(logging.WARNING, logger=CLIENT_MODULE):
            close_all()

        healthy.close.assert_called_once()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "host='a'" in warnings[0].getMessage()
        assert "secret-a" not in caplog.text

        get_client("http://a:8080", "secret-a")
        assert mock_weaviate.connect_to_custom.call_count == 3


# ---------------------------------------------------------------------------
# atexit (AC 6)
# ---------------------------------------------------------------------------


class TestAtexit:
    """close_all is registered once, on the first successful connect — never at import."""

    def test_registered_once_across_two_clusters(self) -> None:
        """Two connects, one registration."""
        _install_mock_weaviate()
        import akgentic.tool.vector_store.client as client_module

        with patch("atexit.register") as register:
            client_module.get_client("http://a:8080")
            client_module.get_client("http://b:8080")

        register.assert_called_once_with(client_module.close_all)

    def test_import_registers_nothing(self) -> None:
        """A process that never touches Weaviate registers no shutdown hook."""
        _install_mock_weaviate()

        with patch("atexit.register") as register:
            import akgentic.tool.vector_store.client  # noqa: F401

        register.assert_not_called()

    def test_a_failed_connect_registers_nothing(self) -> None:
        """Registration follows the first *successful* connect."""
        mock_weaviate = _install_mock_weaviate()
        mock_weaviate.connect_to_custom.side_effect = RuntimeError("cluster down")
        from akgentic.tool.vector_store.client import get_client

        with patch("atexit.register") as register, pytest.raises(RuntimeError):
            get_client(URL)

        register.assert_not_called()


# ---------------------------------------------------------------------------
# Import guard (AC 7) — moved from test_weaviate.py, now targeting get_client
# ---------------------------------------------------------------------------


class TestImportGuard:
    """ImportError with install instructions when weaviate-client is missing."""

    def test_import_error_when_weaviate_missing(self) -> None:
        """get_client raises ImportError naming the extra to install."""
        with patch.dict(sys.modules, {"weaviate": None}):
            from akgentic.tool.vector_store.client import get_client

            with pytest.raises(ImportError, match=r"akgentic-tool\[weaviate\]"):
                get_client(URL)

    def test_the_check_runs_at_call_time_not_at_import(self) -> None:
        """A module imported while weaviate was absent still connects once it is present."""
        with patch.dict(sys.modules, {"weaviate": None}):
            import akgentic.tool.vector_store.client as client_module

            with pytest.raises(ImportError):
                client_module.get_client(URL)

        mock_weaviate = _install_mock_weaviate()
        client_module.get_client(URL)

        mock_weaviate.connect_to_custom.assert_called_once()

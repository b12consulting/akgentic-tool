"""What only a running Weaviate can prove about the two vector shapes a class may have.

``WeaviateBackend.create_collection`` creates a new class with one self-provided
**named** vector, ``default``. Every class created before that holds the legacy
**unnamed** vector, so a cluster can hold both shapes at once. ``add`` passes a
bare vector and ``search`` names no target vector. Whether the server accepts
that on each shape is a fact about Weaviate, which no mocked spec in
``test_weaviate.py`` can reach. So each shape is created for real here, bound by
a second backend, written, searched and removed, and the hits are asserted.

**The image is pinned.** ``1.32.5`` is supported by both clients the project runs:
weaviate-client 4.16.2 (the workspace venv, minimum server 1.23.7) and 4.24.1
(the worker image, minimum server 1.27.0).

**The gate is the one ``tests/sandbox/test_docker_container_integration.py``
uses, for the same reason.** ``integration`` keeps these specs out of a run that
deselects it. Package CI runs ``pytest tests/`` with no ``-m`` filter, so the
``skipif`` on a bounded daemon probe is what protects a runner with no daemon.
GitHub's ``ubuntu-latest`` has one, so these specs do run in package CI. Docker
is driven through ``subprocess``, so no dependency is added. The module timeout
covers the image pull and the readiness wait, which the suite's per-test
default would not.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import warnings
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

    import weaviate

    from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
    from akgentic.tool.vector_store.vector import VectorEntry

IMAGE = "cr.weaviate.io/semitechnologies/weaviate:1.32.5"
_READY_TIMEOUT_S = 90.0
_PULL_TIMEOUT_S = 480.0


def _daemon_available() -> bool:
    """True when a docker daemon answers, under a bound so a dead socket cannot hang."""
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


_DAEMON = _daemon_available()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _DAEMON, reason="needs a running Docker daemon"),
    pytest.mark.timeout(_PULL_TIMEOUT_S + _READY_TIMEOUT_S + 60),
]


def _docker(*args: str, timeout: float = 60.0) -> str:
    """Run one docker command and return its stdout, raising on failure."""
    done = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=True
    )
    return done.stdout.strip()


def _host_port(container: str, port: int) -> int:
    """The host port docker published for the container's *port*."""
    mapping = _docker("port", container, f"{port}/tcp").splitlines()[0]
    return int(mapping.rsplit(":", 1)[1])


def _wait_ready(http_port: int) -> None:
    """Poll the readiness endpoint until it answers 200, or fail after the timeout."""
    url = f"http://127.0.0.1:{http_port}/v1/.well-known/ready"
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    pytest.fail(f"Weaviate at {url} was not ready within {_READY_TIMEOUT_S:.0f}s")


@pytest.fixture(scope="module")
def client() -> Iterator[weaviate.WeaviateClient]:
    """A connected client to a fresh Weaviate, whose container is removed afterwards."""
    weaviate = pytest.importorskip("weaviate")
    present = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=30)
    if present.returncode != 0:
        _docker("pull", IMAGE, timeout=_PULL_TIMEOUT_S)
    container = _docker(
        "run",
        "-d",
        "-p",
        "127.0.0.1::8080",
        "-p",
        "127.0.0.1::50051",
        "-e",
        "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true",
        "-e",
        "PERSISTENCE_DATA_PATH=/var/lib/weaviate",
        "-e",
        "DEFAULT_VECTORIZER_MODULE=none",
        "-e",
        "CLUSTER_HOSTNAME=node1",
        IMAGE,
    )
    try:
        http_port = _host_port(container, 8080)
        _wait_ready(http_port)
        connected = weaviate.connect_to_local(
            host="127.0.0.1", port=http_port, grpc_port=_host_port(container, 50051)
        )
        try:
            yield connected
        finally:
            connected.close()
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=60)


def _class_name(shape: str) -> str:
    """A class name no other spec uses, so the specs share the container safely."""
    return f"Shape_{shape}_{uuid.uuid4().hex[:8]}"


def _backend(client: weaviate.WeaviateClient) -> WeaviateBackend:
    from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

    return WeaviateBackend(client=client, team_id="team-a")


def _create_unnamed_shape(
    client: weaviate.WeaviateClient, name: str, *, multi_tenant: bool
) -> None:
    """Create a class the way ``create_collection`` did before the named vector."""
    from weaviate.classes.config import Configure, DataType, Property

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="ref_type", data_type=DataType.TEXT),
                Property(name="ref_id", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="team_id", data_type=DataType.TEXT),
            ],
            multi_tenancy_config=Configure.multi_tenancy(enabled=multi_tenant),
        )


def _entry(ref_id: str, vector: list[float]) -> VectorEntry:
    from akgentic.tool.vector_store.vector import VectorEntry

    return VectorEntry(ref_type="entity", ref_id=ref_id, text=f"text {ref_id}", vector=vector)


@pytest.mark.parametrize("shape", ["named", "unnamed"])
def test_add_search_and_remove_work_on_both_vector_shapes(
    client: weaviate.WeaviateClient, shape: str
) -> None:
    from akgentic.tool.vector_store.protocol import VectorStoreParam

    name = _class_name(shape)
    param = VectorStoreParam(dimension=3)
    if shape == "named":
        _backend(client).create_collection(name, param)
    else:
        _create_unnamed_shape(client, name, multi_tenant=True)

    # The spec must reach the shape it names, or it proves nothing about it.
    vectors = client.collections.get(name).config.get().vector_config
    assert (set(vectors or {}) == {"default"}) is (shape == "named")

    backend = _backend(client)
    backend.create_collection(name, param)  # binds the existing class
    backend.add(name, [_entry("near", [1.0, 0.0, 0.0]), _entry("far", [0.0, 1.0, 0.0])])

    hits = backend.search(name, [1.0, 0.1, 0.0], top_k=5).hits
    assert [hit.ref_id for hit in hits] == ["near", "far"]
    assert hits[0].text == "text near"
    assert hits[0].score > hits[1].score

    backend.remove(name, ["near"])
    assert [hit.ref_id for hit in backend.search(name, [1.0, 0.1, 0.0], top_k=5).hits] == ["far"]


def test_a_new_class_is_created_without_the_deprecated_vectorizer_argument(
    client: weaviate.WeaviateClient,
) -> None:
    from akgentic.tool.vector_store.protocol import VectorStoreParam

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _backend(client).create_collection(_class_name("fresh"), VectorStoreParam(dimension=3))

    assert not [w for w in caught if "vectorizer_config" in str(w.message)]


def test_a_plain_class_is_refused_by_the_real_server(client: weaviate.WeaviateClient) -> None:
    from akgentic.tool.vector_store.protocol import VectorStoreParam

    name = _class_name("plain")
    _create_unnamed_shape(client, name, multi_tenant=False)
    backend = _backend(client)

    with pytest.raises(ValueError, match=r"plain \(not multi-tenant\)"):
        backend.create_collection(name, VectorStoreParam(dimension=3))
    with pytest.raises(ValueError, match="does not exist"):
        backend.add(name, [_entry("r1", [1.0, 0.0, 0.0])])

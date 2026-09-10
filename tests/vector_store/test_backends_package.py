"""The backends live in their own package, and the protocol names none of them.

Each guard here reads what was actually imported — ``protocol.__file__``,
``registry.__file__``, ``akgentic.tool.__file__`` — so a run against the wrong source tree
checks, and fails on, the wrong file rather than passing on a file nobody ran.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
import sys
import textwrap
import tokenize
from pathlib import Path
from types import ModuleType

import pytest

import akgentic.tool
from akgentic.tool.vector_store import actor, protocol, registry
from akgentic.tool.vector_store.registry import available_backends

VECTOR_STORE = "akgentic.tool.vector_store"
BACKENDS = f"{VECTOR_STORE}.backends"
BUILT_IN_NAMES = ("inmemory", "weaviate", "qdrant")


# ---------------------------------------------------------------------------
# The protocol names no backend
# ---------------------------------------------------------------------------


class TestTheProtocolNamesNoBackend:
    """``protocol.py`` is the contract every backend implements, so it names none of them."""

    def test_no_registered_backend_name_occurs_anywhere_in_the_file(self) -> None:
        """The whole file, lowercased: code, docstrings, comments and Field descriptions.

        The tokens are the registered names, and the floor ties them to the registry so the
        tuple cannot drift from what is registered. The category words "in-memory",
        "in memory" and "cluster" are deliberately allowed: they are not registered names.
        They describe the two answers of ``persists_in_actor_state``, which the protocol
        itself defines through ``ActorStateBackend``. That boundary is the rule, not a
        narrowing of it — a vendor-specific claim reworded to dodge the tokens is a review
        defect this spec cannot see.
        """
        registered = available_backends()
        for name in BUILT_IN_NAMES:
            assert name in registered, f"'{name}' is not a registered backend: {registered}"
        text = Path(protocol.__file__).read_text(encoding="utf-8")
        assert text, "read nothing from protocol.py"
        assert "class VectorStoreService(Protocol)" in text
        assert "def check_path_prefix" in text

        offenders = [
            f"{number}: {line.strip()}"
            for number, line in enumerate(text.lower().splitlines(), start=1)
            if any(name in line for name in BUILT_IN_NAMES)
        ]

        assert offenders == [], "protocol.py names a backend:\n" + "\n".join(offenders)

    @pytest.mark.parametrize("module", [protocol, actor], ids=["protocol", "actor"])
    def test_no_import_reaches_into_the_backends_package(self, module: ModuleType) -> None:
        """Every import statement, ``TYPE_CHECKING`` blocks and function bodies included.

        For a ``from X import a`` both ``X`` and ``X.a`` are checked, so
        ``from akgentic.tool.vector_store import backends`` is caught as well as
        ``from akgentic.tool.vector_store.backends.inmemory import InMemoryBackend``.
        """
        path = Path(str(module.__file__))
        package = module.__name__.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"))

        reached: list[str] = []
        from_vector_store = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                reached.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = _absolute_module(node, package)
                if base == VECTOR_STORE or base.startswith(f"{VECTOR_STORE}."):
                    from_vector_store += 1
                reached.append(base)
                reached.extend(f"{base}.{alias.name}" for alias in node.names)

        assert from_vector_store >= 1, f"the walk saw no vector_store import in {path}"
        offenders = [
            name for name in reached if name == BACKENDS or name.startswith(f"{BACKENDS}.")
        ]
        assert offenders == [], f"{path.name} imports from the backends package: {offenders}"


def _absolute_module(node: ast.ImportFrom, package: str) -> str:
    """The absolute module an ``ImportFrom`` names, resolving a relative import."""
    if node.level == 0:
        return node.module or ""
    parts = package.split(".")
    base = ".".join(parts[: len(parts) - (node.level - 1)])
    return f"{base}.{node.module}" if node.module else base


# ---------------------------------------------------------------------------
# The old module paths are gone, with no alias left behind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", BUILT_IN_NAMES)
def test_the_old_module_path_no_longer_exists(name: str) -> None:
    """No stored record and no other package names these paths, so no alias is kept."""
    old_path = f"{VECTOR_STORE}.{name}"

    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module(old_path)

    assert excinfo.value.name == old_path


# ---------------------------------------------------------------------------
# The registry's built-in loader resolves
# ---------------------------------------------------------------------------


def test_the_registry_loads_the_built_ins_from_the_backends_package() -> None:
    """The only spec that sees a stale path in ``_ensure_builtins``.

    Its Qdrant import sits inside ``except Exception``, and the package root imports all
    three modules eagerly anyway, so a wrong path there is swallowed and then masked:
    every behavioural spec stays green. Only reading the statements, and asking the
    import system whether each one resolves, goes red.
    """
    tree = ast.parse(Path(registry.__file__).read_text(encoding="utf-8"))
    loaders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_builtins"
    ]
    assert len(loaders) == 1, "registry.py must define _ensure_builtins exactly once"
    imports = [node for node in ast.walk(loaders[0]) if isinstance(node, ast.ImportFrom)]

    assert len(imports) == 3, [ast.unparse(node) for node in imports]
    assert {node.module for node in imports} == {BACKENDS}
    names = {alias.name for node in imports for alias in node.names}
    assert names == set(BUILT_IN_NAMES)
    for node in imports:
        for alias in node.names:
            target = f"{node.module}.{alias.name}"
            assert importlib.util.find_spec(target) is not None, f"{target} does not resolve"


# ---------------------------------------------------------------------------
# The package imports with both cluster clients absent
# ---------------------------------------------------------------------------

_BLOCKER = textwrap.dedent(
    """
    import sys


    class _Blocker:
        \"\"\"Refuse the two cluster clients to every import below this point.\"\"\"

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {"weaviate", "qdrant_client"}:
                raise ModuleNotFoundError(name)
            return None


    sys.meta_path.insert(0, _Blocker())
    """
)

_IMPORT_THE_PACKAGE = _BLOCKER + textwrap.dedent(
    """
    import akgentic.tool.vector_store as vs

    for name in ("WeaviateBackend", "QdrantBackend"):
        cls = getattr(vs, name)
        assert cls is not None, name
        assert isinstance(cls, type), (name, cls)
        assert cls.__module__.startswith("akgentic.tool.vector_store.backends."), cls.__module__
    assert "weaviate" not in sys.modules, "the package imported the weaviate client"
    assert "qdrant_client" not in sys.modules, "the package imported the qdrant client"
    print("IMPORTED")
    """
)

_PROBE_THE_BLOCKER = _BLOCKER + textwrap.dedent(
    """
    try:
        import weaviate  # noqa: F401
    except ModuleNotFoundError:
        print("BLOCKED")
    else:
        print("NOT BLOCKED")
    """
)


def _run(script: str) -> subprocess.CompletedProcess[str]:
    """Run *script* in a clean interpreter that inherits this one's environment.

    The child inherits ``PYTHONPATH``, so it imports the same source tree this run does.
    """
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_package_imports_without_either_cluster_client() -> None:
    """Both vendors are imported lazily, so the classes exist without their clients.

    The package root imports the cluster classes plainly — no ``try/except`` that would
    set a class to ``None`` — and this is what shows that is safe: no client is imported
    at module level, so nothing can raise.
    """
    result = _run(_IMPORT_THE_PACKAGE)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "IMPORTED", result.stdout


def test_the_blocker_really_refuses_the_weaviate_client() -> None:
    """Guard the guard: the client is installed here, so a leaky finder would pass vacuously."""
    assert importlib.util.find_spec("weaviate") is not None, "weaviate is not installed here"

    result = _run(_PROBE_THE_BLOCKER)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "BLOCKED", result.stdout


# ---------------------------------------------------------------------------
# The retired construction path left no identifier behind
# ---------------------------------------------------------------------------

RETIRED_IDENTIFIERS = (
    "require_weaviate_configured",
    "legacy_actor_accessor",
    "_get_or_create_backend",
    "_get_or_create_weaviate_backend",
)
"""Swept as NAME tokens across the whole package.

``weaviate_url`` is deliberately not here: it is a live function in the Weaviate backend
module. Strings and comments are stripped, so a docstring recording the deletion is not a
reference.
"""


def _name_tokens(path: Path) -> set[str]:
    """Every NAME token in *path*: identifiers and keywords, never strings or comments."""
    with path.open("rb") as handle:
        return {
            token.string
            for token in tokenize.tokenize(handle.readline)
            if token.type == tokenize.NAME
        }


def _package_files() -> list[Path]:
    """Every Python module of the imported ``akgentic.tool`` package."""
    return sorted(Path(akgentic.tool.__file__).parent.rglob("*.py"))


def test_the_sweep_reads_real_code() -> None:
    """Non-vacuity: a sweep that read nothing would pass every assertion below."""
    files = _package_files()
    assert len(files) >= 60, f"the sweep found only {len(files)} modules"
    names = set().union(*(_name_tokens(path) for path in files))
    assert {"create_collection", "VectorStoreParam", "register_backend", "_get_backend"} <= names


@pytest.mark.parametrize("identifier", RETIRED_IDENTIFIERS)
def test_a_retired_identifier_is_absent_from_source(identifier: str) -> None:
    offenders = [str(path) for path in _package_files() if identifier in _name_tokens(path)]
    assert offenders == [], f"'{identifier}' still referenced in: {offenders}"

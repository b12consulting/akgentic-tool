"""The ``[vector_search]`` optional-extra guard, proven rather than assumed.

``akgentic/tool/__init__.py`` wraps the vector import in ``try/except ImportError`` so the
package still imports when the extra is not installed. That guard has never fired: the
vector module imports ``numpy`` and ``openai`` only under ``TYPE_CHECKING`` and inside
function bodies, so the import cannot raise. Moving the module into ``vector_store/``
widened what sits behind the ``try`` — the new path runs ``vector_store/__init__.py`` and
its whole surface — so the guard staying inert is now worth pinning behind a test rather
than an argument.

Two halves are asserted, because only the first is obvious: the package imports, **and**
the three names are still in ``__all__``. A test that asserted only "no crash" would stay
green if the guard silently started swallowing an unrelated failure and dropping them.

The extra has to be made absent for real. A ``sys.modules`` patch does not do it — numpy is
loaded long before this test runs, and ``test_kg_imports.py`` records that dead end in its
own body. A subprocess with a blocking meta-path finder is what actually works.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_BLOCK_AND_IMPORT = textwrap.dedent(
    """
    import sys


    class _Blocker:
        \"\"\"Refuse numpy and openai to every import below this point.\"\"\"

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {"numpy", "openai"}:
                raise ModuleNotFoundError(name)
            return None


    for _loaded in [key for key in sys.modules if key.split(".")[0] in {"numpy", "openai"}]:
        del sys.modules[_loaded]
    sys.meta_path.insert(0, _Blocker())

    import akgentic.tool

    assert "VectorEntry" in akgentic.tool.__all__, akgentic.tool.__all__
    assert "EmbeddingService" in akgentic.tool.__all__, akgentic.tool.__all__
    assert "VectorIndex" in akgentic.tool.__all__, akgentic.tool.__all__
    """
)


def _run_with_extra_blocked() -> subprocess.CompletedProcess[str]:
    """Import ``akgentic.tool`` in a clean interpreter with the extra unimportable."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_AND_IMPORT],
        capture_output=True,
        text=True,
        check=False,
    )


def test_package_imports_and_still_exports_the_three_names_without_the_extra() -> None:
    """The guard stays inert: the import succeeds and ``__all__`` is unchanged."""
    result = _run_with_extra_blocked()
    assert result.returncode == 0, result.stderr


def test_the_blocker_actually_blocks() -> None:
    """Guard the guard: a finder that let numpy through would make the test vacuous."""
    probe = textwrap.dedent(
        """
        import sys


        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {"numpy", "openai"}:
                    raise ModuleNotFoundError(name)
                return None


        for _loaded in [key for key in sys.modules if key.split(".")[0] in {"numpy", "openai"}]:
            del sys.modules[_loaded]
        sys.meta_path.insert(0, _Blocker())

        try:
            import numpy
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("numpy was importable despite the blocker")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr

"""Benchmarks that are imported by the suite but never collected by it.

``pytest``'s ``python_files`` pattern only matches ``test_*.py``, so nothing in
this package runs in CI. The smoke test in ``tests/workspace/`` imports the
harness at a tiny size and asserts its *shape*, which is what keeps a benchmark
nobody runs weekly from rotting into an import error.
"""

"""Backward-compat shim — import from akgentic.tool.vector instead."""

from __future__ import annotations

from akgentic.tool.vector import EmbeddingService, VectorIndex  # noqa: F401

__all__ = ["EmbeddingService", "VectorIndex"]

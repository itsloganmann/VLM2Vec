"""Retriever package exposing multi-vector retriever implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import compression, memory, scoring
from .base import BatchOutput, BaseRetriever

if TYPE_CHECKING:  # pragma: no cover - import-only for typing
    from .nemoretriever import NemoRetriever

__all__ = [
    "BaseRetriever",
    "BatchOutput",
    "NemoRetriever",
    "scoring",
    "memory",
    "compression",
]


def __getattr__(name: str):  # pragma: no cover - trivial proxy
    if name == "NemoRetriever":
        from .nemoretriever import NemoRetriever as _NemoRetriever

        return _NemoRetriever
    raise AttributeError(f"module 'retriever' has no attribute {name!r}")

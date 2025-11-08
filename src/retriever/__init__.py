"""Retriever package exposing multi-vector retriever implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import compression, memory, scoring
from .base import BatchOutput, BaseRetriever

if TYPE_CHECKING:  # pragma: no cover - import-only for typing
    from .colqwen2 import ColQwen2Retriever
    from .colnomic import ColNomicRetriever
    from .nemoretriever import NemoRetriever

__all__ = [
    "BaseRetriever",
    "BatchOutput",
    "ColQwen2Retriever",
    "NemoRetriever",
    "ColNomicRetriever",
    "scoring",
    "memory",
    "compression",
]


def __getattr__(name: str):  # pragma: no cover - trivial proxy
    if name == "ColQwen2Retriever":
        from .colqwen2 import ColQwen2Retriever as _ColQwen2Retriever

        return _ColQwen2Retriever
    if name == "NemoRetriever":
        from .nemoretriever import NemoRetriever as _NemoRetriever

        return _NemoRetriever
    if name == "ColNomicRetriever":
        from .colnomic import ColNomicRetriever as _ColNomicRetriever

        return _ColNomicRetriever
    raise AttributeError(f"module 'retriever' has no attribute {name!r}")

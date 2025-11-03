from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
import yaml
from torch import Tensor

logger = logging.getLogger(__name__)


class CacheManager:
    """Handles persistent storage of pre-computed embeddings on disk."""

    def __init__(self, root: Path, dtype: torch.dtype = torch.bfloat16, device: Optional[torch.device] = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        logger.debug("Cache root initialised at %s", self.root)

    def _key(self, metadata: Dict[str, Any]) -> str:
        packed = json.dumps(metadata, sort_keys=True).encode("utf-8")
        return hashlib.sha256(packed).hexdigest()

    def resolve(self, metadata: Dict[str, Any]) -> Path:
        return self.root / f"{self._key(metadata)}.pt"

    def exists(self, metadata: Dict[str, Any]) -> bool:
        return self.resolve(metadata).exists()

    def load(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        path = self.resolve(metadata)
        logger.debug("Loading embeddings from %s", path)
        payload = torch.load(path, map_location=self.device)
        return payload

    def save(self, metadata: Dict[str, Any], embeddings: Tensor, extra: Optional[Dict[str, Any]] = None) -> Path:
        path = self.resolve(metadata)
        payload = {"embeddings": embeddings.to(self.device, dtype=self.dtype)}
        if extra:
            payload["meta"] = extra  # type: ignore[assignment]
        torch.save(payload, path)
        logger.debug("Saved embeddings to %s", path)
        return path


def chunk_iterable(collection: Sequence[Any], chunk_size: int) -> Iterator[Sequence[Any]]:
    for idx in range(0, len(collection), chunk_size):
        yield collection[idx : idx + chunk_size]


class ResumeManager:
    """YAML-backed resume checkpoints for long-running evaluations."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}\n")

    def _read(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        with open(self.path, "r") as fh:
            return yaml.safe_load(fh) or {}

    def update(self, model_alias: str, task_name: str, stage: str, info: Dict[str, Any]) -> None:
        state = self._read()
        state.setdefault(model_alias, {})
        state[model_alias].setdefault(task_name, {})
        state[model_alias][task_name][stage] = info
        with open(self.path, "w") as fh:
            yaml.safe_dump(state, fh)

    def last(self) -> Dict[str, Any]:
        return self._read()

    def print_resume(self) -> None:
        state = self._read()
        if not state:
            print("Resume manager has no checkpoints yet.")
            return
        print("Resume checkpoints:")
        for model_alias, tasks in state.items():
            for task, stages in tasks.items():
                for stage, info in stages.items():
                    print(f"- model={model_alias} task={task} stage={stage} info={info}")


def tensor_nbytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def estimate_vram_usage(tensors: Sequence[Tensor]) -> float:
    total_bytes = sum(tensor_nbytes(t) for t in tensors)
    return total_bytes / (1024 ** 3)


__all__ = [
    "CacheManager",
    "chunk_iterable",
    "ResumeManager",
    "tensor_nbytes",
    "estimate_vram_usage",
]

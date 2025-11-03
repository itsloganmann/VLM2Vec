from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

TensorOrTensors = Union[Tensor, Sequence[Tensor]]


@dataclass
class BatchOutput:
    """Container for per-item embedding tensors and metadata."""

    embeddings: List[Tensor]
    meta: Dict[str, Any]


class BaseRetriever(abc.ABC):
    """Abstract base retriever for multi-vector encoders.

    Subclasses must implement :meth:`_load_model`, :meth:`embed_query`, and
    :meth:`embed_document`. The base class provides convenience helpers for
    precision management, padding, and synthetic dry-run behaviour used in unit
    tests and smoke runs.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        synthetic_dim: int = 128,
        synthetic_tokens: int = 32,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (
            torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        )
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self.dry_run = dry_run
        self.synthetic_dim = synthetic_dim
        self.synthetic_tokens = synthetic_tokens
        self.embedding_dim = synthetic_dim
        if self.dry_run:
            torch.manual_seed(42)
            self.logger.warning("%s running in dry-run mode with synthetic embeddings.", self.__class__.__name__)
        else:
            self._load_model()
        self.logger.info(
            "Initialized retriever %s (model_id=%s, device=%s, dtype=%s, dry_run=%s)",
            self.__class__.__name__,
            model_id,
            self.device,
            self.dtype,
            self.dry_run,
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _load_model(self) -> None:
        """Load underlying model weights and processors."""

    @abc.abstractmethod
    def embed_query(self, batch: Dict[str, Any]) -> BatchOutput:
        """Embed a batch of queries returning per-item tensors."""

    @abc.abstractmethod
    def embed_document(self, batch: Dict[str, Any]) -> BatchOutput:
        """Embed a batch of candidate documents returning per-item tensors."""

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def score(self, query_embeddings: TensorOrTensors, document_embeddings: TensorOrTensors, scorer) -> Tensor:
        """Delegate to scorer callable for computing similarity."""

        return scorer(query_embeddings, document_embeddings)

    def _as_tensor_list(self, embeddings: TensorOrTensors) -> List[Tensor]:
        if isinstance(embeddings, Tensor):
            if embeddings.ndim == 2:
                return [embeddings]
            if embeddings.ndim == 3:
                return [embeddings[i] for i in range(embeddings.size(0))]
        return list(embeddings)

    def _synthetic_embeddings(self, batch_size: int) -> List[Tensor]:
        tokens = self.config.get("synthetic_tokens", self.synthetic_tokens)
        dim = self.config.get("synthetic_dim", self.synthetic_dim)
        emb = torch.randn(batch_size, tokens, dim, device=self.device, dtype=torch.float32)
        return [emb[i] for i in range(batch_size)]

    def _batch_size(self, batch: Dict[str, Any]) -> int:
        for value in batch.values():
            if isinstance(value, (list, tuple)):
                return len(value)
            if isinstance(value, torch.Tensor):
                return value.size(0)
        raise ValueError("Unable to infer batch size from batch keys")

    def pad(self, batch: Sequence[Tensor], pad_multiple: int = 1, value: float = 0.0) -> Tensor:
        """Pad variable-length token sequences to a static shape."""

        max_len = max(t.size(-2) for t in batch)
        if pad_multiple > 1:
            max_len = ((max_len + pad_multiple - 1) // pad_multiple) * pad_multiple
        tensors = list(batch)
        padded = pad_sequence(tensors, batch_first=True, padding_value=value)
        if padded.size(1) < max_len:
            diff = max_len - padded.size(1)
            pad = torch.full((padded.size(0), diff, padded.size(-1)), value, dtype=padded.dtype, device=padded.device)
            padded = torch.cat([padded, pad], dim=1)
        return padded

    def ensure_precision(self, tensor: Tensor) -> Tensor:
        """Move tensor to configured device and dtype."""

        target_dtype = self.dtype
        if target_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            target_dtype = torch.float16
        return tensor.to(device=self.device, dtype=target_dtype)

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------
    def calibrate_batch_size(
        self,
        *,
        probe: Callable[[int], bool],
        start: int,
        limit: int,
        step: int = 1,
    ) -> int:
        """Binary-ish search for the largest batch size satisfying ``probe``.

        ``probe`` should accept an integer batch size and return ``True`` when
        the batch fits in memory. The method increases the batch size lazily and
        returns the last successful value, capturing and logging any CUDA OOM
        errors along the way. When running on CPU or in dry-run mode, it simply
        returns ``start``.
        """

        if self.dry_run or not torch.cuda.is_available():
            return start

        best = start
        current = start
        while current <= limit:
            try:
                if probe(current):
                    best = current
                    current += step
                else:
                    break
            except RuntimeError as exc:  # pragma: no cover - CUDA only
                if "CUDA out of memory" in str(exc):
                    self.logger.warning("OOM detected at batch size %s for %s", current, self.__class__.__name__)
                    break
                raise
        self.logger.info("Calibrated batch size for %s: %s", self.__class__.__name__, best)
        return best


__all__ = ["BaseRetriever", "BatchOutput", "TensorOrTensors"]

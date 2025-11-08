from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoModel

try:
    from transformers import AutoProcessor
except ImportError:  # pragma: no cover - older transformers releases
    AutoProcessor = None  # type: ignore[assignment]

from .base import BaseRetriever, BatchOutput

logger = logging.getLogger(__name__)

MIN_RECOMMENDED_TRANSFORMERS = "4.57.1"


class ColNomicRetriever(BaseRetriever):
    """Wrapper for nomic-ai/colnomic-embed-multimodal-3b."""

    def __init__(
        self,
        model_id: str = "nomic-ai/colnomic-embed-multimodal-3b",
        *,
        output_format: str = "multi",
        fallback_output_format: Optional[str] = "single",
        pooling: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.output_format = output_format
        self.fallback_output_format = fallback_output_format
        self.pooling = pooling or "none"
        super().__init__(model_id=model_id, **kwargs)

    def _load_model(self) -> None:  # pragma: no cover - heavy dependency
        if AutoProcessor is None:
            raise RuntimeError(
                "transformers.AutoProcessor is unavailable. Upgrade transformers to >=%s to load ColNomic retriever."
                % MIN_RECOMMENDED_TRANSFORMERS
            )
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = getattr(self.model.config, "hidden_size", self.synthetic_dim)
        logger.info("Loaded ColNomic retriever with embedding dim %s", self.embedding_dim)

    def _prepare(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        if "texts" in batch:
            inputs["text"] = batch["texts"]
        if "images" in batch:
            inputs["images"] = batch["images"]
        if self.dry_run:
            return {"batch_size": self._batch_size(batch)}
        processed = self.processor(**inputs, return_tensors="pt")
        processed = {k: v.to(self.device) for k, v in processed.items()}
        return processed

    def _select_output(self, outputs: Any, vector_mode: str) -> torch.Tensor:
        preferred_keys = [
            f"{vector_mode}_token_embeds",
            f"{vector_mode}_embeds",
            "token_embeds",
            "last_hidden_state",
        ]
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, dict):
            for key in preferred_keys:
                tensor = outputs.get(key)
                if isinstance(tensor, torch.Tensor):
                    return tensor
            for key in outputs:
                tensor = outputs[key]
                if isinstance(tensor, torch.Tensor):
                    return tensor
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                if isinstance(item, torch.Tensor):
                    return item
        raise RuntimeError("Unable to extract embeddings from ColNomic output")

    def _maybe_pool(self, embeds: torch.Tensor) -> torch.Tensor:
        if self.output_format == "single":
            if embeds.ndim == 3:
                return embeds.mean(dim=-2)
            return embeds
        return embeds

    def embed_query(self, batch: Dict[str, Any]) -> BatchOutput:
        batch_size = self._batch_size(batch)
        if self.dry_run:
            embeddings = self._synthetic_embeddings(batch_size)
            if self.output_format == "single":
                embeddings = [emb.mean(dim=0) for emb in embeddings]
            return BatchOutput(embeddings=embeddings, meta={"synthetic": True})
        processed = self._prepare(batch)
        with torch.no_grad():
            outputs = self.model.encode_text(**processed)
        embeds = self._select_output(outputs, vector_mode="text").to(dtype=self.dtype)
        if self.output_format == "single" and embeds.ndim == 3:
            embeds = embeds.mean(dim=1)
            chunks = [embeds[i] for i in range(embeds.size(0))]
        else:
            if embeds.ndim == 2:
                embeds = embeds.unsqueeze(1)
            chunks = [embeds[i] for i in range(embeds.size(0))]
        return BatchOutput(embeddings=chunks, meta={"tokens": [c.size(-2) if c.ndim == 2 else 1 for c in chunks]})

    def embed_document(self, batch: Dict[str, Any]) -> BatchOutput:
        batch_size = self._batch_size(batch)
        if self.dry_run:
            embeddings = self._synthetic_embeddings(batch_size)
            if self.output_format == "single":
                embeddings = [emb.mean(dim=0) for emb in embeddings]
            return BatchOutput(embeddings=embeddings, meta={"synthetic": True})
        processed = self._prepare(batch)
        with torch.no_grad():
            outputs = self.model.encode_image(**processed)
        embeds = self._select_output(outputs, vector_mode="image").to(dtype=self.dtype)
        if self.output_format == "single" and embeds.ndim == 3:
            embeds = embeds.mean(dim=1)
            chunks = [embeds[i] for i in range(embeds.size(0))]
        else:
            if embeds.ndim == 2:
                embeds = embeds.unsqueeze(1)
            chunks = [embeds[i] for i in range(embeds.size(0))]
        return BatchOutput(embeddings=chunks, meta={"tokens": [c.size(-2) if c.ndim == 2 else 1 for c in chunks]})


__all__ = ["ColNomicRetriever"]

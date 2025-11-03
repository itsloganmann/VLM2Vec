from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from transformers import AutoModel, AutoProcessor

from .base import BaseRetriever, BatchOutput

logger = logging.getLogger(__name__)


class NemoRetriever(BaseRetriever):
    """Wrapper for nvidia/llama-nemoretriever-colembed-3b-v1."""

    def __init__(
        self,
        model_id: str = "nvidia/llama-nemoretriever-colembed-3b-v1",
        *,
        image_resolution: int = 448,
        patch_budget: int | None = None,
        pooling: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.image_resolution = image_resolution
        self.patch_budget = patch_budget
        self.pooling = pooling or "none"
        super().__init__(model_id=model_id, **kwargs)

    def _load_model(self) -> None:  # pragma: no cover - heavy dependency
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = getattr(self.model.config, "hidden_size", self.synthetic_dim)
        logger.info("Loaded Nemo retriever with embedding dim %s", self.embedding_dim)

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

    def _postprocess(self, outputs: Any, key: str) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                if isinstance(item, torch.Tensor):
                    return item
        if isinstance(outputs, dict):
            if key in outputs and isinstance(outputs[key], torch.Tensor):
                return outputs[key]
            for fallback in ["image_embeds", "text_embeds", "patch_embeds", "last_hidden_state"]:
                tensor = outputs.get(fallback)
                if isinstance(tensor, torch.Tensor):
                    return tensor
        raise RuntimeError("Unable to extract embeddings from Nemo retriever output")

    def embed_query(self, batch: Dict[str, Any]) -> BatchOutput:
        batch_size = self._batch_size(batch)
        if self.dry_run:
            embeddings = self._synthetic_embeddings(batch_size)
            return BatchOutput(embeddings=embeddings, meta={"synthetic": True})
        processed = self._prepare(batch)
        with torch.no_grad():
            outputs = self.model.encode_text(**processed)
        embeds = self._postprocess(outputs, key="query_tokens").to(dtype=self.dtype)
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(1)
        chunks = [embeds[i] for i in range(embeds.size(0))]
        return BatchOutput(embeddings=chunks, meta={"tokens": [c.size(-2) for c in chunks]})

    def embed_document(self, batch: Dict[str, Any]) -> BatchOutput:
        batch_size = self._batch_size(batch)
        if self.dry_run:
            embeddings = self._synthetic_embeddings(batch_size)
            return BatchOutput(embeddings=embeddings, meta={"synthetic": True})
        processed = self._prepare(batch)
        with torch.no_grad():
            outputs = self.model.encode_image(**processed)
        embeds = self._postprocess(outputs, key="image_tokens").to(dtype=self.dtype)
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(1)
        chunks = [embeds[i] for i in range(embeds.size(0))]
        return BatchOutput(embeddings=chunks, meta={"tokens": [c.size(-2) for c in chunks]})


__all__ = ["NemoRetriever"]

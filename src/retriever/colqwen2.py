from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
)

from .base import BaseRetriever, BatchOutput

logger = logging.getLogger(__name__)


class ColQwen2Retriever(BaseRetriever):
    """vidore/colqwen2.5-v0.2 multi-vector retriever wrapper."""

    def __init__(
        self,
        model_id: str = "vidore/colqwen2.5-v0.2",
        *,
        image_resolution: int = 560,
        patch_budget: Optional[int] = None,
        pooling: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.image_resolution = image_resolution
        self.patch_budget = patch_budget
        self.pooling = pooling or "none"
        super().__init__(model_id=model_id, **kwargs)

    def _load_model(self) -> None:  # pragma: no cover - heavy dependency
        self.processor = self._load_processor()
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = getattr(self.model.config, "hidden_size", self.synthetic_dim)
        logger.info("Loaded ColQwen2 model with embedding dim %s", self.embedding_dim)

    def _load_processor(self) -> Any:  # pragma: no cover - heavy dependency
        try:
            processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            logger.info("Loaded AutoProcessor for %s", self.model_id)
            return processor
        except ValueError as err:
            logger.warning("AutoProcessor unavailable for %s: %s", self.model_id, err)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        image_processor: Optional[Any] = None

        loaders = (AutoImageProcessor, AutoFeatureExtractor)
        last_error: Optional[Exception] = None
        for loader in loaders:
            try:
                image_processor = loader.from_pretrained(self.model_id, trust_remote_code=True)
                logger.info("Loaded %s for %s", loader.__name__, self.model_id)
                break
            except Exception as exc:  # pragma: no cover - loader availability
                last_error = exc
        if image_processor is None:
            raise RuntimeError(
                "Failed to construct image processor for ColQwen2; last error: %s" % last_error
            )

        class _HybridProcessor:
            def __init__(self, tok, img):
                self.tokenizer = tok
                self.image_processor = img

            def __call__(self, text: Optional[Any] = None, images: Optional[Any] = None, return_tensors: str = "pt"):
                outputs: Dict[str, Any] = {}
                if text is not None:
                    if isinstance(text, str):
                        text_inputs: Any = [text]
                    else:
                        text_inputs = text
                    outputs.update(
                        self.tokenizer(
                            text_inputs,
                            padding=True,
                            truncation=True,
                            return_tensors=return_tensors,
                        )
                    )
                if images is not None:
                    outputs.update(
                        self.image_processor(
                            images=images,
                            return_tensors=return_tensors,
                        )
                    )
                if not outputs:
                    raise ValueError("ColQwen2 processor received no inputs")
                return outputs

        logger.info("Constructed hybrid processor (tokenizer + image processor) for %s", self.model_id)
        return _HybridProcessor(tokenizer, image_processor)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
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

    def _postprocess(self, outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                if isinstance(item, torch.Tensor):
                    return item
        if isinstance(outputs, dict):
            for key in ["image_embeds", "text_embeds", "patch_embeds", "last_hidden_state"]:
                tensor = outputs.get(key)
                if isinstance(tensor, torch.Tensor):
                    return tensor
        raise RuntimeError("Unable to extract embeddings from ColQwen2 output")

    def _run_forward(self, processed: Dict[str, Any], kind: str) -> torch.Tensor:
        candidates = [
            f"encode_{kind}",
            f"{kind}_embeddings",
            f"{kind}_features",
            "forward",
        ]
        for name in candidates:
            if hasattr(self.model, name):
                outputs = getattr(self.model, name)(**processed)
                result = self._postprocess(outputs)
                return result
        outputs = self.model(**processed)
        return self._postprocess(outputs)

    def _ensure_sequence_dim(self, embeds: torch.Tensor) -> torch.Tensor:
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(1)
        return embeds

    def embed_query(self, batch: Dict[str, Any]) -> BatchOutput:
        batch_size = self._batch_size(batch)
        if self.dry_run:
            embeddings = self._synthetic_embeddings(batch_size)
            return BatchOutput(embeddings=embeddings, meta={"synthetic": True})
        processed = self._prepare(batch)
        with torch.no_grad():
            embeds = self._run_forward(processed, kind="query")
        embeds = self._ensure_sequence_dim(embeds).to(dtype=self.dtype)
        chunks = [embeds[i] for i in range(embeds.size(0))]
        return BatchOutput(embeddings=chunks, meta={"tokens": [c.size(-2) for c in chunks]})

    def embed_document(self, batch: Dict[str, Any]) -> BatchOutput:
        batch_size = self._batch_size(batch)
        if self.dry_run:
            embeddings = self._synthetic_embeddings(batch_size)
            return BatchOutput(embeddings=embeddings, meta={"synthetic": True})
        processed = self._prepare(batch)
        with torch.no_grad():
            embeds = self._run_forward(processed, kind="document")
        embeds = self._ensure_sequence_dim(embeds).to(dtype=self.dtype)
        chunks = [embeds[i] for i in range(embeds.size(0))]
        return BatchOutput(embeddings=chunks, meta={"tokens": [c.size(-2) for c in chunks]})


__all__ = ["ColQwen2Retriever"]

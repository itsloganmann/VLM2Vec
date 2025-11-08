from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import transformers
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, AutoTokenizer

try:
	from transformers import AutoProcessor
except ImportError:  # pragma: no cover - older transformers releases
	AutoProcessor = None  # type: ignore[assignment]
try:
	from transformers.dynamic_module_utils import get_class_from_dynamic_module
except ImportError:  # pragma: no cover - older transformers releases
	get_class_from_dynamic_module = None  # type: ignore[assignment]

try:
	from transformers import AutoImageProcessor
except ImportError:  # pragma: no cover - alias removed in newer releases
	AutoImageProcessor = None  # type: ignore[assignment]

try:  # transformers>=4.42
	from transformers import Qwen2VLImageProcessor
except ImportError:  # pragma: no cover - older transformers versions
	Qwen2VLImageProcessor = None  # type: ignore[assignment]

from .base import BaseRetriever, BatchOutput

MIN_RECOMMENDED_TRANSFORMERS = "4.57.1"


logger = logging.getLogger(__name__)


class _HybridProcessor:
	"""Callable wrapper combining tokenizer and image processor."""

	def __init__(self, tokenizer: Any, image_processor: Any) -> None:
		self.tokenizer = tokenizer
		self.image_processor = image_processor

	def __call__(
		self,
		text: Optional[Any] = None,
		images: Optional[Any] = None,
		return_tensors: str = "pt",
		**kwargs: Any,
	) -> Dict[str, Any]:
		outputs: Dict[str, Any] = {}

		if text is not None:
			text_inputs: Any = [text] if isinstance(text, str) else text
			outputs.update(
				self.tokenizer(
					text_inputs,
					padding=True,
					truncation=True,
					return_tensors=return_tensors,
					**kwargs,
				)
			)

		if images is not None:
			outputs.update(
				self.image_processor(
					images=images,
					return_tensors=return_tensors,
					**kwargs,
				)
			)

		if not outputs:
			raise ValueError("ColQwen2 processor received no inputs")

		return outputs


class ColQwen2Retriever(BaseRetriever):
	"""`vidore/colqwen2.5-v0.2` multi-vector retriever wrapper."""

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

	# ------------------------------------------------------------------
	# Model & processor loading
	# ------------------------------------------------------------------
	def _load_model(self) -> None:  # pragma: no cover - heavy dependency
		self.processor = self._load_processor()
		try:
			self.model = AutoModel.from_pretrained(
				self.model_id,
				trust_remote_code=True,
				torch_dtype=self.dtype,
			)
		except ValueError as err:
			self.model = self._load_model_with_dynamic_module(err)
		self.model.to(self.device)
		self.model.eval()
		self.embedding_dim = getattr(self.model.config, "hidden_size", self.synthetic_dim)
		logger.info("Loaded ColQwen2 model with embedding dim %s", self.embedding_dim)

	def _load_model_with_dynamic_module(self, original_error: Exception) -> Any:
		message = str(original_error)
		if "Qwen2_5_VLModel" not in message:
			raise original_error
		if get_class_from_dynamic_module is None:
			raise RuntimeError(
				"transformers %s is missing Qwen2.5-VL support and cannot load custom code dynamically. "
				"Upgrade transformers to >=%s."
				% (transformers.__version__, MIN_RECOMMENDED_TRANSFORMERS)
			) from original_error
		config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
		auto_map = getattr(config, "auto_map", {}) or {}
		class_reference = auto_map.get("AutoModel")
		if isinstance(class_reference, (list, tuple)):
			class_reference = class_reference[0]
		if not class_reference:
			raise RuntimeError(
				"Model %s does not expose an AutoModel auto_map entry. "
				"Upgrade transformers to >=%s."
				% (self.model_id, MIN_RECOMMENDED_TRANSFORMERS)
			) from original_error
		logger.warning(
			"AutoModel mapping for Qwen2.5-VL is unavailable in transformers %s; using dynamic module %s.",
			transformers.__version__,
			class_reference,
		)
		try:
			model_cls = get_class_from_dynamic_module(
				class_reference,
				self.model_id,
				revision=getattr(config, "_commit_hash", None),
				trust_remote_code=True,
			)
		except Exception as load_err:  # pragma: no cover - network/filesystem dependent
			raise RuntimeError(
				"Unable to dynamically import %s from %s. Upgrade transformers to >=%s or pin a revision with bundled code."
				% (class_reference, self.model_id, MIN_RECOMMENDED_TRANSFORMERS)
			) from load_err
		model = model_cls.from_pretrained(
			self.model_id,
			trust_remote_code=True,
			torch_dtype=self.dtype,
		)
		return model

	def _load_processor(self) -> Any:  # pragma: no cover - heavy dependency
		if AutoProcessor is None:
			logger.warning(
				"transformers.AutoProcessor is unavailable; falling back to hybrid tokenizer/image processor. "
				"Upgrade transformers to >=%s for full ColQwen2 support.",
				MIN_RECOMMENDED_TRANSFORMERS,
			)
			return self._build_hybrid_processor()
		try:
			processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
			if self._processor_has_vision(processor):
				logger.info("Loaded AutoProcessor for %s", self.model_id)
				return processor
			tokenizer = getattr(processor, "tokenizer", processor)
			logger.warning(
				"AutoProcessor for %s lacks vision component (type=%s); constructing hybrid.",
				self.model_id,
				type(processor).__name__,
			)
			return self._build_hybrid_processor(tokenizer=tokenizer)
		except Exception as err:
			logger.warning(
				"AutoProcessor unavailable for %s: %s. Falling back to hybrid tokenizer/image processor.",
				self.model_id,
				err,
			)
			return self._build_hybrid_processor()

	def _processor_has_vision(self, processor: Any) -> bool:
		return any(
			hasattr(processor, attr) and getattr(processor, attr) is not None
			for attr in ("image_processor", "vision_processor", "image_processors")
		)

	def _load_tokenizer(self) -> Any:
		errors: List[Exception] = []
		for use_fast in (True, False):
			try:
				tokenizer = AutoTokenizer.from_pretrained(
					self.model_id,
					trust_remote_code=True,
					use_fast=use_fast,
				)
				logger.info(
					"Loaded %s tokenizer for %s (use_fast=%s)",
					type(tokenizer).__name__,
					self.model_id,
					use_fast,
				)
				return tokenizer
			except Exception as exc:  # pragma: no cover - depends on HF/tokenizers versions
				errors.append(exc)
				logger.warning(
					"Failed to load tokenizer for %s with use_fast=%s: %s",
					self.model_id,
					use_fast,
					exc,
				)
		last_error = errors[-1] if errors else RuntimeError("unknown tokenizer error")
		raise RuntimeError(
			f"Failed to construct tokenizer for ColQwen2 ({self.model_id}); last error: {last_error}"
		)

	def _build_hybrid_processor(self, tokenizer: Optional[Any] = None) -> _HybridProcessor:
		if tokenizer is None:
			tokenizer = self._load_tokenizer()

		image_processor = self._load_image_processor()
		logger.info("Constructed hybrid processor (tokenizer + image processor) for %s", self.model_id)
		return _HybridProcessor(tokenizer, image_processor)

	def _load_image_processor(self) -> Any:
		loaders = tuple(loader for loader in (AutoImageProcessor, AutoFeatureExtractor) if loader is not None)
		if not loaders:
			raise RuntimeError("transformers does not provide an image processor loader (AutoImageProcessor/AutoFeatureExtractor)")
		last_error: Optional[Exception] = None

		for loader in loaders:
			try:
				image_processor = loader.from_pretrained(self.model_id, trust_remote_code=True)
				logger.info("Loaded %s for %s", loader.__name__, self.model_id)
				return image_processor
			except Exception as exc:  # pragma: no cover - loader availability
				last_error = exc
				if "image_processor_type" in str(exc):
					logger.debug(
						"Failed to load %s for %s due to missing image_processor_type, will try Qwen2VLImageProcessor.",
						loader.__name__,
						self.model_id,
					)
					break
				logger.debug("Failed to load %s for %s: %s", loader.__name__, self.model_id, exc)

		if Qwen2VLImageProcessor is not None:
			try:
				image_processor = Qwen2VLImageProcessor.from_pretrained(self.model_id)
				logger.info("Loaded Qwen2VLImageProcessor fallback for %s", self.model_id)
				return image_processor
			except Exception as exc:  # pragma: no cover - requires HF assets
				last_error = exc
				logger.error(
					"Failed to load Qwen2VLImageProcessor fallback for %s: %s", self.model_id, exc
				)

		raise RuntimeError(
			f"Failed to construct image processor for ColQwen2 ({self.model_id}); last error: {last_error}"
		)

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
		return {k: v.to(self.device) for k, v in processed.items()}

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
				return self._postprocess(outputs)
		outputs = self.model(**processed)
		return self._postprocess(outputs)

	def _ensure_sequence_dim(self, embeds: torch.Tensor) -> torch.Tensor:
		if embeds.ndim == 2:
			embeds = embeds.unsqueeze(1)
		return embeds

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
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

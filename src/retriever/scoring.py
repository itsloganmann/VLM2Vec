from __future__ import annotations

import logging
from typing import Sequence, Union, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

TensorLike = Union[Tensor, Sequence[Tensor]]


def _ensure_list(tensors: TensorLike) -> List[Tensor]:
    if isinstance(tensors, Tensor):
        if tensors.ndim == 2:
            return [tensors]
        if tensors.ndim == 3:
            return [tensors[i] for i in range(tensors.size(0))]
    return list(tensors)


def _normalize(t: Tensor, eps: float = 1e-6) -> Tensor:
    return F.normalize(t.float(), p=2, dim=-1, eps=eps)


def maxsim_score(query_tokens: TensorLike, document_tokens: TensorLike) -> Tensor:
    """Compute MaxSim late interaction score for each item in batch."""

    q_list = _ensure_list(query_tokens)
    d_list = _ensure_list(document_tokens)
    if len(q_list) != len(d_list):
        raise ValueError("MaxSim expects equal batch sizes for queries and documents")
    scores: List[Tensor] = []
    for q_tokens, d_tokens in zip(q_list, d_list):
        q = _normalize(q_tokens)
        d = _normalize(d_tokens)
        sim = torch.matmul(q, d.transpose(-1, -2))
        max_sim, _ = torch.max(sim, dim=-1)
        scores.append(torch.sum(max_sim))
    return torch.stack(scores)


def dot_score(query_vecs: TensorLike, doc_vecs: TensorLike) -> Tensor:
    q_list = _ensure_list(query_vecs)
    d_list = _ensure_list(doc_vecs)
    if len(q_list) != len(d_list):
        raise ValueError("Dot score expects equal batch sizes")
    scores = []
    for q_vec, d_vec in zip(q_list, d_list):
        q = _normalize(q_vec)
        d = _normalize(d_vec)
        scores.append(torch.sum(q * d))
    return torch.stack(scores)


def hybrid_score(query_embeds: TensorLike, doc_embeds: TensorLike) -> Tensor:
    """Hybrid scorer dispatching to MaxSim or dot-product based on rank."""

    if isinstance(query_embeds, Tensor) and query_embeds.ndim == 2:
        return dot_score(query_embeds, doc_embeds)
    if isinstance(doc_embeds, Tensor) and doc_embeds.ndim == 2:
        return dot_score(query_embeds, doc_embeds)
    return maxsim_score(query_embeds, doc_embeds)


def pad_to_static(batch: Sequence[Tensor], pad_multiple: int = 1, value: float = 0.0) -> Tensor:
    prepared = []
    for tensor in batch:
        if tensor.ndim == 1:
            prepared.append(tensor.unsqueeze(0))
        elif tensor.ndim == 2:
            prepared.append(tensor)
        elif tensor.ndim == 3 and tensor.size(0) == 1:
            prepared.append(tensor.squeeze(0))
        else:
            raise ValueError("Unsupported tensor rank for pad_to_static")
    max_len = max(t.size(-2) for t in prepared)
    if pad_multiple > 1:
        max_len = ((max_len + pad_multiple - 1) // pad_multiple) * pad_multiple
    padded = pad_sequence(list(prepared), batch_first=True, padding_value=value)
    if padded.size(1) < max_len:
        diff = max_len - padded.size(1)
        pad = torch.full((padded.size(0), diff, padded.size(-1)), value, dtype=padded.dtype, device=padded.device)
        padded = torch.cat([padded, pad], dim=1)
    return padded


__all__ = ["maxsim_score", "dot_score", "hybrid_score", "pad_to_static"]

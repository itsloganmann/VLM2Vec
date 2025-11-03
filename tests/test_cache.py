from pathlib import Path

import torch

from retriever.memory import CacheManager, ResumeManager, estimate_vram_usage


def test_cache_roundtrip(tmp_path):
    cache = CacheManager(tmp_path, dtype=torch.float32, device=torch.device("cpu"))
    tensor = torch.randn(3, 4)
    meta = {"task": "unit", "shard": 0, "size": 1}
    cache.save(meta, tensor, extra={"ids": ["doc-1"]})
    payload = cache.load(meta)
    loaded = payload["embeddings"]
    cosine = torch.nn.functional.cosine_similarity(tensor.flatten(), loaded.flatten(), dim=0)
    assert cosine > 0.999


def test_estimate_vram_usage():
    tensors = [torch.ones(2, 3, 4, dtype=torch.float32)]
    usage = estimate_vram_usage(tensors)
    assert usage > 0


def test_resume_manager(tmp_path):
    resume_path = tmp_path / "resume.yaml"
    manager = ResumeManager(resume_path)
    manager.update("model", "task", "stage", {"status": "ok"})
    state = manager.last()
    assert state["model"]["task"]["stage"]["status"] == "ok"

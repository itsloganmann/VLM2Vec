import json
from pathlib import Path

import yaml

from evaluation.multi_vector_eval import run_mmeb_eval


def build_mock_config(tmp_path: Path) -> Path:
    output_root = tmp_path / "outputs"
    config = {
        "run": {
            "name": "unit-test",
            "precision": "bf16",
            "quick_subset": {"enabled": False},
            "output_root": str(output_root),
            "cache_root": str(output_root / "cache"),
            "log_root": str(output_root / "logs"),
            "metrics_root": str(output_root / "metrics"),
            "profiler_root": str(output_root / "profiler"),
            "resume_state": str(output_root / "resume_state.yaml"),
        },
        "models": [
            {
                "alias": "nemoretriever",
                "id": "nvidia/llama-nemoretriever-colembed-3b-v1",
                "implementation": "retriever.nemoretriever.NemoRetriever",
                "dtype": "bf16",
                "query_batch_size": 2,
                "document_batch_size": 2,
            }
        ],
        "data": {
            "repo_id": "mock",
            "local_root": str(tmp_path / "datasets"),
            "tasks": [
                {
                    "name": "toy_task",
                    "type": "image_text",
                    "subset": "toy",
                    "split": "validation",
                    "id_column": "id",
                    "query_column": "query",
                    "image_column": None,
                    "positive_column": "positive_ids",
                    "candidate_column": "candidates",
                    "max_candidates": 4,
                    "mock_examples": [
                        {
                            "id": "q1",
                            "query": "cat",
                            "candidates": [
                                {"id": "doc1", "text": "a cat"},
                                {"id": "doc2", "text": "a dog"},
                            ],
                            "positive_ids": ["doc1"],
                        },
                        {
                            "id": "q2",
                            "query": "dog",
                            "candidates": [
                                {"id": "doc1", "text": "a cat"},
                                {"id": "doc2", "text": "a dog"},
                            ],
                            "positive_ids": ["doc2"],
                        },
                    ],
                }
            ],
        },
        "scoring": {
            "shard_size": 2,
            "chunk_size": 2,
            "pad_to_static": True,
            "pad_multiple": 2,
            "dtype": "bf16",
            "warn_if_fp32": True,
            "memory_guard_gb": 1,
        },
        "monitoring": {"profiler": False, "memory_summary": False, "log_interval": 1},
        "safety": {"max_wall_clock_minutes": 5, "max_gpu_memory_gb": 2, "min_precision": "bf16"},
        "artifacts": {"save_reports": False},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.safe_dump(config, fh)
    return config_path


def test_run_mmeb_eval_smoke(tmp_path):
    config_path = build_mock_config(tmp_path)
    metrics = run_mmeb_eval(config_path, dry_run=True)
    assert "nemoretriever" in metrics
    toy_metrics = metrics["nemoretriever"]["toy_task"]
    assert 0.0 <= toy_metrics["precision@1"] <= 1.0
    assert toy_metrics["total_queries"] == 2

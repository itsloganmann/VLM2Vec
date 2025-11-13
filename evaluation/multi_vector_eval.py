from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import yaml

from packaging import version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    src_path = str(SRC_ROOT)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from retriever import BaseRetriever, scoring
from retriever.memory import CacheManager, ResumeManager, chunk_iterable, estimate_vram_usage

try:  # pragma: no cover - huggingface-hub optional in tests
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError
except Exception:  # pragma: no cover - allow unit tests without hub
    snapshot_download = None  # type: ignore
    GatedRepoError = RepositoryNotFoundError = HfHubHTTPError = LocalEntryNotFoundError = Exception  # type: ignore

try:  # pragma: no cover - optional dependency
    from datasets import load_dataset
except Exception:  # pragma: no cover - allow tests without datasets
    load_dataset = None  # type: ignore

LOGGER = logging.getLogger(__name__)

MIN_TRANSFORMERS_VERSION = "4.57.1"
MIN_TORCH_VERSION = "2.2.1"
EXPECTED_GPU_SUBSTRING = "A100"
DEFAULT_MIN_DISK_GB = 40.0


@dataclass
class RunConfig:
    name: str
    precision: str = "bf16"
    quick_subset: Dict[str, Any] = field(default_factory=dict)
    output_root: Path = Path("outputs")
    cache_root: Path = Path("outputs/cache")
    log_root: Path = Path("outputs/logs")
    metrics_root: Path = Path("outputs/metrics")
    profiler_root: Path = Path("outputs/profiler")
    resume_state: Path = Path("outputs/resume_state.yaml")
    pad_to_static: bool = True
    model_snapshot_root: Path = Path("outputs/model_snapshots")


@dataclass
class ModelConfig:
    alias: str
    id: str
    implementation: str
    vector_mode: str = "multi"
    dtype: str = "bf16"
    image_resolution: int = 448
    patch_budget: Optional[int] = None
    query_batch_size: int = 4
    document_batch_size: int = 8
    calibration: bool = True
    pooling: Optional[str] = None
    output_format: Optional[str] = None
    fallback_output_format: Optional[str] = None
    snapshot_revision: Optional[str] = None
    snapshot_allow_patterns: Optional[List[str]] = None
    snapshot_ignore_patterns: Optional[List[str]] = None


@dataclass
class TaskConfig:
    name: str
    type: str
    subset: str
    split: str
    id_column: str
    query_column: str
    image_column: Optional[str]
    positive_column: str
    candidate_column: str
    max_candidates: int
    mock_examples: Optional[List[Dict[str, Any]]] = None


@dataclass
class DataConfig:
    repo_id: str
    local_root: Path
    download: bool = True
    streaming: bool = False
    num_workers: int = 4
    quick_sample_per_task: int = 32
    image_column: str = "image"
    text_column: str = "text"
    tasks: List[TaskConfig] = field(default_factory=list)


@dataclass
class ScoringConfig:
    shard_size: int = 256
    chunk_size: int = 32
    pad_to_static: bool = True
    pad_multiple: int = 8
    dtype: str = "bf16"
    warn_if_fp32: bool = True
    memory_guard_gb: float = 36.0
    pad_strategy: str = "static"


@dataclass
class MonitoringConfig:
    profiler: bool = False
    profiler_schedule: str = "warmup=1,active=2,wait=1"
    memory_summary: bool = True
    log_interval: int = 20
    record_batch_stats: bool = True


@dataclass
class SafetyConfig:
    max_wall_clock_minutes: int = 180
    max_gpu_memory_gb: float = 38.0
    min_precision: str = "bf16"
    allow_fallback_precision: bool = True


@dataclass
class ArtifactConfig:
    save_reports: bool = True
    json_path: Path = Path("outputs/metrics/mmeb_metrics.json")
    csv_path: Path = Path("outputs/metrics/mmeb_metrics.csv")
    zip_bundle: Path = Path("outputs/mmeb_reports.zip")


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    return value


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as fh:
        config = yaml.safe_load(fh)
    return _resolve_env(config)


def parse_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    run = RunConfig(**raw.get("run", {}))
    run.output_root = Path(run.output_root)
    run.cache_root = Path(run.cache_root)
    run.log_root = Path(run.log_root)
    run.metrics_root = Path(run.metrics_root)
    run.profiler_root = Path(run.profiler_root)
    run.resume_state = Path(run.resume_state)
    run.model_snapshot_root = Path(run.model_snapshot_root)
    data = raw.get("data", {})
    tasks = [TaskConfig(**task) for task in data.get("tasks", [])]
    data_cfg = DataConfig(tasks=tasks, **{k: v for k, v in data.items() if k != "tasks"})
    data_cfg.local_root = Path(data_cfg.local_root)
    models = [ModelConfig(**model) for model in raw.get("models", [])]
    scoring_cfg = ScoringConfig(**raw.get("scoring", {}))
    monitoring_cfg = MonitoringConfig(**raw.get("monitoring", {}))
    safety_cfg = SafetyConfig(**raw.get("safety", {}))
    artifact_cfg = ArtifactConfig(**raw.get("artifacts", {}))
    memory_cfg = raw.get("memory", {})
    return {
        "run": run,
        "data": data_cfg,
        "models": models,
        "scoring": scoring_cfg,
        "monitoring": monitoring_cfg,
        "safety": safety_cfg,
        "artifacts": artifact_cfg,
        "memory": memory_cfg,
    }


def _bytes_to_gb(count: int) -> float:
    return count / (1024 ** 3)


def _materialize_disk_summary(path: Path) -> Dict[str, float]:
    usage = shutil.disk_usage(path)
    return {
        "total_gb": _bytes_to_gb(usage.total),
        "used_gb": _bytes_to_gb(usage.used),
        "free_gb": _bytes_to_gb(usage.free),
    }


def _validate_environment(run_cfg: RunConfig, *, strict: bool = True) -> None:
    import transformers

    torch_version = version.parse(torch.__version__)
    transformers_version = version.parse(transformers.__version__)
    if torch_version < version.parse(MIN_TORCH_VERSION):
        message = (
            f"Detected torch {torch.__version__}; minimum supported version is {MIN_TORCH_VERSION}. "
            "Refer to https://pytorch.org/get-started/locally/ for upgrade instructions."
        )
        if strict:
            raise RuntimeError(message)
        LOGGER.warning(message)
    if transformers_version < version.parse(MIN_TRANSFORMERS_VERSION):
        message = (
            f"Detected transformers {transformers.__version__}; minimum supported version is {MIN_TRANSFORMERS_VERSION}. "
            "See https://huggingface.co/docs/transformers/installation for upgrade guidance."
        )
        if strict:
            raise RuntimeError(message)
        LOGGER.warning(message)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    except AttributeError:
        LOGGER.debug("TF32 toggles unavailable in this torch build")
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        LOGGER.debug("torch.set_float32_matmul_precision unavailable; continuing")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        LOGGER.info("Primary GPU: %s", gpu_name)
        if EXPECTED_GPU_SUBSTRING not in gpu_name:
            LOGGER.warning("Expected an NVIDIA A100 GPU but detected %s", gpu_name)
    else:
        LOGGER.warning("CUDA is not available; evaluation will fall back to CPU and run significantly slower")
    snapshot_home = run_cfg.model_snapshot_root
    snapshot_home.mkdir(parents=True, exist_ok=True)
    summary = _materialize_disk_summary(snapshot_home)
    if summary["free_gb"] < DEFAULT_MIN_DISK_GB:
        message = (
            f"Insufficient free disk space at {snapshot_home} ({summary['free_gb']:.1f} GiB). "
            "Allocate additional storage or point `run.model_snapshot_root` to a larger volume."
        )
        if strict:
            raise RuntimeError(message)
        LOGGER.warning(message)
    LOGGER.info(
        "Disk usage at %s -> total %.1f GiB, used %.1f GiB, free %.1f GiB",
        snapshot_home,
        summary["total_gb"],
        summary["used_gb"],
        summary["free_gb"],
    )
    os.environ.setdefault("HF_HOME", str(snapshot_home / "hf_home"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(snapshot_home / "hub_cache"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _prefetch_model_assets(models: Sequence[ModelConfig], snapshot_root: Path) -> Dict[str, Path]:
    if snapshot_download is None:
        LOGGER.warning("huggingface_hub is unavailable; model snapshots will be fetched lazily")
        return {}
    snapshot_root.mkdir(parents=True, exist_ok=True)
    resolved: Dict[str, Path] = {}
    for model_cfg in models:
        target_dir = snapshot_root / model_cfg.alias
        allow_patterns = model_cfg.snapshot_allow_patterns
        ignore_patterns = model_cfg.snapshot_ignore_patterns
        kwargs: Dict[str, Any] = {
            "repo_id": model_cfg.id,
            "local_dir": str(target_dir),
            "local_dir_use_symlinks": False,
        }
        if model_cfg.snapshot_revision:
            kwargs["revision"] = model_cfg.snapshot_revision
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        if ignore_patterns:
            kwargs["ignore_patterns"] = ignore_patterns
        try:
            snapshot_path = Path(snapshot_download(**kwargs))
            resolved[model_cfg.alias] = snapshot_path
            size_gb = _bytes_to_gb(sum(f.stat().st_size for f in snapshot_path.rglob("*") if f.is_file()))
            LOGGER.info("Prefetched %s to %s (%.2f GiB)", model_cfg.id, snapshot_path, size_gb)
        except GatedRepoError as exc:
            LOGGER.error("Access to %s requires authentication: %s", model_cfg.id, exc)
            raise
        except RepositoryNotFoundError as exc:
            LOGGER.error("Model repository %s not found. Verify the identifier.", model_cfg.id)
            raise
        except (HfHubHTTPError, LocalEntryNotFoundError) as exc:
            LOGGER.warning(
                "Prefetch for %s encountered %s; the run will rely on on-demand downloads.",
                model_cfg.id,
                exc,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("Unexpected error while downloading %s: %s", model_cfg.id, exc)
    return resolved

def prepare_run_folders(run: RunConfig) -> None:
    for path in [
        run.output_root,
        run.cache_root,
        run.log_root,
        run.metrics_root,
        run.profiler_root,
        run.resume_state.parent,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def instantiate_retriever(
    model_cfg: ModelConfig,
    *,
    dry_run: bool = False,
    snapshot_paths: Optional[Dict[str, Path]] = None,
) -> BaseRetriever:
    module_name, _, attr = model_cfg.implementation.rpartition(".")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        missing_symbol = "AutoProcessor" if "AutoProcessor" in str(exc) else None
        if missing_symbol:
            raise RuntimeError(
                "Failed to load %s due to missing transformers.%s. "
                "Upgrade transformers to >=%s and rerun the installation cell."
                % (model_cfg.alias, missing_symbol, MIN_TRANSFORMERS_VERSION)
            ) from exc
        raise
    cls = getattr(module, attr)
    resolved_model_id = model_cfg.id
    if snapshot_paths:
        local_snapshot = snapshot_paths.get(model_cfg.alias)
        if local_snapshot is not None:
            resolved_model_id = str(local_snapshot)
            LOGGER.info("Loading %s from local snapshot %s", model_cfg.alias, resolved_model_id)
    kwargs: Dict[str, Any] = {
        "model_id": resolved_model_id,
        "dtype": torch.bfloat16 if model_cfg.dtype == "bf16" else torch.float16,
        "dry_run": dry_run,
    }
    optional = {
        "image_resolution": model_cfg.image_resolution,
        "patch_budget": model_cfg.patch_budget,
        "pooling": model_cfg.pooling,
        "output_format": model_cfg.output_format,
        "fallback_output_format": model_cfg.fallback_output_format,
    }
    kwargs.update({k: v for k, v in optional.items() if v is not None})
    cfg_payload = kwargs.get("config") or {}
    cfg_payload.setdefault("original_model_id", model_cfg.id)
    cfg_payload.setdefault("alias", model_cfg.alias)
    if snapshot_paths and model_cfg.alias in snapshot_paths:
        cfg_payload.setdefault("snapshot_path", resolved_model_id)
    kwargs["config"] = cfg_payload
    retriever: BaseRetriever = cls(**kwargs)
    return retriever


def _sample_quick_subset(records: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    if max_items <= 0 or len(records) <= max_items:
        return records
    return random.sample(records, max_items)


def load_task_dataset(task: TaskConfig, data_cfg: DataConfig, quick_subset: Dict[str, Any]) -> List[Dict[str, Any]]:
    if task.mock_examples is not None:
        examples = task.mock_examples
    elif load_dataset is None:
        raise RuntimeError("datasets library not available and no mock examples provided")
    else:  # pragma: no cover - requires datasets
        ds = load_dataset(
            data_cfg.repo_id,
            task.subset,
            split=task.split,
            streaming=data_cfg.streaming,
        )
        if data_cfg.download and hasattr(ds, "download_and_prepare"):
            ds.download_and_prepare()  # type: ignore[attr-defined]
        examples = []
        for record in ds:
            examples.append(dict(record))  # type: ignore[arg-type]
    if quick_subset.get("enabled", False):
        max_queries = quick_subset.get("max_queries", data_cfg.quick_sample_per_task)
        examples = _sample_quick_subset(examples, max_queries)
    return examples


def _apply_quick_candidate_limit(examples: List[Dict[str, Any]], quick_subset: Dict[str, Any], max_candidates: int) -> None:
    limit = quick_subset.get("max_candidates")
    if not limit:
        limit = max_candidates
    for record in examples:
        candidates = record.get("candidates") or record.get("candidate_ids") or []
        record["candidates"] = list(candidates)[:limit]


def collect_candidates(task_examples: List[Dict[str, Any]], task: TaskConfig) -> List[Dict[str, Any]]:
    catalogue: Dict[str, Dict[str, Any]] = {}
    for record in task_examples:
        candidates = record.get("candidates") or record.get(task.candidate_column) or []
        candidate_texts = record.get("candidate_texts") or record.get("documents") or []
        candidate_images = record.get("candidate_images") or []
        for idx, candidate in enumerate(candidates):
            if isinstance(candidate, dict):
                cand_id = str(candidate.get("id") or candidate.get(task.id_column) or candidate.get("candidate_id"))
                if not cand_id:
                    continue
                if cand_id not in catalogue:
                    catalogue[cand_id] = {
                        "id": cand_id,
                        "text": candidate.get("text") or candidate.get("document") or candidate.get("caption"),
                        "image": candidate.get("image"),
                    }
            else:
                cand_id = str(candidate)
                text = candidate_texts[idx] if idx < len(candidate_texts) else None
                image = candidate_images[idx] if idx < len(candidate_images) else None
                if cand_id not in catalogue:
                    catalogue[cand_id] = {"id": cand_id, "text": text, "image": image}
    return list(catalogue.values())


def embed_candidates(
    retriever: BaseRetriever,
    cache: CacheManager,
    task_name: str,
    candidates: Sequence[Dict[str, Any]],
    *,
    shard_size: int,
    resume: ResumeManager,
) -> Dict[str, Dict[str, Any]]:
    cache_paths: Dict[str, Dict[str, Any]] = {}
    for shard_idx, shard in enumerate(chunk_iterable(list(candidates), shard_size)):
        shard_meta = {"task": task_name, "shard": shard_idx, "size": len(shard)}
        cached = cache.exists(shard_meta)
        if not cached:
            batch: Dict[str, Any] = {"texts": [c.get("text") for c in shard]}
            if any(c.get("image") is not None for c in shard):
                batch["images"] = [c.get("image") for c in shard]
            output = retriever.embed_document(batch)
            padded = scoring.pad_to_static(output.embeddings)
            cache.save(shard_meta, padded, extra={"ids": [c.get("id") for c in shard]})
            vram_gb = estimate_vram_usage([padded])
            LOGGER.info("Encoded shard %s size=%s approx_vram=%.3fGB", shard_idx, len(shard), vram_gb)
            resume.update(retriever.model_id, task_name, f"candidates_shard_{shard_idx}", {"status": "encoded"})
        cache_paths[str(shard_idx)] = {"meta": shard_meta, "path": cache.resolve(shard_meta)}
    return cache_paths


def score_queries(
    retriever: BaseRetriever,
    cache: CacheManager,
    cache_paths: Dict[str, Dict[str, Any]],
    task_examples: List[Dict[str, Any]],
    task: TaskConfig,
    scoring_cfg: ScoringConfig,
    monitoring_cfg: MonitoringConfig,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    correct = 0
    total = 0
    for idx, record in enumerate(task_examples):
        batch = {
            "texts": [record.get("query") or record.get(task.query_column)],
            "images": [record.get(task.image_column)] if task.image_column else None,
        }
        if batch["images"] is None:
            batch.pop("images")
        query_output = retriever.embed_query(batch)
        query_embeds = scoring.pad_to_static(query_output.embeddings)
        best_score = None
        best_candidate = None
        for shard_key, entry in cache_paths.items():
            path = entry["path"]
            payload = torch.load(path, map_location=query_embeds.device)
            doc_embeds = payload["embeddings"].to(device=query_embeds.device)
            shard_ids = payload.get("meta", {}).get("ids", [])  # type: ignore[index]
            if not shard_ids:
                shard_ids = [f"{shard_key}_{i}" for i in range(doc_embeds.size(0))]
            for doc_tensor, candidate_id in zip(doc_embeds, shard_ids):
                doc_tensor = doc_tensor.unsqueeze(0)
                value = scoring.hybrid_score(query_embeds, doc_tensor).item()
                if best_score is None or value > best_score:
                    best_score = value
                    best_candidate = candidate_id
        positives = record.get("positive_ids") or record.get(task.positive_column) or []
        if best_candidate in positives:
            correct += 1
        total += 1
        if monitoring_cfg.record_batch_stats and (idx + 1) % max(1, monitoring_cfg.log_interval) == 0:
            LOGGER.info("Processed %s/%s queries", idx + 1, len(task_examples))
    precision = correct / max(total, 1)
    metrics["precision@1"] = precision
    metrics["total_queries"] = total
    metrics["correct_top1"] = correct
    return metrics


def write_reports(artifact_cfg: ArtifactConfig, metrics: Dict[str, Any]) -> None:
    if not artifact_cfg.save_reports:
        return
    with open(artifact_cfg.json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(artifact_cfg.csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "task", "metric", "value"])
        for model_name, task_metrics in metrics.items():
            for task_name, values in task_metrics.items():
                if isinstance(values, dict):
                    for metric_name, metric_value in values.items():
                        writer.writerow([model_name, task_name, metric_name, metric_value])
                else:
                    writer.writerow([model_name, task_name, "value", values])


def run_mmeb_eval(config_path: str | Path, *, dry_run: bool = False) -> Dict[str, Any]:
    start_time = time.time()
    raw_config = load_config(Path(config_path))
    cfg = parse_config(raw_config)
    run_cfg: RunConfig = cfg["run"]
    data_cfg: DataConfig = cfg["data"]
    monitoring_cfg: MonitoringConfig = cfg["monitoring"]
    artifact_cfg: ArtifactConfig = cfg["artifacts"]
    scoring_cfg: ScoringConfig = cfg["scoring"]
    safety_cfg: SafetyConfig = cfg["safety"]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.info("Starting evaluation run %s", run_cfg.name)

    _validate_environment(run_cfg, strict=not dry_run)
    prepare_run_folders(run_cfg)
    resume = ResumeManager(Path(run_cfg.resume_state))
    snapshot_paths = _prefetch_model_assets(cfg["models"], run_cfg.model_snapshot_root)

    metrics: Dict[str, Any] = {}
    for model_cfg in cfg["models"]:
        retriever = instantiate_retriever(model_cfg, dry_run=dry_run, snapshot_paths=snapshot_paths)
        cache = CacheManager(Path(run_cfg.cache_root))
        model_metrics: Dict[str, Any] = {}
        for task in data_cfg.tasks:
            examples = load_task_dataset(task, data_cfg, run_cfg.quick_subset)
            _apply_quick_candidate_limit(examples, run_cfg.quick_subset, task.max_candidates)
            candidates = collect_candidates(examples, task)
            cache_paths = embed_candidates(
                retriever,
                cache,
                task.name,
                candidates,
                shard_size=scoring_cfg.shard_size,
                resume=resume,
            )
            task_metrics = score_queries(retriever, cache, cache_paths, examples, task, scoring_cfg, monitoring_cfg)
            model_metrics[task.name] = task_metrics
        metrics[model_cfg.alias] = model_metrics

    write_reports(artifact_cfg, metrics)
    LOGGER.info("Evaluation completed in %.2f seconds", time.time() - start_time)
    return metrics


def main() -> None:  # pragma: no cover - convenience entrypoint
    import argparse

    parser = argparse.ArgumentParser(description="Run MMEB multi-vector evaluation")
    parser.add_argument("--config", type=str, default="configs/mmeb_quick_smoke.yaml")
    parser.add_argument("--full", action="store_true", help="Run full evaluation (overrides quick subset)")
    args = parser.parse_args()

    metrics = run_mmeb_eval(args.config, dry_run=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI
    main()

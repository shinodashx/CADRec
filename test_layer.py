import argparse
import contextlib
import json
import multiprocessing as mp
import os
import re
import shutil
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

import recursive_bbox_infer as recursive_infer
from yaml_config import (
    find_latest_saved_config,
    get_nested,
    load_config_with_defaults,
    resolve_log_dir_from_checkpoint,
)

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_BBOX_LABEL_TOKEN_PATTERN = re.compile(r"<(SPLIT|STOP)>|\b(SPLIT|STOP)\b")
_BBOX_TOKEN_PATTERN = re.compile(r"<BOX>|\bBOX\b")
_BBOX_SPAN_PATTERN = re.compile(r"<Bs>\s*([^<]+?)\s*<Be>", re.IGNORECASE)
_BBOX_SOURCE_TEXT = "text"
_BBOX_SOURCE_HEAD = "bbox_head"
_BBOX_SOURCE_ORDER = (_BBOX_SOURCE_TEXT, _BBOX_SOURCE_HEAD)
_BBOX_SOURCE_ORDER_INDEX = {
    name: idx for idx, name in enumerate(_BBOX_SOURCE_ORDER)
}


def _normalize_bbox_source(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {"text", "text_bbox", "ce", "ce_text"}:
        return _BBOX_SOURCE_TEXT
    if text in {"bbox_head", "head", "grounding_head"}:
        return _BBOX_SOURCE_HEAD
    return _BBOX_SOURCE_TEXT


def _parse_bbox_sources_arg(value: str | None) -> list[str]:
    text = str(value or "").strip().lower()
    if not text:
        return [_BBOX_SOURCE_TEXT]
    if text in {"text", "text_only", "default"}:
        return [_BBOX_SOURCE_TEXT]
    if text in {"both", "all", "text+bbox_head", "text,bbox_head", "bbox_head,text"}:
        return list(_BBOX_SOURCE_ORDER)
    if text in {"bbox_head", "head", "grounding_head"}:
        return [_BBOX_SOURCE_HEAD]

    requested = []
    for item in text.split(","):
        normalized = _normalize_bbox_source(item)
        if normalized not in requested:
            requested.append(normalized)
    if not requested:
        return [_BBOX_SOURCE_TEXT]

    ordered = [
        bbox_source for bbox_source in _BBOX_SOURCE_ORDER
        if bbox_source in requested
    ]
    return ordered or [_BBOX_SOURCE_TEXT]


def _infer_bbox_source_from_output_dir(output_dir: str | os.PathLike | None) -> str:
    if output_dir is None:
        return _BBOX_SOURCE_TEXT
    try:
        name = Path(output_dir).name.lower()
    except Exception:
        name = str(output_dir).strip().lower()
    if name.endswith(f"_{_BBOX_SOURCE_HEAD}"):
        return _BBOX_SOURCE_HEAD
    return _BBOX_SOURCE_TEXT


def _sample_branch_key(sample_state: dict, bbox_source: str) -> str:
    return f"{sample_state['sample_key']}::{_normalize_bbox_source(bbox_source)}"


def _make_branch_state(sample_state: dict, bbox_source: str) -> dict:
    normalized_source = _normalize_bbox_source(bbox_source)
    return {
        "sample_state": sample_state,
        "sample_key": sample_state["sample_key"],
        "bbox_source": normalized_source,
        "branch_key": _sample_branch_key(sample_state, normalized_source),
    }


def _build_branch_states(sample_states: list[dict], bbox_sources: list[str]) -> list[dict]:
    branch_states = []
    for sample_state in sample_states:
        for bbox_source in bbox_sources:
            branch_states.append(_make_branch_state(sample_state, bbox_source))
    return branch_states


def str2bool(value):
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _log(*args):
    print(*args, flush=True)


def _load_train_config(config_path=None, checkpoint_path=None, save_config_name="run_config.yaml"):
    if config_path:
        return load_config_with_defaults(config_path), str(Path(config_path).resolve())
    if not checkpoint_path:
        raise ValueError("Please pass --checkpoint-path, or explicitly pass --config.")
    log_dir = resolve_log_dir_from_checkpoint(checkpoint_path)
    found_config = find_latest_saved_config(log_dir, save_config_name=save_config_name)
    if found_config is None:
        raise FileNotFoundError(
            f"Cannot find saved YAML config under log directory: {log_dir}. "
            f"Expected {save_config_name} or timestamped copies."
        )
    return load_config_with_defaults(str(found_config)), str(found_config.resolve())


def _collect_stl_files(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*.stl") if recursive else input_dir.glob("*.stl")
    files = sorted(path.resolve() for path in iterator if path.is_file())
    if not files:
        raise FileNotFoundError(f"No STL files found under: {input_dir}")
    return files


def _resolve_gt_stl_path(gt_dir: Path | None, input_dir: Path, input_stl_path: Path) -> Path:
    if gt_dir is None:
        return input_stl_path
    candidate = (gt_dir / input_stl_path.relative_to(input_dir)).resolve()
    fallback = (gt_dir / input_stl_path.name).resolve()
    return candidate if candidate.is_file() else fallback


def _coerce_loaded_mesh(mesh):
    if isinstance(mesh, trimesh.Scene):
        return trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _normalize_mesh(mesh):
    mesh = mesh.copy()
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-center)
    extent = float(np.max(mesh.extents))
    if extent > 1e-7:
        mesh.apply_scale(1.0 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh


def _compute_chamfer_distance(gt_mesh, pred_mesh, n_points: int) -> float:
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, int(n_points))
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, int(n_points))
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    return float(np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance)))


def _compute_iou(gt_mesh, pred_mesh) -> float | None:
    try:
        gt_components = list(gt_mesh.split())
        pred_components = list(pred_mesh.split())
        intersection_volume = 0.0
        for gt_mesh_i in gt_components:
            for pred_mesh_i in pred_components:
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                volume = intersection.volume if intersection is not None else 0.0
                intersection_volume += volume
        gt_volume = sum(component.volume for component in gt_components)
        pred_volume = sum(component.volume for component in pred_components)
        union_volume = gt_volume + pred_volume - intersection_volume
        if union_volume <= 0:
            return None
        return float(intersection_volume / union_volume)
    except Exception:
        return None


def _evaluate_prediction_metrics(gt_stl_path: str, pred_mesh_path: str | None, n_points: int) -> dict:
    metrics = {
        "gt_stl_path": str(gt_stl_path),
        "pred_mesh_path": pred_mesh_path,
        "cd": None,
        "cd_scaled_x1000": None,
        "iou": None,
    }
    if not pred_mesh_path:
        metrics["error"] = "Missing merged prediction mesh."
        return metrics
    pred_path = Path(pred_mesh_path)
    if not pred_path.is_file():
        metrics["error"] = f"Prediction mesh not found: {pred_path}"
        return metrics
    try:
        pred_mesh = _normalize_mesh(_coerce_loaded_mesh(trimesh.load_mesh(pred_path)))
        gt_mesh = _normalize_mesh(_coerce_loaded_mesh(trimesh.load_mesh(gt_stl_path)))
        metrics["cd"] = _compute_chamfer_distance(gt_mesh, pred_mesh, n_points=n_points)
        metrics["cd_scaled_x1000"] = float(metrics["cd"] * 1000.0)
        metrics["iou"] = _compute_iou(gt_mesh, pred_mesh)
    except Exception as exc:
        metrics["error"] = str(exc)
    return metrics


def _build_metrics_report(batch_results: list[dict], max_skip: int = 4) -> dict:
    metrics = defaultdict(dict)
    for item in batch_results:
        metrics[item["relative_path"]] = item.get("metrics") or {}

    cds = []
    ious = []
    ir_cd = 0
    ir_iou = 0
    failed_metric_files = []
    for relative_path, metric in metrics.items():
        cd_value = metric.get("cd")
        iou_value = metric.get("iou")
        if cd_value is None:
            ir_cd += 1
            failed_metric_files.append(relative_path)
        else:
            cds.append(float(cd_value))
        if iou_value is None:
            ir_iou += 1
        else:
            ious.append(float(iou_value))

    cd_scaled = sorted(float(value) * 1000.0 for value in cds)
    total = len(metrics)
    skip_stats = []
    for skip in range(max_skip + 1):
        kept = len(cd_scaled) - skip
        mean_cd = None if kept <= 0 else float(np.mean(cd_scaled[:kept]))
        skip_stats.append(
            {
                "skip": skip,
                "ir_percent": None if total == 0 else float((ir_cd + skip) / total * 100.0),
                "mean_cd": mean_cd,
            }
        )

    return {
        "sample_count": total,
        "valid_cd_count": len(cds),
        "valid_iou_count": len(ious),
        "ir_cd": ir_cd,
        "ir_iou": ir_iou,
        "mean_iou": None if not ious else float(np.mean(ious)),
        "median_cd": None if not cd_scaled else float(np.median(cd_scaled)),
        "skip_stats": skip_stats,
        "failed_metric_files": sorted(set(failed_metric_files)),
    }


def _print_metrics_report(report: dict) -> None:
    mean_iou = report.get("mean_iou")
    median_cd = report.get("median_cd")
    if mean_iou is not None or median_cd is not None:
        mean_iou_text = "nan" if mean_iou is None else f"{mean_iou:.3f}"
        median_cd_text = "nan" if median_cd is None else f"{median_cd:.3f}"
        print(f"mean iou: {mean_iou_text} median cd: {median_cd_text}")
    for item in report.get("skip_stats", []):
        mean_cd = item.get("mean_cd")
        mean_cd_text = "nan" if mean_cd is None else f"{mean_cd:.3f}"
        ir_text = "nan" if item.get("ir_percent") is None else f"{item['ir_percent']:.2f}"
        print(f"skip: {item['skip']} ir: {ir_text} mean cd: {mean_cd_text}")


def _best_attempt(attempt_results: list[dict], *, bbox_source: str | None = None) -> dict | None:
    best = None
    best_cd = None
    for item in attempt_results:
        if bbox_source is not None and _normalize_bbox_source(item.get("bbox_source")) != _normalize_bbox_source(bbox_source):
            continue
        cd_value = (item.get("metrics") or {}).get("cd")
        if cd_value is None:
            continue
        cd_value = float(cd_value)
        if best_cd is None or cd_value < best_cd:
            best_cd = cd_value
            best = item
    return best


def _sample_summary_path(sample_state: dict) -> Path:
    return sample_state["sample_output_dir"] / f"{sample_state['stl_path'].stem}_attempt_summary.json"


def _prune_attempt_result_fields(attempt_result: dict) -> dict:
    bbox_source = _normalize_bbox_source(attempt_result.get("bbox_source"))
    return {
        "status": attempt_result.get("status", "failed"),
        "attempt_index": int(attempt_result["attempt_index"]),
        "bbox_source": bbox_source,
        "branch_key": attempt_result.get("branch_key"),
        "relative_path": attempt_result["relative_path"],
        "input_path": str(attempt_result["input_path"]),
        "output_dir": str(attempt_result["output_dir"]),
        "result_path": str(attempt_result["result_path"]),
        "merged_mesh_path": attempt_result.get("merged_mesh_path"),
        "summary": attempt_result.get("summary") or {},
        "metrics": attempt_result.get("metrics") or {},
    }


def _write_sample_summary(sample_state: dict) -> dict:
    attempt_results = sorted(
        sample_state["attempt_results"],
        key=lambda item: (
            int(item["attempt_index"]),
            _BBOX_SOURCE_ORDER_INDEX.get(_normalize_bbox_source(item.get("bbox_source")), len(_BBOX_SOURCE_ORDER_INDEX)),
        ),
    )
    best_attempt = _best_attempt(attempt_results)
    best_metrics = {
        "bbox_source": None if best_attempt is None else _normalize_bbox_source(best_attempt.get("bbox_source")),
        "cd": None if best_attempt is None else (best_attempt.get("metrics") or {}).get("cd"),
        "cd_scaled_x1000": None if best_attempt is None else (best_attempt.get("metrics") or {}).get("cd_scaled_x1000"),
        "iou": None if best_attempt is None else (best_attempt.get("metrics") or {}).get("iou"),
        "best_attempt_index": None if best_attempt is None else int(best_attempt["attempt_index"]),
    }
    attempts_by_bbox_source = {}
    best_metrics_by_bbox_source = {}
    for bbox_source in _BBOX_SOURCE_ORDER:
        source_attempts = [
            item for item in attempt_results
            if _normalize_bbox_source(item.get("bbox_source")) == bbox_source
        ]
        attempts_by_bbox_source[bbox_source] = [
            _prune_attempt_result_fields(item) for item in source_attempts
        ]
        source_best_attempt = _best_attempt(source_attempts, bbox_source=bbox_source)
        best_metrics_by_bbox_source[bbox_source] = {
            "bbox_source": bbox_source,
            "cd": None if source_best_attempt is None else (source_best_attempt.get("metrics") or {}).get("cd"),
            "cd_scaled_x1000": None if source_best_attempt is None else (source_best_attempt.get("metrics") or {}).get("cd_scaled_x1000"),
            "iou": None if source_best_attempt is None else (source_best_attempt.get("metrics") or {}).get("iou"),
            "best_attempt_index": None if source_best_attempt is None else int(source_best_attempt["attempt_index"]),
        }
    payload = {
        "relative_path": sample_state["relative_path"],
        "input_path": str(sample_state["stl_path"]),
        "gt_stl_path": str(sample_state["gt_stl_path"]),
        "sample_output_dir": str(sample_state["sample_output_dir"]),
        "attempts": [_prune_attempt_result_fields(item) for item in attempt_results],
        "attempts_by_bbox_source": attempts_by_bbox_source,
        "best_metrics": best_metrics,
        "best_metrics_by_bbox_source": best_metrics_by_bbox_source,
    }
    path = _sample_summary_path(sample_state)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _prepare_sample_state(*, stl_path: Path, input_dir: Path, gt_dir: Path | None, output_dir: Path) -> dict:
    relative_path = stl_path.relative_to(input_dir)
    sample_output_dir = output_dir / relative_path.parent / relative_path.stem
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "sample_key": str(relative_path),
        "relative_path": str(relative_path),
        "stl_path": stl_path,
        "gt_stl_path": _resolve_gt_stl_path(gt_dir, input_dir, stl_path),
        "sample_output_dir": sample_output_dir,
        "root_dense": None,
        "attempt_results": [],
    }


def _point_count_from_chunk(dense_cloud) -> int:
    if dense_cloud is None:
        return 0
    points = getattr(dense_cloud, "points", None)
    if points is None:
        return 0
    points = np.asarray(points)
    if points.ndim != 2:
        return 0
    return int(points.shape[0])


def _validate_point_cloud_chunk(dense_cloud, *, min_points: int, min_span: float) -> str | None:
    if dense_cloud is None:
        return "missing point cloud"
    points = np.asarray(getattr(dense_cloud, "points", None), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return f"invalid point shape {points.shape}"
    points = points[:, :3]
    if points.shape[0] < int(min_points):
        return f"point count {points.shape[0]} < min_points {int(min_points)}"
    if not np.isfinite(points).all():
        return "point cloud contains NaN or Inf"
    span = float(np.max(points.max(axis=0) - points.min(axis=0)))
    if not np.isfinite(span):
        return "point cloud span is NaN or Inf"
    if span <= float(min_span):
        return f"point cloud span {span:.3e} <= min_span {float(min_span):.3e}"

    normals = getattr(dense_cloud, "normals", None)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.ndim != 2 or normals.shape[0] != points.shape[0] or normals.shape[1] < 3:
            return f"invalid normal shape {normals.shape}"
        if not np.isfinite(normals[:, :3]).all():
            return "point normals contain NaN or Inf"
    return None


def _load_root_dense(sample_state: dict, args) -> recursive_infer.PointCloudChunk:
    if sample_state["root_dense"] is None:
        sample_state["root_dense"] = recursive_infer._load_stl_point_cloud(
            str(sample_state["stl_path"]),
            dense_points=int(args.dense_points),
            use_point_normals=bool(args.use_point_normals),
        )
    return sample_state["root_dense"]


def _init_attempt_context(branch_state: dict, attempt_index: int) -> dict:
    sample_state = branch_state["sample_state"]
    bbox_source = _normalize_bbox_source(branch_state["bbox_source"])
    attempt_output_dir = sample_state["sample_output_dir"] / f"attempt_{attempt_index:02d}_{bbox_source}"
    if attempt_output_dir.exists():
        shutil.rmtree(attempt_output_dir, ignore_errors=True)
    attempt_output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "sample_key": sample_state["sample_key"],
        "branch_key": branch_state["branch_key"],
        "bbox_source": bbox_source,
        "relative_path": sample_state["relative_path"],
        "stl_path": sample_state["stl_path"],
        "gt_stl_path": sample_state["gt_stl_path"],
        "sample_state": sample_state,
        "attempt_index": int(attempt_index),
        "attempt_output_dir": attempt_output_dir,
        "tree": None,
        "frontier": [],
        "summary": {
            "total_nodes": 0,
            "split_nodes": 0,
            "stop_nodes": 0,
            "invalid_nodes": 0,
            "filtered_nodes": 0,
            "max_depth_stops": 0,
            "leaf_codes": 0,
            "leaf_meshes": 0,
        },
    }


def _make_node_result(node_job: dict, parsed_json, bbox_json_path: str) -> dict:
    dense_cloud = node_job["dense_cloud"]
    shift = dense_cloud.to_global_shift
    if shift is None:
        shift = np.zeros(3, dtype=np.float32)
    return {
        "node_id": node_job["node_id"],
        "depth": int(node_job["depth"]),
        "status": "pending",
        "label": None,
        "dense_point_count": _point_count_from_chunk(dense_cloud),
        "to_global_scale": float(dense_cloud.to_global_scale),
        "to_global_shift": np.asarray(shift, dtype=np.float32).tolist(),
        "bbox_json_path": bbox_json_path,
        "prediction": parsed_json,
    }


def _skip_sample(meta: dict, reason: str) -> dict:
    meta["skip_reason"] = str(reason)
    return {"batch_input": None, "meta": meta}


class _BBoxNodeDataset(Dataset):
    def __init__(self, args, node_jobs: list[dict]):
        self.args = args
        self.node_jobs = node_jobs

    def __len__(self):
        return len(self.node_jobs)

    def __getitem__(self, index):
        node_job = self.node_jobs[index]
        dense_cloud = node_job["dense_cloud"]
        meta = {"job_index": int(index), "dense_cloud": dense_cloud, "sampled": None}
        try:
            if dense_cloud is None:
                dense_cloud = _load_root_dense(
                    node_job["attempt_ctx"]["sample_state"],
                    self.args,
                )
                meta["dense_cloud"] = dense_cloud
            invalid_reason = _validate_point_cloud_chunk(
                dense_cloud,
                min_points=int(getattr(self.args, "min_valid_points", 2)),
                min_span=float(getattr(self.args, "min_valid_span", 1e-6)),
            )
            if invalid_reason is not None:
                return _skip_sample(meta, invalid_reason)
            sampled = recursive_infer._fps_sample(
                dense_cloud.points,
                target_count=int(self.args.n_points),
                normals=dense_cloud.normals if bool(self.args.use_point_normals) else None,
            )
            invalid_reason = _validate_point_cloud_chunk(
                sampled,
                min_points=int(getattr(self.args, "min_valid_points", 2)),
                min_span=float(getattr(self.args, "min_valid_span", 1e-6)),
            )
            if invalid_reason is not None:
                return _skip_sample(meta, f"sampled point cloud invalid: {invalid_reason}")
        except Exception as exc:
            return _skip_sample(meta, str(exc))
        meta["sampled"] = sampled
        batch_input = recursive_infer._prepare_bbox_batch(
            sampled=sampled,
            node_id=node_job["node_id"],
            normalize_std=float(self.args.normalize_std),
        )[0]
        return {"batch_input": batch_input, "meta": meta}


class _CodeLeafDataset(Dataset):
    def __init__(self, args, leaf_jobs: list[dict]):
        self.args = args
        self.leaf_jobs = leaf_jobs

    def __len__(self):
        return len(self.leaf_jobs)

    def __getitem__(self, index):
        leaf_job = self.leaf_jobs[index]
        meta = {"leaf_job_index": int(index)}
        try:
            invalid_reason = _validate_point_cloud_chunk(
                leaf_job["sampled"],
                min_points=int(getattr(self.args, "min_valid_points", 2)),
                min_span=float(getattr(self.args, "min_valid_span", 1e-6)),
            )
            if invalid_reason is not None:
                return _skip_sample(meta, invalid_reason)
            batch_input = recursive_infer._prepare_code_batch(
                sampled=leaf_job["sampled"],
                node_id=leaf_job["node_id"],
                normalize_std=float(self.args.normalize_std),
            )[0]
        except Exception as exc:
            return _skip_sample(meta, str(exc))
        return {"batch_input": batch_input, "meta": meta}


def _task_loader_collate(samples, *, processor, runtime: dict, n_points: int):
    from cadrec import collate

    valid_samples = [sample for sample in samples if sample.get("batch_input") is not None]
    batch_inputs = [sample["batch_input"] for sample in valid_samples]
    metas = [sample["meta"] for sample in valid_samples]
    skipped_metas = [sample["meta"] for sample in samples if sample.get("batch_input") is None]
    if not batch_inputs:
        return {
            "model_inputs": None,
            "metas": [],
            "skipped_metas": skipped_metas,
        }
    model_inputs = collate(
        batch_inputs,
        processor=processor,
        n_points=int(n_points),
        n_point_tokens=runtime["n_point_tokens"],
        n_part_point_tokens=runtime["part_points_per_bbox"],
        eval=True,
        pc_encoder_type=runtime["pc_encoder_type"],
        utonia_scale=runtime["utonia_scale"],
        utonia_normalize_coord=runtime["utonia_normalize_coord"],
        utonia_use_normal=runtime["utonia_use_normal"],
    )
    return {
        "model_inputs": model_inputs,
        "metas": metas,
        "skipped_metas": skipped_metas,
    }


def _move_model_inputs_to_device(model_inputs: dict, device, non_blocking: bool) -> dict:
    moved_inputs = {}
    for key, value in model_inputs.items():
        if value is None or not torch.is_tensor(value):
            moved_inputs[key] = value
            continue
        moved_inputs[key] = value.to(device, non_blocking=non_blocking)
    return moved_inputs


def _iter_prefetched_loader_batches(loader, device, *, iterator=None):
    if iterator is None:
        iterator = iter(loader)

    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        yield from iterator
        return

    stream = torch.cuda.Stream(device=device)
    next_batch = None

    def _preload():
        nonlocal next_batch
        try:
            batch = next(iterator)
        except StopIteration:
            next_batch = None
            return
        with torch.cuda.stream(stream):
            next_batch = {
                "model_inputs": _move_model_inputs_to_device(
                    batch["model_inputs"],
                    device=device,
                    non_blocking=True,
                ) if batch["model_inputs"] is not None else None,
                "metas": batch["metas"],
                "skipped_metas": batch.get("skipped_metas", []),
            }

    _preload()
    while next_batch is not None:
        torch.cuda.current_stream(device).wait_stream(stream)
        current_batch = next_batch
        _preload()
        yield current_batch


def _iter_recursive_batches(*, dataset, batch_size: int, loader_workers: int, collate_fn, device):
    batch_size = max(int(batch_size), 1)
    loader_workers = max(int(loader_workers), 0)
    dataset_size = len(dataset)
    use_sync = loader_workers == 0 or dataset_size <= batch_size * 2
    if use_sync:
        for start in range(0, dataset_size, batch_size):
            samples = [dataset[index] for index in range(start, min(start + batch_size, dataset_size))]
            yield collate_fn(samples)
        return

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=loader_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=2 if loader_workers > 0 else None,
        collate_fn=collate_fn,
    )
    loader_iterator = iter(loader)
    try:
        yield from _iter_prefetched_loader_batches(loader, device, iterator=loader_iterator)
    finally:
        shutdown_workers = getattr(loader_iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
        del loader_iterator
        del loader


def _resolve_execution_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped in hf_device_map.values():
            if isinstance(mapped, str):
                if mapped.startswith("cuda"):
                    return torch.device(mapped)
            elif isinstance(mapped, int):
                return torch.device(f"cuda:{mapped}")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _build_generate_kwargs(model_inputs: dict, max_new_tokens: int, *, bbox_grounding_infer: bool = False) -> dict:
    generate_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "point_clouds": model_inputs["point_clouds"],
        "is_pc": model_inputs["is_pc"],
        "is_img": model_inputs["is_img"],
        "max_new_tokens": int(max_new_tokens),
    }
    if model_inputs.get("pc_type_ids") is not None:
        generate_kwargs["pc_type_ids"] = model_inputs["pc_type_ids"]
    if model_inputs.get("part_point_indices") is not None:
        generate_kwargs["part_point_indices"] = model_inputs["part_point_indices"]
    if model_inputs.get("n_parts") is not None:
        generate_kwargs["n_parts"] = model_inputs["n_parts"]
    if model_inputs.get("is_step_pc") is not None:
        generate_kwargs["is_step_pc"] = model_inputs["is_step_pc"]
    if model_inputs.get("pixel_values_videos") is not None:
        generate_kwargs["pixel_values_videos"] = model_inputs["pixel_values_videos"]
    if model_inputs.get("video_grid_thw") is not None:
        generate_kwargs["video_grid_thw"] = model_inputs["video_grid_thw"]
    if bbox_grounding_infer:
        generate_kwargs["bbox_grounding_infer"] = True
    return generate_kwargs


def _decode_generated_ids(processor, input_ids: torch.Tensor, generated_ids: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )


def _normalize_bbox_prediction_dict(parsed_json):
    if not isinstance(parsed_json, dict):
        return parsed_json
    normalized = dict(parsed_json)
    model_box = normalized.get("model_box")
    if "parts" not in normalized and isinstance(model_box, dict) and isinstance(model_box.get("parts"), dict):
        normalized["parts"] = model_box.get("parts")
    if "label" not in normalized or not str(normalized.get("label", "")).strip():
        if isinstance(normalized.get("parts"), dict):
            normalized["label"] = "SPLIT"
        elif isinstance(normalized.get("steps"), dict):
            normalized["label"] = "STOP"
    return _convert_to_legacy_bbox_json(normalized)


def _infer_bbox_label_from_text(output_text: str) -> str:
    if not isinstance(output_text, str):
        return ""
    match = _BBOX_LABEL_TOKEN_PATTERN.search(output_text)
    if match is None:
        return ""
    return str(match.group(1) or match.group(2) or "").strip().upper()


def _count_box_tokens(output_text: str) -> int:
    if not isinstance(output_text, str):
        return 0
    return int(len(_BBOX_TOKEN_PATTERN.findall(output_text)))


def _count_bbox_spans(output_text: str) -> int:
    if not isinstance(output_text, str):
        return 0
    return int(len(_BBOX_SPAN_PATTERN.findall(output_text)))


def _denormalize_predicted_boxes(predicted_boxes, normalize_std: float) -> list[list[float]]:
    if predicted_boxes is None:
        return []
    if torch.is_tensor(predicted_boxes):
        predicted_boxes = predicted_boxes.detach().float().cpu().numpy()
    boxes = np.asarray(predicted_boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    out = []
    for box in boxes:
        box = np.asarray(box, dtype=np.float32).reshape(-1)
        if box.shape[0] < 6 or np.any(~np.isfinite(box[:6])):
            continue
        lower = np.minimum(box[:3], box[3:6]) * float(normalize_std)
        upper = np.maximum(box[:3], box[3:6]) * float(normalize_std)
        out.append(np.concatenate([lower, upper], axis=0).astype(np.float32).tolist())
    return out


def _compute_model_bbox_from_boxes(box_values: list[list[float]]) -> list[float]:
    if not box_values:
        return [0.0] * 6
    boxes = np.asarray(box_values, dtype=np.float32).reshape(-1, 6)
    lower = np.minimum(boxes[:, :3], boxes[:, 3:6])
    upper = np.maximum(boxes[:, :3], boxes[:, 3:6])
    model_lower = np.min(lower, axis=0)
    model_upper = np.max(upper, axis=0)
    return np.concatenate([model_lower, model_upper], axis=0).astype(np.float32).tolist()


def _parse_bbox_spans_from_text(output_text: str) -> list[list[float]]:
    if not isinstance(output_text, str):
        return []
    box_values = []
    for match in _BBOX_SPAN_PATTERN.finditer(output_text):
        raw_text = str(match.group(1) or "").strip()
        if not raw_text:
            continue
        pieces = [piece.strip() for piece in raw_text.split(",")]
        if len(pieces) < 6:
            continue
        try:
            box_value = [float(piece) for piece in pieces[:6]]
        except ValueError:
            continue
        if not np.isfinite(np.asarray(box_value, dtype=np.float32)).all():
            continue
        box_values.append(box_value)
    return box_values


def _build_legacy_bbox_json_from_text(output_text: str):
    label = _infer_bbox_label_from_text(output_text)
    if label not in {"SPLIT", "STOP"}:
        return None

    box_values = _parse_bbox_spans_from_text(output_text)
    normalized = {
        "label": label,
        "model_bbox": _compute_model_bbox_from_boxes(box_values),
    }
    container_key = "parts" if label == "SPLIT" else "steps"
    item_prefix = "part" if label == "SPLIT" else "step"
    normalized[container_key] = {
        f"{item_prefix}_{idx}": box_value
        for idx, box_value in enumerate(box_values, start=1)
    }
    return _convert_to_legacy_bbox_json(normalized)


def _convert_to_legacy_bbox_json(parsed_json):
    if not isinstance(parsed_json, dict):
        return parsed_json

    normalized = dict(parsed_json)
    label = str(normalized.get("label", "")).strip().upper()
    if label not in {"SPLIT", "STOP"}:
        return normalized

    container_key = "parts" if label == "SPLIT" else "steps"
    legacy_container = normalized.get(container_key)
    if not isinstance(legacy_container, dict):
        legacy_container = {}

    if not legacy_container and label == "SPLIT" and isinstance(normalized.get("model_box"), dict):
        model_box = normalized.get("model_box") or {}
        if isinstance(model_box.get("parts"), dict):
            legacy_container = dict(model_box.get("parts") or {})

    cleaned_container = {}
    for name, value in legacy_container.items():
        try:
            bbox_min, bbox_max = recursive_infer._box_array_to_min_max(value)
        except Exception:
            continue
        cleaned_container[str(name)] = np.concatenate([bbox_min, bbox_max], axis=0).astype(np.float32).tolist()

    model_bbox_value = normalized.get("model_bbox")
    if model_bbox_value is not None:
        try:
            bbox_min, bbox_max = recursive_infer._box_array_to_min_max(model_bbox_value)
            model_bbox = np.concatenate([bbox_min, bbox_max], axis=0).astype(np.float32).tolist()
        except Exception:
            model_bbox = _compute_model_bbox_from_boxes(list(cleaned_container.values()))
    else:
        model_bbox = _compute_model_bbox_from_boxes(list(cleaned_container.values()))

    return {
        "label": label,
        "model_bbox": model_bbox,
        container_key: cleaned_container,
    }


def _extract_text_bbox_prediction(output_text: str):
    parsed_json = _normalize_bbox_prediction_dict(recursive_infer._extract_json_block(output_text))
    parsed_label = recursive_infer._infer_label(parsed_json) if isinstance(parsed_json, dict) else ""
    if parsed_label in {"SPLIT", "STOP"}:
        return parsed_json
    text_bbox = _build_legacy_bbox_json_from_text(output_text)
    if text_bbox is not None:
        return text_bbox
    return parsed_json


def _rebuild_bbox_prediction(output_text: str, head_prediction, normalize_std: float, *, prefer_head: bool = False):
    parsed_json = _extract_text_bbox_prediction(output_text)
    parsed_label = recursive_infer._infer_label(parsed_json) if isinstance(parsed_json, dict) else ""
    if parsed_label in {"SPLIT", "STOP"} and not prefer_head:
        return parsed_json

    label = parsed_label if parsed_label in {"SPLIT", "STOP"} else _infer_bbox_label_from_text(output_text)
    if label not in {"SPLIT", "STOP"}:
        return None if prefer_head else parsed_json

    box_values = _denormalize_predicted_boxes(
        None if not isinstance(head_prediction, dict) else head_prediction.get("boxes"),
        normalize_std=normalize_std,
    )
    n_box_items = _count_bbox_spans(output_text)
    if n_box_items <= 0:
        n_box_items = _count_box_tokens(output_text)
    if n_box_items > 0:
        box_values = box_values[:n_box_items]

    if label == "SPLIT" and not box_values:
        return None if prefer_head else parsed_json

    normalized = {
        "label": label,
        "model_bbox": _compute_model_bbox_from_boxes(box_values),
    }
    if box_values:
        container_key = "parts" if label == "SPLIT" else "steps"
        item_prefix = "part" if label == "SPLIT" else "step"
        normalized[container_key] = {
            f"{item_prefix}_{idx}": box_value
            for idx, box_value in enumerate(box_values, start=1)
        }
    else:
        normalized["steps" if label == "STOP" else "parts"] = {}
    return _convert_to_legacy_bbox_json(normalized)


def _ensure_bbox_grounding_box_token_id(model, processor) -> bool:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return False
    vocab = tokenizer.get_vocab()
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    resolved_any = False

    def resolve_token_id(token: str):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            return None
        if unk_token_id is not None and int(token_id) == int(unk_token_id) and token not in vocab:
            return None
        return int(token_id)

    token_mappings = {
        "bbox_grounding_box_token_id": "<BOX>",
        "bbox_grounding_span_start_token_id": "<Bs>",
        "bbox_grounding_span_end_token_id": "<Be>",
        "bbox_grounding_split_token_id": "<SPLIT>",
        "bbox_grounding_stop_token_id": "<STOP>",
    }
    for attr_name, token in token_mappings.items():
        token_id = resolve_token_id(token)
        setattr(model, attr_name, -1 if token_id is None else int(token_id))
        if token_id is not None and attr_name in {
            "bbox_grounding_box_token_id",
            "bbox_grounding_span_start_token_id",
            "bbox_grounding_span_end_token_id",
        }:
            resolved_any = True
    return resolved_any


def _generate_from_model_inputs(model, processor, model_inputs: dict, max_new_tokens: int) -> list[str]:
    exec_device = _resolve_execution_device(model)
    non_blocking = exec_device.type == "cuda"
    model_inputs = _move_model_inputs_to_device(
        model_inputs,
        device=exec_device,
        non_blocking=non_blocking,
    )
    generate_kwargs = _build_generate_kwargs(model_inputs, max_new_tokens=max_new_tokens)

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    return _decode_generated_ids(
        processor,
        model_inputs["input_ids"],
        generated_ids,
        skip_special_tokens=True,
    )


def _generate_bbox_outputs_from_model_inputs(
    model,
    processor,
    model_inputs: dict,
    max_new_tokens: int,
    normalize_std: float,
    bbox_prediction_mode: str = _BBOX_SOURCE_TEXT,
) -> list[tuple[str, object]]:
    exec_device = _resolve_execution_device(model)
    non_blocking = exec_device.type == "cuda"
    model_inputs = _move_model_inputs_to_device(
        model_inputs,
        device=exec_device,
        non_blocking=non_blocking,
    )

    bbox_grounding_ready = bool(getattr(model, "bbox_grounding_enabled", False)) and _ensure_bbox_grounding_box_token_id(model, processor)
    generate_kwargs = _build_generate_kwargs(
        model_inputs,
        max_new_tokens=max_new_tokens,
        bbox_grounding_infer=bbox_grounding_ready,
    )

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    output_texts = _decode_generated_ids(
        processor,
        model_inputs["input_ids"],
        generated_ids,
        skip_special_tokens=not bbox_grounding_ready,
    )

    bbox_prediction_mode = _normalize_bbox_source(bbox_prediction_mode)
    if not bbox_grounding_ready:
        return [
            (output_text, _extract_text_bbox_prediction(output_text))
            for output_text in output_texts
        ]

    if bbox_prediction_mode == _BBOX_SOURCE_TEXT:
        return [
            (output_text, _extract_text_bbox_prediction(output_text))
            for output_text in output_texts
        ]

    append_len = int(generated_ids.shape[1] - model_inputs["input_ids"].shape[1])
    if append_len > 0:
        generated_mask = torch.ones(
            model_inputs["attention_mask"].shape[0],
            append_len,
            dtype=model_inputs["attention_mask"].dtype,
            device=model_inputs["attention_mask"].device,
        )
        full_attention_mask = torch.cat([model_inputs["attention_mask"], generated_mask], dim=1)
    else:
        full_attention_mask = model_inputs["attention_mask"]

    forward_kwargs = {
        "input_ids": generated_ids,
        "attention_mask": full_attention_mask,
        "point_clouds": model_inputs["point_clouds"],
        "is_pc": model_inputs["is_pc"],
        "is_img": model_inputs["is_img"],
        "bbox_grounding_infer": True,
    }
    for key in ("pc_type_ids", "part_point_indices", "n_parts", "is_step_pc", "pixel_values_videos", "video_grid_thw"):
        if model_inputs.get(key) is not None:
            forward_kwargs[key] = model_inputs[key]

    with torch.no_grad():
        model(**forward_kwargs)

    head_predictions = getattr(model, "latest_bbox_grounding_predictions", None)
    if not isinstance(head_predictions, list):
        head_predictions = [None] * len(output_texts)

    return [
        (
            output_text,
            _rebuild_bbox_prediction(
                output_text,
                head_prediction=head_prediction,
                normalize_std=normalize_std,
                prefer_head=(bbox_prediction_mode == _BBOX_SOURCE_HEAD),
            ),
        )
        for output_text, head_prediction in zip(output_texts, head_predictions)
    ]


def _run_bbox_layer(
    model,
    processor,
    runtime: dict,
    args,
    current_jobs: list[dict],
    *,
    attempt_index: int,
    bbox_source: str,
    ) -> tuple[list[dict], list[dict]]:
    next_frontier = []
    code_jobs = []
    depth = int(current_jobs[0]["depth"]) if current_jobs else -1
    dataset = _BBoxNodeDataset(args, current_jobs)
    collate_fn = partial(
        _task_loader_collate,
        processor=processor,
        runtime=runtime,
        n_points=int(args.n_points),
    )
    loader_iter = _iter_recursive_batches(
        dataset=dataset,
        batch_size=int(args.recursive_bbox_batch_size),
        loader_workers=int(getattr(args, "recursive_loader_workers", 8)),
        collate_fn=collate_fn,
        device=_resolve_execution_device(model),
    )
    total_batches = 0 if len(dataset) <= 0 else (len(dataset) + int(args.recursive_bbox_batch_size) - 1) // int(args.recursive_bbox_batch_size)
    for loader_batch in tqdm(
        loader_iter,
        total=total_batches,
        desc=f"Attempt {attempt_index} bbox[{bbox_source}] depth={depth}",
        leave=False,
        dynamic_ncols=True,
    ):
        for skipped_meta in loader_batch.get("skipped_metas", []):
            node_job = current_jobs[int(skipped_meta["job_index"])]
            attempt_ctx = node_job["attempt_ctx"]
            result = {
                "node_id": node_job["node_id"],
                "depth": int(node_job["depth"]),
                "status": "invalid_input",
                "label": None,
                "dense_point_count": _point_count_from_chunk(skipped_meta.get("dense_cloud")),
                "bbox_json_path": "",
                "prediction": None,
                "error": skipped_meta.get("skip_reason"),
            }
            if node_job["parent_children"] is None:
                attempt_ctx["tree"] = result
            else:
                node_job["parent_children"][node_job["child_name"]] = result
            attempt_ctx["summary"]["total_nodes"] += 1
            attempt_ctx["summary"]["invalid_nodes"] += 1
            attempt_ctx["summary"]["filtered_nodes"] += 1

        if loader_batch.get("model_inputs") is None or not loader_batch.get("metas"):
            continue

        bbox_outputs = _generate_bbox_outputs_from_model_inputs(
            model=model,
            processor=processor,
            model_inputs=loader_batch["model_inputs"],
            max_new_tokens=int(args.max_new_tokens_bbox),
            normalize_std=float(args.normalize_std),
            bbox_prediction_mode=bbox_source,
        )

        for meta, (output_text, parsed_json) in zip(loader_batch["metas"], bbox_outputs):
            node_job = current_jobs[int(meta["job_index"])]
            sampled = meta["sampled"]
            node_job["dense_cloud"] = meta["dense_cloud"]
            attempt_ctx = node_job["attempt_ctx"]
            _, bbox_json_path = recursive_infer._save_node_outputs(
                attempt_ctx["attempt_output_dir"],
                node_job["node_id"],
                output_text,
                parsed_json,
            )
            result = _make_node_result(node_job, parsed_json, bbox_json_path)
            if node_job["parent_children"] is None:
                attempt_ctx["tree"] = result
            else:
                node_job["parent_children"][node_job["child_name"]] = result
            attempt_ctx["summary"]["total_nodes"] += 1

            if parsed_json is None:
                attempt_ctx["summary"]["invalid_nodes"] += 1
                result["status"] = "invalid_json"
                continue

            label = recursive_infer._infer_label(parsed_json)
            result["label"] = label

            if label == "STOP":
                attempt_ctx["summary"]["stop_nodes"] += 1
                result["status"] = "leaf"
                code_jobs.append(
                    {
                        "attempt_ctx": attempt_ctx,
                        "node_id": node_job["node_id"],
                        "sampled": sampled,
                        "result": result,
                    }
                )
                continue

            if label != "SPLIT":
                attempt_ctx["summary"]["invalid_nodes"] += 1
                result["status"] = "unknown_label"
                continue

            if int(node_job["depth"]) >= int(args.max_depth):
                attempt_ctx["summary"]["max_depth_stops"] += 1
                result["status"] = "leaf"
                result["label"] = "STOP"
                code_jobs.append(
                    {
                        "attempt_ctx": attempt_ctx,
                        "node_id": node_job["node_id"],
                        "sampled": sampled,
                        "result": result,
                    }
                )
                continue

            parts = parsed_json.get("parts")
            if not isinstance(parts, dict) or not parts:
                attempt_ctx["summary"]["invalid_nodes"] += 1
                result["status"] = "split_without_parts"
                continue

            attempt_ctx["summary"]["split_nodes"] += 1
            result["status"] = "split"
            result["children"] = {}
            for child_index, (child_name, child_bbox) in enumerate(recursive_infer._sort_named_boxes(parts), start=1):
                child_id = recursive_infer._make_child_node_id(node_job["node_id"], child_name, child_index)
                try:
                    child_dense = recursive_infer._crop_and_renormalize(
                        node_dense=node_job["dense_cloud"],
                        bbox_value=child_bbox,
                        crop_eps=float(args.crop_eps),
                    )
                except Exception as exc:
                    attempt_ctx["summary"]["invalid_nodes"] += 1
                    result["children"][child_name] = {
                        "node_id": child_id,
                        "depth": int(node_job["depth"] + 1),
                        "status": "child_error",
                        "error": str(exc),
                        "source_bbox": child_bbox,
                    }
                    continue
                invalid_reason = _validate_point_cloud_chunk(
                    child_dense,
                    min_points=int(getattr(args, "min_valid_points", 2)),
                    min_span=float(getattr(args, "min_valid_span", 1e-6)),
                )
                if invalid_reason is not None:
                    attempt_ctx["summary"]["invalid_nodes"] += 1
                    attempt_ctx["summary"]["filtered_nodes"] += 1
                    result["children"][child_name] = {
                        "node_id": child_id,
                        "depth": int(node_job["depth"] + 1),
                        "status": "filtered_input",
                        "error": invalid_reason,
                        "dense_point_count": _point_count_from_chunk(child_dense),
                        "source_bbox": child_bbox,
                    }
                    continue
                next_frontier.append(
                    {
                        "attempt_ctx": attempt_ctx,
                        "node_id": child_id,
                        "depth": int(node_job["depth"] + 1),
                        "dense_cloud": child_dense,
                        "parent_children": result["children"],
                        "child_name": child_name,
                    }
                )
    return next_frontier, code_jobs


def _run_code_layer(
    model,
    processor,
    runtime: dict,
    args,
    code_jobs: list[dict],
    *,
    attempt_index: int,
    bbox_source: str,
) -> None:
    dataset = _CodeLeafDataset(args, code_jobs)
    collate_fn = partial(
        _task_loader_collate,
        processor=processor,
        runtime=runtime,
        n_points=int(args.n_points),
    )
    loader_iter = _iter_recursive_batches(
        dataset=dataset,
        batch_size=int(args.recursive_code_batch_size),
        loader_workers=int(getattr(args, "recursive_loader_workers", 8)),
        collate_fn=collate_fn,
        device=_resolve_execution_device(model),
    )
    total_batches = 0 if len(dataset) <= 0 else (len(dataset) + int(args.recursive_code_batch_size) - 1) // int(args.recursive_code_batch_size)
    for loader_batch in tqdm(
        loader_iter,
        total=total_batches,
        desc=f"Attempt {attempt_index} code[{bbox_source}] depth={batch_jobs_depth(code_jobs)}",
        leave=False,
        dynamic_ncols=True,
    ):
        for skipped_meta in loader_batch.get("skipped_metas", []):
            leaf_job = code_jobs[int(skipped_meta["leaf_job_index"])]
            leaf_job["result"]["status"] = "invalid_input"
            leaf_job["result"]["error"] = skipped_meta.get("skip_reason")

        if loader_batch.get("model_inputs") is None or not loader_batch.get("metas"):
            continue

        output_texts = _generate_from_model_inputs(
            model=model,
            processor=processor,
            model_inputs=loader_batch["model_inputs"],
            max_new_tokens=int(args.max_new_tokens_code),
        )
        for meta, code_text in zip(loader_batch["metas"], output_texts):
            leaf_job = code_jobs[int(meta["leaf_job_index"])]
            code_path = recursive_infer._save_leaf_code(
                leaf_job["attempt_ctx"]["attempt_output_dir"],
                leaf_job["node_id"],
                code_text,
            )
            leaf_job["attempt_ctx"]["summary"]["leaf_codes"] += 1
            leaf_job["result"]["leaf_code_path"] = code_path


def batch_jobs_depth(code_jobs: list[dict]) -> int:
    if not code_jobs:
        return -1
    return int((code_jobs[0].get("result") or {}).get("depth", -1))


def _leaf_mesh_worker(code_path: str, scale: float, shift: list[float], mesh_path: str, result_conn):
    error = None
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                mesh = recursive_infer._load_code_mesh(code_path)
                mesh = recursive_infer._transform_mesh_to_global(
                    mesh,
                    scale=scale,
                    shift=np.asarray(shift, dtype=np.float32),
                )
                mesh.export(mesh_path)
    except BaseException:
        error = traceback.format_exc()
    finally:
        result_conn.send(error)
        result_conn.close()


def _terminate_process(proc: mp.Process, *, join_timeout: float = 2.0) -> None:
    if proc.is_alive():
        proc.terminate()
        proc.join(float(join_timeout))
    if proc.is_alive():
        proc.kill()
        proc.join(float(join_timeout))


def _export_leaf_mesh_with_timeout(code_path: str, scale: float, shift: list[float], mesh_path: str, timeout_sec: float) -> tuple[str | None, str | None]:
    mesh_path_obj = Path(mesh_path)
    mesh_path_obj.unlink(missing_ok=True)
    parent_conn, child_conn = mp.Pipe(duplex=False)
    proc = mp.Process(
        target=_leaf_mesh_worker,
        args=(code_path, float(scale), list(shift), str(mesh_path_obj), child_conn),
    )
    started = False
    try:
        proc.start()
        started = True
        child_conn.close()
        child_conn = None
        proc.join(float(timeout_sec))
        if proc.is_alive():
            _terminate_process(proc)
            return None, f"mesh export timeout after {float(timeout_sec):.1f}s"
        worker_error = parent_conn.recv() if parent_conn.poll() else None
        if worker_error:
            return None, str(worker_error)
        if proc.exitcode not in (0, None):
            return None, f"mesh export failed with exit code {proc.exitcode}"
        if not mesh_path_obj.is_file():
            return None, "mesh export failed without output mesh"
        return str(mesh_path_obj), None
    finally:
        if started and proc.is_alive():
            _terminate_process(proc)
        parent_conn.close()
        if child_conn is not None:
            child_conn.close()
        if started:
            proc.close()


def _finalize_attempt(sample_key: str, attempt_ctx: dict, args) -> dict:
    attempt_output_dir = attempt_ctx["attempt_output_dir"]
    result_path = attempt_output_dir / f"{attempt_ctx['stl_path'].stem}_recursive_tree.json"
    try:
        tree = attempt_ctx["tree"]
        merge_errors = []
        leaf_mesh_paths = []
        leaf_meshes = []
        for leaf in recursive_infer._iter_leaf_nodes(tree):
            code_path = leaf.get("leaf_code_path")
            if not code_path:
                continue
            mesh_path = attempt_output_dir / "nodes" / f"{recursive_infer._sanitize_node_id(leaf['node_id'])}_mesh.stl"
            saved_mesh_path, error = _export_leaf_mesh_with_timeout(
                code_path=code_path,
                scale=float(leaf.get("to_global_scale", 1.0)),
                shift=leaf.get("to_global_shift", [0.0, 0.0, 0.0]),
                mesh_path=str(mesh_path),
                timeout_sec=float(args.mesh_export_timeout_sec),
            )
            if error:
                leaf["leaf_mesh_error"] = error
                merge_errors.append({"node_id": leaf.get("node_id", ""), "error": error})
                continue
            leaf["leaf_mesh_path"] = saved_mesh_path
            leaf_mesh_paths.append(saved_mesh_path)
            try:
                leaf_meshes.append(_coerce_loaded_mesh(trimesh.load_mesh(saved_mesh_path)))
            except Exception as exc:
                leaf["leaf_mesh_error"] = str(exc)
                merge_errors.append({"node_id": leaf.get("node_id", ""), "error": str(exc)})

        merged_mesh_path = None
        if leaf_meshes:
            merged_mesh = trimesh.util.concatenate(leaf_meshes)
            merged_mesh_path = str(attempt_output_dir / "merged_result.stl")
            merged_mesh.export(merged_mesh_path)

        metrics = _evaluate_prediction_metrics(
            gt_stl_path=str(attempt_ctx["gt_stl_path"]),
            pred_mesh_path=merged_mesh_path,
            n_points=int(args.eval_n_points),
        )
        attempt_ctx["summary"]["leaf_meshes"] = len(leaf_mesh_paths)
        detailed_result = {
            "attempt_index": int(attempt_ctx["attempt_index"]),
            "bbox_source": attempt_ctx["bbox_source"],
            "branch_key": attempt_ctx["branch_key"],
            "input_path": str(attempt_ctx["stl_path"]),
            "gt_stl_path": str(attempt_ctx["gt_stl_path"]),
            "output_dir": str(attempt_output_dir),
            "summary": attempt_ctx["summary"],
            "mesh_export": {
                "leaf_mesh_count": len(leaf_mesh_paths),
                "leaf_mesh_paths": leaf_mesh_paths,
                "merged_mesh_path": merged_mesh_path,
                "merge_errors": merge_errors,
            },
            "metrics": metrics,
            "tree": tree,
        }
        result_path.write_text(json.dumps(detailed_result, ensure_ascii=False, indent=2), encoding="utf-8")
        status = "failed" if metrics.get("cd") is None else "success"
        return {
            "status": status,
            "attempt_index": int(attempt_ctx["attempt_index"]),
            "bbox_source": attempt_ctx["bbox_source"],
            "branch_key": attempt_ctx["branch_key"],
            "relative_path": attempt_ctx["relative_path"],
            "input_path": str(attempt_ctx["stl_path"]),
            "output_dir": str(attempt_output_dir),
            "result_path": str(result_path),
            "merged_mesh_path": merged_mesh_path,
            "summary": attempt_ctx["summary"],
            "metrics": metrics,
        }
    except Exception as exc:
        error_path = attempt_output_dir / "error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        detailed_result = {
            "attempt_index": int(attempt_ctx["attempt_index"]),
            "bbox_source": attempt_ctx["bbox_source"],
            "branch_key": attempt_ctx["branch_key"],
            "input_path": str(attempt_ctx["stl_path"]),
            "gt_stl_path": str(attempt_ctx["gt_stl_path"]),
            "output_dir": str(attempt_output_dir),
            "summary": attempt_ctx["summary"],
            "mesh_export": {
                "leaf_mesh_count": 0,
                "leaf_mesh_paths": [],
                "merged_mesh_path": None,
                "merge_errors": [{"node_id": "", "error": str(exc)}],
            },
            "metrics": {
                "gt_stl_path": str(attempt_ctx["gt_stl_path"]),
                "pred_mesh_path": None,
                "cd": None,
                "cd_scaled_x1000": None,
                "iou": None,
                "error": str(exc),
            },
            "tree": attempt_ctx.get("tree"),
            "error_path": str(error_path),
        }
        result_path.write_text(json.dumps(detailed_result, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "status": "failed",
            "attempt_index": int(attempt_ctx["attempt_index"]),
            "bbox_source": attempt_ctx["bbox_source"],
            "branch_key": attempt_ctx["branch_key"],
            "relative_path": attempt_ctx["relative_path"],
            "input_path": str(attempt_ctx["stl_path"]),
            "output_dir": str(attempt_output_dir),
            "result_path": str(result_path),
            "merged_mesh_path": None,
            "summary": attempt_ctx["summary"],
            "metrics": detailed_result["metrics"],
        }


def _record_finalize_result(future, branch_key: str, branch_states_by_key: dict, attempt_results: dict) -> None:
    branch_state = branch_states_by_key[branch_key]
    sample_state = branch_state["sample_state"]
    try:
        result = future.result()
    except Exception as exc:
        error_path = sample_state["sample_output_dir"] / f"attempt_error_{int(time.time())}.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        result = {
            "status": "failed",
            "attempt_index": -1,
            "bbox_source": branch_state["bbox_source"],
            "branch_key": branch_key,
            "relative_path": sample_state["relative_path"],
            "input_path": str(sample_state["stl_path"]),
            "output_dir": str(sample_state["sample_output_dir"]),
            "result_path": "",
            "merged_mesh_path": None,
            "summary": {},
            "metrics": {"cd": None, "cd_scaled_x1000": None, "iou": None, "error": str(exc)},
        }
    sample_state["attempt_results"].append(result)
    attempt_results[branch_key] = result
    _write_sample_summary(sample_state)


def _drain_finalize_futures(finalize_futures: dict, branch_states_by_key: dict, attempt_results: dict) -> None:
    for future, branch_key in list(finalize_futures.items()):
        if not future.done():
            continue
        finalize_futures.pop(future, None)
        _record_finalize_result(future, branch_key, branch_states_by_key, attempt_results)


def _wait_finalize_futures(finalize_futures: dict, branch_states_by_key: dict, attempt_results: dict) -> None:
    for future in as_completed(list(finalize_futures)):
        branch_key = finalize_futures.pop(future)
        _record_finalize_result(future, branch_key, branch_states_by_key, attempt_results)


def _should_retry(attempt_result: dict, retry_cd_threshold: float) -> bool:
    metrics = attempt_result.get("metrics") or {}
    if metrics.get("cd") is None:
        return True
    cd_scaled = metrics.get("cd_scaled_x1000")
    if cd_scaled is None and metrics.get("cd") is not None:
        cd_scaled = float(metrics["cd"]) * 1000.0
    return float(cd_scaled) > float(retry_cd_threshold)


def _run_attempt(
    model,
    processor,
    runtime: dict,
    args,
    branch_states: list[dict],
    attempt_index: int,
    bbox_source: str,
    finalize_executor,
) -> dict:
    attempt_contexts = {}
    branch_states_by_key = {branch_state["branch_key"]: branch_state for branch_state in branch_states}
    for branch_state in branch_states:
        attempt_ctx = _init_attempt_context(branch_state, attempt_index)
        attempt_ctx["frontier"].append(
            {
                "attempt_ctx": attempt_ctx,
                "node_id": "root",
                "depth": 0,
                "dense_cloud": None,
                "parent_children": None,
                "child_name": None,
            }
        )
        attempt_contexts[branch_state["branch_key"]] = attempt_ctx

    finalize_futures = {}
    attempt_results = {}
    inference_pending = set(attempt_contexts.keys())

    while inference_pending:
        current_jobs = []
        for sample_key in list(inference_pending):
            current_jobs.extend(attempt_contexts[sample_key]["frontier"])
            attempt_contexts[sample_key]["frontier"] = []

        if current_jobs:
            _log(
                f"[Attempt {attempt_index}] "
                f"source={bbox_source} "
                f"bbox depth={int(current_jobs[0]['depth'])} "
                f"nodes={len(current_jobs)}"
            )
            next_frontier, code_jobs = _run_bbox_layer(
                model,
                processor,
                runtime,
                args,
                current_jobs,
                attempt_index=attempt_index,
                bbox_source=bbox_source,
            )
            if code_jobs:
                _log(
                    f"[Attempt {attempt_index}] "
                    f"source={bbox_source} "
                    f"code depth={batch_jobs_depth(code_jobs)} "
                    f"stop_nodes={len(code_jobs)}"
                )
                _run_code_layer(
                    model,
                    processor,
                    runtime,
                    args,
                    code_jobs,
                    attempt_index=attempt_index,
                    bbox_source=bbox_source,
                )
            for node_job in next_frontier:
                node_job["attempt_ctx"]["frontier"].append(node_job)

        finished_keys = [
            sample_key
            for sample_key in list(inference_pending)
            if not attempt_contexts[sample_key]["frontier"]
        ]
        for sample_key in finished_keys:
            inference_pending.remove(sample_key)
            attempt_ctx = attempt_contexts[sample_key]
            future = finalize_executor.submit(_finalize_attempt, sample_key, attempt_ctx, args)
            finalize_futures[future] = sample_key

        _drain_finalize_futures(finalize_futures, branch_states_by_key, attempt_results)

    _wait_finalize_futures(finalize_futures, branch_states_by_key, attempt_results)

    return attempt_results


def _write_attempt_summary(output_dir: Path, attempt_index: int, attempt_results: dict, retry_cd_threshold: float) -> dict:
    ordered_results = [
        attempt_results[key]
        for key in sorted(
            attempt_results,
            key=lambda item: (
                item.split("::", 1)[0],
                _BBOX_SOURCE_ORDER_INDEX.get(_normalize_bbox_source(item.split("::", 1)[1] if "::" in item else None), len(_BBOX_SOURCE_ORDER_INDEX)),
            ),
        )
    ]
    report = _build_metrics_report(ordered_results)
    results_by_bbox_source = {}
    metrics_report_by_bbox_source = {}
    retry_candidates_by_bbox_source = {}
    for bbox_source in _BBOX_SOURCE_ORDER:
        source_results = [
            result for result in ordered_results
            if _normalize_bbox_source(result.get("bbox_source")) == bbox_source
        ]
        results_by_bbox_source[bbox_source] = [
            _prune_attempt_result_fields(item) for item in source_results
        ]
        metrics_report_by_bbox_source[bbox_source] = _build_metrics_report(source_results)
        retry_candidates_by_bbox_source[bbox_source] = sorted(
            result.get("branch_key") or f"{result['relative_path']}::{bbox_source}"
            for result in source_results
            if _should_retry(result, retry_cd_threshold)
        )
    retry_candidates = sorted(
        result.get("branch_key") or f"{result['relative_path']}::{_normalize_bbox_source(result.get('bbox_source'))}"
        for result in ordered_results
        if _should_retry(result, retry_cd_threshold)
    )
    payload = {
        "attempt_index": int(attempt_index),
        "sample_count": len({result["relative_path"] for result in ordered_results}),
        "branch_count": len(ordered_results),
        "retry_cd_threshold_x1000": float(retry_cd_threshold),
        "metrics_report": report,
        "metrics_report_by_bbox_source": metrics_report_by_bbox_source,
        "retry_candidates": retry_candidates,
        "retry_candidates_by_bbox_source": retry_candidates_by_bbox_source,
        "results": [_prune_attempt_result_fields(item) for item in ordered_results],
        "results_by_bbox_source": results_by_bbox_source,
    }
    path = output_dir / f"attempt_{attempt_index:02d}_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _normalize_loaded_attempt_result(sample_state: dict, attempt_result: dict) -> dict:
    normalized = dict(attempt_result or {})
    normalized["status"] = normalized.get("status", "failed")
    normalized["attempt_index"] = int(normalized["attempt_index"])
    normalized["bbox_source"] = _normalize_bbox_source(
        normalized.get("bbox_source") or _infer_bbox_source_from_output_dir(normalized.get("output_dir"))
    )
    normalized["branch_key"] = normalized.get("branch_key") or _sample_branch_key(sample_state, normalized["bbox_source"])
    normalized["relative_path"] = normalized.get("relative_path") or sample_state["relative_path"]
    normalized["input_path"] = str(normalized.get("input_path") or sample_state["stl_path"])
    normalized["output_dir"] = str(
        normalized.get("output_dir")
        or (sample_state["sample_output_dir"] / f"attempt_{int(normalized['attempt_index']):02d}_{normalized['bbox_source']}")
    )
    normalized["result_path"] = str(normalized.get("result_path") or "")
    normalized["merged_mesh_path"] = normalized.get("merged_mesh_path")
    normalized["summary"] = normalized.get("summary") or {}
    normalized["metrics"] = normalized.get("metrics") or {}
    return normalized


def _load_sample_attempt_results(sample_state: dict, *, before_attempt: int | None = None) -> list[dict]:
    path = _sample_summary_path(sample_state)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _log(f"[Resume] Failed to read {path}: {exc}")
        return []

    loaded = []
    for attempt_result in payload.get("attempts", []):
        try:
            normalized = _normalize_loaded_attempt_result(sample_state, attempt_result)
        except Exception as exc:
            _log(f"[Resume] Skip malformed attempt in {path}: {exc}")
            continue
        if before_attempt is not None and int(normalized["attempt_index"]) >= int(before_attempt):
            continue
        loaded.append(normalized)
    loaded.sort(key=lambda item: int(item["attempt_index"]))
    return loaded


def _build_attempt_results_by_index(sample_states: list[dict]) -> dict[int, dict]:
    results_by_index = defaultdict(dict)
    for sample_state in sample_states:
        for attempt_result in sample_state["attempt_results"]:
            branch_key = attempt_result.get("branch_key") or _sample_branch_key(
                sample_state,
                attempt_result.get("bbox_source"),
            )
            results_by_index[int(attempt_result["attempt_index"])][branch_key] = attempt_result
    return dict(results_by_index)


def _load_attempt_summary(output_dir: Path, attempt_index: int) -> dict | None:
    path = output_dir / f"attempt_{int(attempt_index):02d}_summary.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _log(f"[Resume] Failed to read {path}: {exc}")
        return None
    return payload if isinstance(payload, dict) else None


def _restore_resume_state(
    sample_states: list[dict],
    branch_states: list[dict],
    output_dir: Path,
    resume_from_attempt: int,
    retry_cd_threshold: float,
) -> tuple[list[dict], list[dict]]:
    for sample_state in sample_states:
        sample_state["attempt_results"] = _load_sample_attempt_results(
            sample_state,
            before_attempt=resume_from_attempt,
        )

    attempt_results_by_index = _build_attempt_results_by_index(sample_states)
    attempt_summaries = []
    for attempt_index in range(1, int(resume_from_attempt)):
        summary = _load_attempt_summary(output_dir, attempt_index)
        if summary is None and attempt_index in attempt_results_by_index:
            summary = _write_attempt_summary(
                output_dir=output_dir,
                attempt_index=attempt_index,
                attempt_results=attempt_results_by_index[attempt_index],
                retry_cd_threshold=float(retry_cd_threshold),
            )
        if summary is not None:
            attempt_summaries.append(summary)

    pending_keys = {branch_state["branch_key"] for branch_state in branch_states}
    missing_history = []
    for attempt_index in range(1, int(resume_from_attempt)):
        next_pending = set()
        for branch_state in branch_states:
            branch_key = branch_state["branch_key"]
            if branch_key not in pending_keys:
                continue
            result = attempt_results_by_index.get(attempt_index, {}).get(branch_key)
            if result is None:
                missing_history.append((branch_key, attempt_index))
                next_pending.add(branch_key)
                continue
            if _should_retry(result, float(retry_cd_threshold)):
                next_pending.add(branch_key)
        pending_keys = next_pending

    if missing_history:
        preview = ", ".join(
            f"{branch_key}@{attempt_index}"
            for branch_key, attempt_index in missing_history[:8]
        )
        suffix = "" if len(missing_history) <= 8 else " ..."
        _log(
            f"[Resume] Missing historical attempts for {len(missing_history)} sample-attempt pairs; "
            f"they will be rerun from attempt {int(resume_from_attempt)}. {preview}{suffix}"
        )

    pending_branch_states = [
        branch_state
        for branch_state in branch_states
        if branch_state["branch_key"] in pending_keys
    ]
    return pending_branch_states, attempt_summaries


def _build_final_summary(sample_states: list[dict], bbox_sources: list[str], output_dir: Path, attempt_summaries: list[dict]) -> dict:
    final_results = []
    results_by_bbox_source = {bbox_source: [] for bbox_source in bbox_sources}
    for sample_state in sample_states:
        summary = _write_sample_summary(sample_state)
        for bbox_source in bbox_sources:
            best_metrics = (summary.get("best_metrics_by_bbox_source") or {}).get(
                bbox_source,
                {"best_attempt_index": None, "cd": None, "cd_scaled_x1000": None, "iou": None},
            )
            source_result = {
                "bbox_source": bbox_source,
                "relative_path": sample_state["relative_path"],
                "input_path": str(sample_state["stl_path"]),
                "summary_path": str(_sample_summary_path(sample_state)),
                "best_attempt_index": best_metrics["best_attempt_index"],
                "metrics": {
                    "cd": best_metrics["cd"],
                    "cd_scaled_x1000": best_metrics["cd_scaled_x1000"],
                    "iou": best_metrics["iou"],
                },
            }
            results_by_bbox_source[bbox_source].append(source_result)
            final_results.append(source_result)
    for bbox_source in bbox_sources:
        results_by_bbox_source[bbox_source].sort(key=lambda item: item["relative_path"])
    final_results.sort(
        key=lambda item: (
            item["relative_path"],
            _BBOX_SOURCE_ORDER_INDEX.get(_normalize_bbox_source(item.get("bbox_source")), len(_BBOX_SOURCE_ORDER_INDEX)),
        )
    )
    report = _build_metrics_report(final_results)
    metrics_report_by_bbox_source = {
        bbox_source: _build_metrics_report(results_by_bbox_source[bbox_source])
        for bbox_source in bbox_sources
    }
    payload = {
        "output_dir": str(output_dir),
        "attempt_summaries": attempt_summaries,
        "metrics_report": report,
        "metrics_report_by_bbox_source": metrics_report_by_bbox_source,
        "results": final_results,
        "results_by_bbox_source": results_by_bbox_source,
    }
    path = output_dir / "batch_recursive_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Layered batch recursive STL inference")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save-config-name", type=str, default="run_config.yaml")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gt-dir", type=str, default=None)
    parser.add_argument("--recursive", type=str2bool, default=True)
    parser.add_argument("--processor-path", type=str, default="auto")
    parser.add_argument("--task-mode", type=str, default=None)
    parser.add_argument("--pc-encoder-type", type=str, default=None)
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--n-point-tokens", type=int, default=None)
    parser.add_argument("--part-points-per-bbox", type=int, default=None)
    parser.add_argument("--normalize-std", type=float, default=None)
    parser.add_argument("--utonia-scale", type=float, default=None)
    parser.add_argument("--utonia-normalize-coord", type=str2bool, default=None)
    parser.add_argument("--utonia-use-normal", type=str2bool, default=None)
    parser.add_argument("--utonia-use-fourier-fusion", type=str2bool, default=True)
    parser.add_argument("--normalization-mode", type=str, default="train0311")
    parser.add_argument("--max-new-tokens-bbox", type=int, default=3768)
    parser.add_argument("--max-new-tokens-code", type=int, default=3768)
    parser.add_argument("--recursive-bbox-batch-size", type=int, default=128)
    parser.add_argument("--recursive-code-batch-size", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--crop-eps", type=float, default=0.01)
    parser.add_argument("--dense-points", type=int, default=32768)
    parser.add_argument("--eval-n-points", type=int, default=8192)
    parser.add_argument("--mesh-export-timeout-sec", type=float, default=60.0)
    parser.add_argument("--finalize-workers", type=int, default=8)
    parser.add_argument("--recursive-loader-workers", type=int, default=8)
    parser.add_argument("--max-retry-attempts", type=int, default=3)
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--resume-from-attempt", type=int, default=1)
    parser.add_argument("--min-valid-points", type=int, default=2)
    parser.add_argument("--min-valid-span", type=float, default=1e-8)
    parser.add_argument(
        "--bbox-sources",
        type=str,
        default="text",
        help="BBox prediction sources: text (default), both, bbox_head, or a comma-separated list.",
    )
    parser.add_argument(
        "--retry-cd-threshold",
        type=float,
        default=0.1,
        help="Retry when cd*1000 is larger than this threshold.",
    )
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    args = parser.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    config, loaded_config_path = _load_train_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint_path,
        save_config_name=args.save_config_name,
    )
    args.config = loaded_config_path

    common_cfg = get_nested(config, ["common"], {}) or {}
    common_model_cfg = get_nested(common_cfg, ["model"], {}) or {}
    train_cfg = get_nested(config, ["train"], {}) or {}
    train_task_cfg = get_nested(train_cfg, ["task"], {}) or {}
    train_contrastive_cfg = get_nested(train_cfg, ["contrastive"], {}) or {}
    train_bbox_grounding_cfg = get_nested(train_cfg, ["bbox_grounding"], {}) or {}
    if args.task_mode is None:
        args.task_mode = "two_stage" if bool(train_task_cfg.get("two_stage", False)) else "multitask"
    if args.pc_encoder_type is None:
        args.pc_encoder_type = common_model_cfg.get("pc_encoder_type", "pointbert")
    args.contrastive_enabled = bool(train_contrastive_cfg.get("enabled", False))
    args.bbox_grounding_enabled = bool(train_bbox_grounding_cfg.get("enabled", False))

    args = recursive_infer._apply_yaml_defaults(args)
    recursive_infer._validate_args(args)
    if int(args.resume_from_attempt) <= 0:
        raise ValueError("--resume-from-attempt must be >= 1.")
    if int(args.resume_from_attempt) > int(args.max_retry_attempts):
        raise ValueError("--resume-from-attempt cannot be larger than --max-retry-attempts.")
    if int(args.min_valid_points) <= 0:
        raise ValueError("--min-valid-points must be positive.")
    if float(args.min_valid_span) < 0:
        raise ValueError("--min-valid-span must be >= 0.")

    input_dir = Path(args.input_dir).resolve()
    gt_dir = None if args.gt_dir is None else Path(args.gt_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stl_files = _collect_stl_files(input_dir, recursive=bool(args.recursive))
    sample_states = [
        _prepare_sample_state(
            stl_path=stl_path,
            input_dir=input_dir,
            gt_dir=gt_dir,
            output_dir=output_dir,
        )
        for stl_path in stl_files
    ]
    _log(f"[Data] Found {len(sample_states)} STL files")

    _log("[Model] Loading model...")
    model, _, checkpoint_hints, load_attempt = recursive_infer.load_model_with_fallbacks(args)
    model.eval()
    runtime = recursive_infer.resolve_runtime_settings(model, args)
    args.use_point_normals = bool(runtime["pc_encoder_type"] == "utonia" and runtime["utonia_use_normal"])
    runtime_summary = {
        "task_mode": runtime["task_mode"],
        "pc_encoder_type": runtime["pc_encoder_type"],
        "n_point_tokens": runtime["n_point_tokens"],
        "part_points_per_bbox": runtime["part_points_per_bbox"],
        "bbox_grounding_enabled": bool(getattr(model, "bbox_grounding_enabled", False)),
        "contrastive_enabled": bool(getattr(model, "contrastive_enabled", False)),
        "utonia_use_normal": runtime["utonia_use_normal"],
        "use_point_normals": bool(args.use_point_normals),
        "normalize_std": float(args.normalize_std),
        "load_attempt": int(load_attempt) + 1,
        "config_path": loaded_config_path,
        "execution_device": str(_resolve_execution_device(model)),
    }
    _log("[Runtime]", json.dumps(runtime_summary, ensure_ascii=False))
    if checkpoint_hints.get("config_found"):
        _log("[Runtime] Checkpoint config detected.")

    processor_path = recursive_infer.resolve_processor_path(args)
    _log(f"[Processor] {processor_path}")
    processor = AutoProcessor.from_pretrained(
        processor_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side="left",
        use_fast=True,
    )
    _ensure_bbox_grounding_box_token_id(model, processor)
    bbox_sources = _parse_bbox_sources_arg(args.bbox_sources)
    branch_states = _build_branch_states(sample_states, bbox_sources=bbox_sources)
    _log(
        f"[BBoxModes] enabled={bool(getattr(model, 'bbox_grounding_enabled', False))} "
        f"sources={bbox_sources}"
    )

    if bool(args.resume):
        pending_branch_states, attempt_summaries = _restore_resume_state(
            sample_states=sample_states,
            branch_states=branch_states,
            output_dir=output_dir,
            resume_from_attempt=int(args.resume_from_attempt),
            retry_cd_threshold=float(args.retry_cd_threshold),
        )
        _log(
            f"[Resume] start_attempt={int(args.resume_from_attempt)} "
            f"loaded_previous_attempts={len(attempt_summaries)} "
            f"pending_branches={len(pending_branch_states)}"
        )
    else:
        pending_branch_states = list(branch_states)
        attempt_summaries = []

    start_attempt = int(args.resume_from_attempt) if bool(args.resume) else 1
    with ThreadPoolExecutor(max_workers=max(1, int(args.finalize_workers))) as finalize_executor:
        for attempt_index in range(start_attempt, int(args.max_retry_attempts) + 1):
            if not pending_branch_states:
                break
            _log(f"[Attempt {attempt_index}] branch_count={len(pending_branch_states)}")
            attempt_results = {}
            for bbox_source in bbox_sources:
                source_branch_states = [
                    branch_state
                    for branch_state in pending_branch_states
                    if branch_state["bbox_source"] == bbox_source
                ]
                if not source_branch_states:
                    continue
                _log(
                    f"[Attempt {attempt_index}] source={bbox_source} "
                    f"sample_count={len(source_branch_states)}"
                )
                attempt_results.update(
                    _run_attempt(
                        model=model,
                        processor=processor,
                        runtime=runtime,
                        args=args,
                        branch_states=source_branch_states,
                        attempt_index=attempt_index,
                        bbox_source=bbox_source,
                        finalize_executor=finalize_executor,
                    )
                )
            attempt_summary = _write_attempt_summary(
                output_dir=output_dir,
                attempt_index=attempt_index,
                attempt_results=attempt_results,
                retry_cd_threshold=float(args.retry_cd_threshold),
            )
            attempt_summaries.append(attempt_summary)
            _print_metrics_report(attempt_summary["metrics_report"])
            pending_keys = {
                sample_key
                for sample_key, result in attempt_results.items()
                if _should_retry(result, float(args.retry_cd_threshold))
            }
            pending_branch_states = [
                branch_state for branch_state in branch_states
                if branch_state["branch_key"] in pending_keys
            ]

    final_summary = _build_final_summary(sample_states, bbox_sources, output_dir, attempt_summaries)
    final_summary["runtime"] = runtime_summary
    final_summary["input_dir"] = str(input_dir)
    final_summary["gt_dir"] = None if gt_dir is None else str(gt_dir)
    final_summary["checkpoint_path"] = str(Path(args.checkpoint_path).resolve())
    summary_path = output_dir / "batch_recursive_summary.json"
    summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"[Done] Saved summary to {summary_path}")
    _print_metrics_report(final_summary["metrics_report"])

    
if __name__ == "__main__":
    main()

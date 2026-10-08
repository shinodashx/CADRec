import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.ops import sample_farthest_points
from transformers import AutoProcessor

from inference import (
    TASK_DESCRIPTIONS,
    _extract_json_block,
    load_model_with_fallbacks,
    resolve_processor_path,
    resolve_runtime_settings,
    str2bool,
    normalize_pc_encoder_type,
    normalize_task_mode,
    _extract_points_and_normals_from_geometry,
)
from yaml_config import load_config_with_defaults, get_nested

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_DIGIT_SUFFIX_PATTERN = re.compile(r"(\d+)$")
_BBOX_LABEL_TOKEN_PATTERN = re.compile(r"<(SPLIT|STOP)>|\b(SPLIT|STOP)\b")
_BBOX_TOKEN_PATTERN = re.compile(r"<BOX>|\bBOX\b")


@dataclass
class PointCloudChunk:
    points: np.ndarray
    normals: np.ndarray | None = None
    to_global_scale: float = 1.0
    to_global_shift: np.ndarray | None = None


def _sort_named_boxes(boxes: dict) -> list[tuple[str, object]]:
    def sort_key(item):
        name = str(item[0])
        match = _DIGIT_SUFFIX_PATTERN.search(name)
        if match:
            return (0, int(match.group(1)), name)
        return (1, name)

    return sorted(boxes.items(), key=sort_key)


def _normalize_points_to_range(points: np.ndarray, target_range: float = 100.0, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(points, dtype=np.float32)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    half_range = np.max(maxs - mins) / 2.0
    if half_range < eps:
        half_range = float(eps)
        normalized = np.zeros_like(points, dtype=np.float32)
    else:
        normalized = ((points - center) / half_range * float(target_range)).astype(np.float32)
    return normalized, center.astype(np.float32), float(half_range)


def _fps_sample(points: np.ndarray, target_count: int, normals: np.ndarray | None = None) -> PointCloudChunk:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        raise ValueError(f"Points must have shape [N, >=3], got {points.shape}")
    points = points[:, :3]
    target_count = int(target_count)
    if target_count <= 0:
        raise ValueError("target_count must be positive.")

    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.ndim != 2 or normals.shape[0] != points.shape[0] or normals.shape[1] < 3:
            raise ValueError("Normals must have shape [N, >=3] and match points.")
        normals = normals[:, :3]

    if points.shape[0] < target_count:
        repeats = (target_count + points.shape[0] - 1) // points.shape[0]
        sample_ids = np.tile(np.arange(points.shape[0], dtype=np.int64), repeats)[:target_count]
    elif points.shape[0] == target_count:
        sample_ids = np.arange(points.shape[0], dtype=np.int64)
    else:
        _, fps_ids = sample_farthest_points(
            torch.tensor(points, dtype=torch.float32).unsqueeze(0),
            K=target_count)
        sample_ids = fps_ids[0].cpu().numpy().astype(np.int64)

    sampled_points = points[sample_ids].astype(np.float32, copy=False)
    sampled_normals = None if normals is None else normals[sample_ids].astype(np.float32, copy=False)
    return PointCloudChunk(points=sampled_points, normals=sampled_normals)


def _box_array_to_min_max(box_value) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(box_value, dtype=np.float32).reshape(-1)
    if arr.size < 6:
        raise ValueError(f"Invalid bbox value: {box_value}")
    bbox_min = np.minimum(arr[:3], arr[3:6]).astype(np.float32)
    bbox_max = np.maximum(arr[:3], arr[3:6]).astype(np.float32)
    return bbox_min, bbox_max


def _crop_and_renormalize(node_dense: PointCloudChunk, bbox_value, crop_eps: float) -> PointCloudChunk:
    bbox_min, bbox_max = _box_array_to_min_max(bbox_value)
    mask = np.all(
        (node_dense.points >= (bbox_min - crop_eps)) &
        (node_dense.points <= (bbox_max + crop_eps)),
        axis=1)

    cropped_points = node_dense.points[mask]
    cropped_normals = None if node_dense.normals is None else node_dense.normals[mask]

    if cropped_points.shape[0] == 0:
        center = (bbox_min + bbox_max) / 2.0
        distances = np.linalg.norm(node_dense.points - center[None, :], axis=1)
        fallback_count = min(max(1, 128), node_dense.points.shape[0])
        nearest_ids = np.argsort(distances)[:fallback_count]
        cropped_points = node_dense.points[nearest_ids]
        cropped_normals = None if node_dense.normals is None else node_dense.normals[nearest_ids]

    cropped_points, local_center, local_half_range = _normalize_points_to_range(cropped_points, target_range=100.0)
    if cropped_normals is not None:
        cropped_normals = np.asarray(cropped_normals, dtype=np.float32)
    parent_scale = float(node_dense.to_global_scale)
    parent_shift = np.zeros(3, dtype=np.float32) if node_dense.to_global_shift is None else np.asarray(node_dense.to_global_shift, dtype=np.float32)
    child_scale = parent_scale * (local_half_range / 100.0)
    child_shift = local_center * parent_scale + parent_shift
    return PointCloudChunk(
        points=cropped_points.astype(np.float32),
        normals=cropped_normals,
        to_global_scale=float(child_scale),
        to_global_shift=child_shift.astype(np.float32))


def _load_npy_point_cloud(path: str, use_point_normals: bool) -> PointCloudChunk:
    point_array = np.load(path)
    point_array = np.asarray(point_array, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[0] == 0 or point_array.shape[1] < 3:
        raise ValueError(f"NPY point cloud must have shape [N, >=3]: {path}")
    points = point_array[:, :3].astype(np.float32, copy=False)
    normals = None
    if use_point_normals:
        if point_array.shape[1] < 6:
            raise ValueError(f"Expected xyz+normal npy with shape [N, >=6]: {path}")
        normals = point_array[:, 3:6].astype(np.float32, copy=False)
    return PointCloudChunk(
        points=points,
        normals=normals,
        to_global_scale=1.0,
        to_global_shift=np.zeros(3, dtype=np.float32))


def _load_ply_point_cloud(path: str, use_point_normals: bool) -> PointCloudChunk:
    geometry = trimesh.load(path, process=False)
    points, normals = _extract_points_and_normals_from_geometry(
        geometry,
        use_point_normals=use_point_normals,
    )
    points = np.asarray(points, dtype=np.float32)
    loaded_normals = None
    if use_point_normals:
        if normals is None:
            loaded_normals = np.zeros_like(points, dtype=np.float32)
        else:
            loaded_normals = np.asarray(normals, dtype=np.float32)
    return PointCloudChunk(
        points=points,
        normals=loaded_normals,
        to_global_scale=1.0,
        to_global_shift=np.zeros(3, dtype=np.float32))


def _load_stl_point_cloud(path: str, dense_points: int, use_point_normals: bool) -> PointCloudChunk:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"Empty trimesh scene: {path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    sampled_points, face_ids = trimesh.sample.sample_surface(mesh, int(dense_points))
    sampled_points = np.asarray(sampled_points, dtype=np.float32)
    sampled_points, _, _ = _normalize_points_to_range(sampled_points, target_range=100.0)

    sampled_normals = None
    if use_point_normals:
        if getattr(mesh, "face_normals", None) is not None and len(mesh.face_normals) > 0:
            sampled_normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
        else:
            sampled_normals = np.zeros_like(sampled_points, dtype=np.float32)
    return PointCloudChunk(
        points=sampled_points,
        normals=sampled_normals,
        to_global_scale=1.0,
        to_global_shift=np.zeros(3, dtype=np.float32))


def _resolve_input_path(args) -> tuple[str, str]:
    candidates = [
        ("npy", args.npy_path),
        ("ply", args.ply_path),
        ("stl", args.stl_path),
    ]
    active = [(kind, path) for kind, path in candidates if path]
    if len(active) != 1:
        raise ValueError("Exactly one of --npy-path, --ply-path, or --stl-path must be provided.")
    kind, path = active[0]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    return kind, path


def _build_root_dense_cloud(args, use_point_normals: bool) -> tuple[str, PointCloudChunk]:
    input_kind, input_path = _resolve_input_path(args)
    if input_kind == "npy":
        dense = _load_npy_point_cloud(input_path, use_point_normals=use_point_normals)
    elif input_kind == "ply":
        dense = _load_ply_point_cloud(input_path, use_point_normals=use_point_normals)
    else:
        dense = _load_stl_point_cloud(
            input_path,
            dense_points=args.dense_points,
            use_point_normals=use_point_normals)
    return input_path, dense


def _sanitize_node_id(node_id: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(node_id)).strip("_")
    return safe or "root"


def _batched(items, batch_size: int):
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _resolve_inference_batch_size(args, attr_name: str, default: int = 1) -> int:
    value = getattr(args, attr_name, default)
    if value is None:
        value = default
    value = int(value)
    return max(value, 1)


def _prepare_bbox_batch(
    sampled: PointCloudChunk,
    node_id: str,
    normalize_std: float,
) -> list[dict]:
    batch_item = {
        "point_cloud": (sampled.points / float(normalize_std)).astype(np.float32),
        "description": TASK_DESCRIPTIONS["bbox"],
        "task_type": "bbox",
        "file_name": _sanitize_node_id(node_id),
    }
    if sampled.normals is not None:
        batch_item["point_normal"] = sampled.normals.astype(np.float32)
    return [batch_item]


def _prepare_code_batch(
    sampled: PointCloudChunk,
    node_id: str,
    normalize_std: float,
) -> list[dict]:
    batch_item = {
        "point_cloud": (sampled.points / float(normalize_std)).astype(np.float32),
        "description": TASK_DESCRIPTIONS["code"],
        "task_type": "code",
        "file_name": _sanitize_node_id(node_id),
    }
    if sampled.normals is not None:
        batch_item["point_normal"] = sampled.normals.astype(np.float32)
    return [batch_item]


def _infer_label(parsed_json: dict) -> str:
    if not isinstance(parsed_json, dict):
        return ""
    label = str(parsed_json.get("label", "")).strip().upper()
    if label in {"SPLIT", "STOP"}:
        return label
    if isinstance(parsed_json.get("parts"), dict):
        return "SPLIT"
    if isinstance(parsed_json.get("steps"), dict):
        return "STOP"
    return label


def _resolve_execution_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped in hf_device_map.values():
            if isinstance(mapped, str) and mapped.startswith("cuda"):
                return torch.device(mapped)
            if isinstance(mapped, int):
                return torch.device(f"cuda:{mapped}")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _ensure_bbox_grounding_box_token_id(model, processor) -> bool:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return False
    token = "<BOX>"
    token_id = tokenizer.convert_tokens_to_ids(token)
    vocab = tokenizer.get_vocab()
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None:
        return False
    if unk_token_id is not None and int(token_id) == int(unk_token_id) and token not in vocab:
        return False
    model.bbox_grounding_box_token_id = int(token_id)
    return True


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
            bbox_min, bbox_max = _box_array_to_min_max(value)
        except Exception:
            continue
        cleaned_container[str(name)] = np.concatenate([bbox_min, bbox_max], axis=0).astype(np.float32).tolist()

    model_bbox_value = normalized.get("model_bbox")
    if model_bbox_value is not None:
        try:
            bbox_min, bbox_max = _box_array_to_min_max(model_bbox_value)
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


def _rebuild_bbox_prediction(output_text: str, head_prediction, normalize_std: float):
    parsed_json = _convert_to_legacy_bbox_json(_extract_json_block(output_text))
    parsed_label = _infer_label(parsed_json) if isinstance(parsed_json, dict) else ""
    if parsed_label in {"SPLIT", "STOP"}:
        return parsed_json

    label = _infer_bbox_label_from_text(output_text)
    if label not in {"SPLIT", "STOP"}:
        return parsed_json

    box_values = _denormalize_predicted_boxes(
        None if not isinstance(head_prediction, dict) else head_prediction.get("boxes"),
        normalize_std=normalize_std,
    )
    n_box_tokens = _count_box_tokens(output_text)
    if n_box_tokens > 0:
        box_values = box_values[:n_box_tokens]

    if label == "SPLIT" and not box_values:
        return None

    container_key = "parts" if label == "SPLIT" else "steps"
    item_prefix = "part" if label == "SPLIT" else "step"
    normalized = {
        "label": label,
        "model_bbox": _compute_model_bbox_from_boxes(box_values),
        container_key: {
            f"{item_prefix}_{idx}": box_value
            for idx, box_value in enumerate(box_values, start=1)
        },
    }
    return _convert_to_legacy_bbox_json(normalized)


def _save_node_outputs(output_dir: Path, node_id: str, raw_text: str, parsed_json) -> tuple[str, str]:
    node_dir = output_dir / "nodes"
    node_dir.mkdir(parents=True, exist_ok=True)
    safe_id = _sanitize_node_id(node_id)
    raw_path = node_dir / f"{safe_id}_bbox.txt"
    json_path = node_dir / f"{safe_id}_bbox.json"

    raw_path.write_text(raw_text.strip(), encoding="utf-8")
    if parsed_json is not None:
        json_path.write_text(json.dumps(parsed_json, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        json_path.write_text(raw_text.strip(), encoding="utf-8")
    return str(raw_path), str(json_path)


def _iter_bbox_items(parsed_json: dict):
    if not isinstance(parsed_json, dict):
        return
    if "model_bbox" in parsed_json:
        yield "model_bbox", "model_bbox", parsed_json["model_bbox"]
    for container_key in ("parts", "steps"):
        container = parsed_json.get(container_key)
        if isinstance(container, dict):
            for name, bbox_value in _sort_named_boxes(container):
                yield container_key, str(name), bbox_value


def _box_mesh_from_value(box_value) -> trimesh.Trimesh:
    bbox_min, bbox_max = _box_array_to_min_max(box_value)
    center = (bbox_min + bbox_max) / 2.0
    extents = np.maximum(bbox_max - bbox_min, 1e-3).astype(np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = center.astype(np.float32)
    return trimesh.creation.box(extents=extents, transform=transform)


def _transform_box_to_global(box_value, scale: float, shift) -> list[float]:
    bbox_min, bbox_max = _box_array_to_min_max(box_value)
    shift = np.asarray(shift, dtype=np.float32)
    global_min = bbox_min * float(scale) + shift
    global_max = bbox_max * float(scale) + shift
    return np.concatenate([global_min, global_max], axis=0).astype(np.float32).tolist()


def _box_edges_from_value(box_value):
    bbox_min, bbox_max = _box_array_to_min_max(box_value)
    x0, y0, z0 = bbox_min.tolist()
    x1, y1, z1 = bbox_max.tolist()
    corners = np.asarray([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return corners, edges


def _set_axes_equal(ax, points: np.ndarray):
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius < 1e-6:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _render_bbox_matplotlib(image_path: Path, title: str, point_cloud: np.ndarray | None, box_items: list[tuple[str, str, object]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    stacked_points = []
    if point_cloud is not None and np.asarray(point_cloud).size > 0:
        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            s=1.0,
            c="lightgray",
            alpha=0.25,
            depthshade=False)
        stacked_points.append(point_cloud)

    cmap = plt.get_cmap("tab20")
    for idx, (container_key, name, box_value) in enumerate(box_items):
        corners, edges = _box_edges_from_value(box_value)
        color = cmap(idx % 20)
        for start_idx, end_idx in edges:
            seg = corners[[start_idx, end_idx]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=1.5)
        center = corners.mean(axis=0)
        ax.text(center[0], center[1], center[2], str(name), color=color, fontsize=7)
        stacked_points.append(corners)

    if stacked_points:
        _set_axes_equal(ax, np.concatenate(stacked_points, axis=0))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(image_path, dpi=220)
    plt.close(fig)


def _collect_split_boxes(tree: dict, collected: list[tuple[str, str, object]] | None = None):
    if collected is None:
        collected = []
    if not isinstance(tree, dict):
        return collected

    if tree.get("status") == "split":
        prediction = tree.get("prediction")
        node_id = str(tree.get("node_id", "node"))
        scale = float(tree.get("to_global_scale", 1.0))
        shift = np.asarray(tree.get("to_global_shift", [0.0, 0.0, 0.0]), dtype=np.float32)
        if isinstance(prediction, dict):
            parts = prediction.get("parts")
            if isinstance(parts, dict):
                for part_name, part_box in _sort_named_boxes(parts):
                    collected.append((
                        "parts",
                        f"{node_id}:{part_name}",
                        _transform_box_to_global(part_box, scale=scale, shift=shift),
                    ))

    for child in (tree.get("children") or {}).values():
        _collect_split_boxes(child, collected=collected)
    return collected


def _save_aggregate_split_visualization(output_dir: Path, root_point_cloud: np.ndarray, tree: dict) -> dict:
    split_boxes = _collect_split_boxes(tree, collected=[])
    image_path = None
    json_path = None
    render_error = None

    if split_boxes:
        try:
            image_file = output_dir / "all_split_parts_global.png"
            _render_bbox_matplotlib(
                image_path=image_file,
                title="All split parts on full point cloud",
                point_cloud=root_point_cloud,
                box_items=split_boxes)
            image_path = str(image_file)

            json_file = output_dir / "all_split_parts_global.json"
            json_payload = {
                "parts": {
                    name: box_value
                    for _, name, box_value in split_boxes
                }
            }
            json_file.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            json_path = str(json_file)
        except Exception as exc:
            render_error = str(exc)

    result = {
        "count": int(len(split_boxes)),
        "image_path": image_path,
        "json_path": json_path,
    }
    if render_error is not None:
        result["render_error"] = render_error
    return result


def _save_leaf_code(output_dir: Path, node_id: str, code_text: str) -> str:
    node_dir = output_dir / "nodes"
    node_dir.mkdir(parents=True, exist_ok=True)
    code_path = node_dir / f"{_sanitize_node_id(node_id)}_code.py"
    code_path.write_text(code_text, encoding="utf-8")
    return str(code_path)


def _save_leaf_mesh(output_dir: Path, node_id: str, mesh: trimesh.Trimesh) -> str:
    node_dir = output_dir / "nodes"
    node_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = node_dir / f"{_sanitize_node_id(node_id)}_mesh.stl"
    mesh.export(mesh_path)
    return str(mesh_path)


def _make_child_node_id(parent_node_id: str, child_name: str, child_index: int) -> str:
    if parent_node_id == "root":
        return f"{child_name}_{child_index}"
    return f"{parent_node_id}/{child_name}_{child_index}"


def _generate_texts(model, processor, batch, runtime, n_points, max_new_tokens):
    from cadrec import collate

    if not batch:
        return []

    model_inputs = collate(
        batch,
        processor=processor,
        n_points=n_points,
        n_point_tokens=runtime["n_point_tokens"],
        n_part_point_tokens=runtime["part_points_per_bbox"],
        eval=True,
        pc_encoder_type=runtime["pc_encoder_type"],
        utonia_scale=runtime["utonia_scale"],
        utonia_normalize_coord=runtime["utonia_normalize_coord"],
        utonia_use_normal=runtime["utonia_use_normal"],
    )

    exec_device = _resolve_execution_device(model)
    generate_kwargs = {
        "input_ids": model_inputs["input_ids"].to(exec_device),
        "attention_mask": model_inputs["attention_mask"].to(exec_device),
        "point_clouds": model_inputs["point_clouds"].to(exec_device),
        "is_pc": model_inputs["is_pc"].to(exec_device),
        "is_img": model_inputs["is_img"].to(exec_device),
        "max_new_tokens": max_new_tokens,
    }
    if model_inputs.get("part_point_indices") is not None:
        generate_kwargs["part_point_indices"] = model_inputs["part_point_indices"].to(exec_device)
    if model_inputs.get("n_parts") is not None:
        generate_kwargs["n_parts"] = model_inputs["n_parts"].to(exec_device)
    if model_inputs.get("is_step_pc") is not None:
        generate_kwargs["is_step_pc"] = model_inputs["is_step_pc"].to(exec_device)
    if model_inputs.get("pixel_values_videos") is not None:
        generate_kwargs["pixel_values_videos"] = model_inputs["pixel_values_videos"].to(exec_device)
    if model_inputs.get("video_grid_thw") is not None:
        generate_kwargs["video_grid_thw"] = model_inputs["video_grid_thw"].to(exec_device)

    bbox_grounding_ready = bool(getattr(model, "bbox_grounding_enabled", False)) and _ensure_bbox_grounding_box_token_id(model, processor)
    if bbox_grounding_ready:
        generate_kwargs["bbox_grounding_infer"] = True

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=not bbox_grounding_ready,
        clean_up_tokenization_spaces=False,
    )
    if not bbox_grounding_ready:
        return output_texts

    append_len = int(generated_ids.shape[1] - model_inputs["input_ids"].shape[1])
    if append_len > 0:
        generated_mask = torch.ones(
            model_inputs["attention_mask"].shape[0],
            append_len,
            dtype=model_inputs["attention_mask"].dtype,
            device=exec_device,
        )
        full_attention_mask = torch.cat([model_inputs["attention_mask"].to(exec_device), generated_mask], dim=1)
    else:
        full_attention_mask = model_inputs["attention_mask"].to(exec_device)

    forward_kwargs = {
        "input_ids": generated_ids,
        "attention_mask": full_attention_mask,
        "point_clouds": model_inputs["point_clouds"].to(exec_device),
        "is_pc": model_inputs["is_pc"].to(exec_device),
        "is_img": model_inputs["is_img"].to(exec_device),
        "bbox_grounding_infer": True,
    }
    for key in ("pc_type_ids", "part_point_indices", "n_parts", "is_step_pc", "pixel_values_videos", "video_grid_thw"):
        if model_inputs.get(key) is not None:
            forward_kwargs[key] = model_inputs[key].to(exec_device)

    with torch.no_grad():
        model(**forward_kwargs)

    head_predictions = getattr(model, "latest_bbox_grounding_predictions", None)
    if not isinstance(head_predictions, list):
        return output_texts

    rebuilt = []
    for output_text, head_prediction in zip(output_texts, head_predictions):
        parsed = _rebuild_bbox_prediction(
            output_text,
            head_prediction=head_prediction,
            normalize_std=100.0,
        )
        rebuilt.append(json.dumps(parsed, ensure_ascii=False) if parsed is not None else output_text)
    return rebuilt


def infer_recursive_tree(
    model,
    processor,
    runtime: dict,
    dense_cloud: PointCloudChunk,
    node_id: str,
    depth: int,
    args,
    summary: dict,
) -> dict:
    bbox_batch_size = _resolve_inference_batch_size(args, "recursive_bbox_batch_size", default=1)
    code_batch_size = _resolve_inference_batch_size(args, "recursive_code_batch_size", default=1)

    root_result = None
    pending_leaf_jobs = []
    frontier = [{
        "dense_cloud": dense_cloud,
        "node_id": node_id,
        "depth": int(depth),
        "parent_children": None,
        "child_name": None,
    }]

    while frontier:
        next_frontier = []
        for node_batch in _batched(frontier, bbox_batch_size):
            prepared = []
            for node_state in node_batch:
                sampled = _fps_sample(
                    node_state["dense_cloud"].points,
                    target_count=args.n_points,
                    normals=node_state["dense_cloud"].normals if args.use_point_normals else None)
                bbox_item = _prepare_bbox_batch(
                    sampled=sampled,
                    node_id=node_state["node_id"],
                    normalize_std=args.normalize_std)[0]
                prepared.append((node_state, sampled, bbox_item))

            output_texts = _generate_texts(
                model=model,
                processor=processor,
                batch=[item[2] for item in prepared],
                runtime=runtime,
                n_points=int(prepared[0][1].points.shape[0]),
                max_new_tokens=args.max_new_tokens_bbox)

            for (node_state, sampled, _), output_text in zip(prepared, output_texts):
                parsed_json = _extract_json_block(output_text)
                raw_path, json_path = _save_node_outputs(
                    args.output_dir,
                    node_state["node_id"],
                    output_text,
                    parsed_json)

                result = {
                    "node_id": node_state["node_id"],
                    "depth": int(node_state["depth"]),
                    "dense_point_count": int(node_state["dense_cloud"].points.shape[0]),
                    "input_point_count": int(sampled.points.shape[0]),
                    "to_global_scale": float(node_state["dense_cloud"].to_global_scale),
                    "to_global_shift": (
                        np.zeros(3, dtype=np.float32)
                        if node_state["dense_cloud"].to_global_shift is None
                        else np.asarray(node_state["dense_cloud"].to_global_shift, dtype=np.float32)
                    ).tolist(),
                    "raw_output_path": raw_path,
                    "json_output_path": json_path,
                    "prediction": parsed_json,
                }

                if node_state["parent_children"] is None:
                    root_result = result
                else:
                    node_state["parent_children"][node_state["child_name"]] = result

                summary["total_nodes"] += 1

                if parsed_json is None:
                    summary["invalid_nodes"] += 1
                    result["status"] = "invalid_json"
                    continue

                label = _infer_label(parsed_json)
                result["label"] = label

                if label == "STOP":
                    summary["stop_nodes"] += 1
                    result["status"] = "leaf"
                    result["leaf_code_task"] = TASK_DESCRIPTIONS["code"]
                    pending_leaf_jobs.append({
                        "result": result,
                        "sampled": sampled,
                        "node_id": node_state["node_id"],
                    })
                    continue

                if label != "SPLIT":
                    summary["invalid_nodes"] += 1
                    result["status"] = "unknown_label"
                    continue

                summary["split_nodes"] += 1

                if node_state["depth"] >= args.max_depth:
                    summary["max_depth_stops"] += 1
                    result["status"] = "max_depth_reached"
                    continue

                parts = parsed_json.get("parts")
                if not isinstance(parts, dict) or len(parts) == 0:
                    summary["invalid_nodes"] += 1
                    result["status"] = "split_without_parts"
                    continue

                result["status"] = "split"
                result["children"] = {}
                for child_index, (child_name, child_bbox) in enumerate(_sort_named_boxes(parts), start=1):
                    child_id = _make_child_node_id(node_state["node_id"], child_name, child_index)
                    try:
                        child_dense = _crop_and_renormalize(
                            node_dense=node_state["dense_cloud"],
                            bbox_value=child_bbox,
                            crop_eps=args.crop_eps)
                    except Exception as exc:
                        summary["invalid_nodes"] += 1
                        result["children"][child_name] = {
                            "node_id": child_id,
                            "depth": int(node_state["depth"] + 1),
                            "status": "child_error",
                            "error": str(exc),
                            "source_bbox": child_bbox,
                        }
                        continue

                    next_frontier.append({
                        "dense_cloud": child_dense,
                        "node_id": child_id,
                        "depth": int(node_state["depth"] + 1),
                        "parent_children": result["children"],
                        "child_name": child_name,
                    })

        frontier = next_frontier

    for leaf_batch in _batched(pending_leaf_jobs, code_batch_size):
        output_texts = _generate_texts(
            model=model,
            processor=processor,
            batch=[
                _prepare_code_batch(
                    sampled=leaf_job["sampled"],
                    node_id=leaf_job["node_id"],
                    normalize_std=args.normalize_std)[0]
                for leaf_job in leaf_batch
            ],
            runtime=runtime,
            n_points=int(leaf_batch[0]["sampled"].points.shape[0]),
            max_new_tokens=args.max_new_tokens_code)

        for leaf_job, code_text in zip(leaf_batch, output_texts):
            leaf_job["result"]["leaf_code_path"] = _save_leaf_code(
                args.output_dir,
                leaf_job["node_id"],
                code_text)
            summary["leaf_codes"] += 1

    return root_result


def generate_bbox_text(model, processor, batch, runtime, n_points, max_new_tokens):
    return _generate_texts(
        model=model,
        processor=processor,
        batch=batch,
        runtime=runtime,
        n_points=n_points,
        max_new_tokens=max_new_tokens)[0]


def generate_code_text(model, processor, batch, runtime, n_points, max_new_tokens):
    return _generate_texts(
        model=model,
        processor=processor,
        batch=batch,
        runtime=runtime,
        n_points=n_points,
        max_new_tokens=max_new_tokens)[0]


def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def _cadquery_obj_to_shape(obj):
    import cadquery as cq

    if obj is None:
        return None
    if isinstance(obj, cq.Workplane):
        try:
            obj = obj.val()
        except Exception:
            return None
    if hasattr(obj, "toCompound"):
        try:
            obj = obj.toCompound()
        except Exception:
            pass
    if isinstance(obj, cq.Shape):
        return obj
    return None


def _extract_shape_from_namespace(ns):
    preferred_names = ["r", "result", "solid", "model", "assembly"]
    for name in preferred_names:
        if name in ns:
            shape = _cadquery_obj_to_shape(ns[name])
            if shape is not None:
                return shape
    for value in ns.values():
        shape = _cadquery_obj_to_shape(value)
        if shape is not None:
            return shape
    raise ValueError("No CadQuery solids found in script namespace.")


def _load_code_mesh(py_path: str) -> trimesh.Trimesh:
    import cadquery as cq

    py_string = Path(py_path).read_text(encoding="utf-8")
    namespace = {"__name__": "__cq_eval__", "cq": cq}
    exec(py_string, namespace)
    shape = _extract_shape_from_namespace(namespace)
    mesh = compound_to_mesh(shape)
    if len(mesh.faces) <= 2:
        raise ValueError(f"Generated mesh is too small: {py_path}")
    return mesh


def _transform_mesh_to_global(mesh: trimesh.Trimesh, scale: float, shift) -> trimesh.Trimesh:
    transformed = mesh.copy()
    transformed.vertices = (
        np.asarray(transformed.vertices, dtype=np.float64) * float(scale) +
        np.asarray(shift, dtype=np.float64)[None, :]
    )
    return transformed


def _iter_leaf_nodes(tree: dict):
    if not isinstance(tree, dict):
        return
    if tree.get("status") == "leaf":
        yield tree
    for child in (tree.get("children") or {}).values():
        yield from _iter_leaf_nodes(child)


def export_and_merge_leaf_meshes(output_dir: Path, tree: dict) -> dict:
    leaf_meshes = []
    leaf_outputs = []
    merge_errors = []

    for leaf in _iter_leaf_nodes(tree):
        code_path = leaf.get("leaf_code_path")
        if not code_path:
            continue
        try:
            local_mesh = _load_code_mesh(code_path)
            global_mesh = _transform_mesh_to_global(
                local_mesh,
                scale=float(leaf.get("to_global_scale", 1.0)),
                shift=np.asarray(leaf.get("to_global_shift", [0.0, 0.0, 0.0]), dtype=np.float32))
            mesh_path = _save_leaf_mesh(output_dir, leaf.get("node_id", "leaf"), global_mesh)
            leaf["leaf_mesh_path"] = mesh_path
            leaf_outputs.append(mesh_path)
            leaf_meshes.append(global_mesh)
        except Exception as exc:
            leaf["leaf_mesh_error"] = str(exc)
            merge_errors.append({
                "node_id": leaf.get("node_id", ""),
                "error": str(exc),
            })

    merged_mesh_path = None
    if leaf_meshes:
        merged_mesh = trimesh.util.concatenate(leaf_meshes)
        merged_mesh_path = str(output_dir / "merged_result.stl")
        merged_mesh.export(merged_mesh_path)

    return {
        "leaf_mesh_count": len(leaf_outputs),
        "leaf_mesh_paths": leaf_outputs,
        "merged_mesh_path": merged_mesh_path,
        "merge_errors": merge_errors,
    }


def _apply_yaml_defaults(args):
    config = load_config_with_defaults(args.config)
    common_cfg = get_nested(config, ["common"], {}) or {}
    common_model_cfg = get_nested(common_cfg, ["model"], {}) or {}
    common_pc_cfg = get_nested(common_cfg, ["point_cloud"], {}) or {}
    common_utonia_cfg = get_nested(common_cfg, ["utonia"], {}) or {}
    train_cfg = get_nested(config, ["train"], {}) or {}
    train_task_cfg = get_nested(train_cfg, ["task"], {}) or {}
    train_dataset_cfg = get_nested(train_cfg, ["dataset"], {}) or {}

    if args.n_points is None:
        args.n_points = int(common_pc_cfg.get("n_points", 8192))
    if args.n_point_tokens is None:
        args.n_point_tokens = int(common_pc_cfg.get("n_point_tokens", 512))
    if args.part_points_per_bbox is None:
        args.part_points_per_bbox = int(common_pc_cfg.get("part_points_per_bbox", 64))
    if args.normalize_std is None:
        args.normalize_std = float(train_dataset_cfg.get("normalize_std_pc", 100.0))
    if args.pc_encoder_type is None:
        args.pc_encoder_type = str(common_model_cfg.get("pc_encoder_type", "pointbert"))
    if args.task_mode is None:
        args.task_mode = "two_stage" if bool(train_task_cfg.get("two_stage", False)) else "multitask"
    if args.utonia_scale is None:
        args.utonia_scale = float(common_utonia_cfg.get("scale", 2.0))
    if args.utonia_normalize_coord is None:
        args.utonia_normalize_coord = bool(common_utonia_cfg.get("normalize_coord", True))
    if args.utonia_use_normal is None:
        args.utonia_use_normal = bool(common_utonia_cfg.get("use_normal", True))
    return args


def _validate_args(args):
    args.task_mode = normalize_task_mode(args.task_mode, allow_auto=True)
    args.pc_encoder_type = normalize_pc_encoder_type(args.pc_encoder_type, allow_auto=True)
    if args.n_points <= 0:
        raise ValueError("--n-points must be a positive integer.")
    if args.n_point_tokens is not None and args.n_point_tokens <= 0:
        raise ValueError("--n-point-tokens must be a positive integer.")
    if args.part_points_per_bbox is not None and args.part_points_per_bbox <= 0:
        raise ValueError("--part-points-per-bbox must be a positive integer.")
    if args.normalize_std <= 0:
        raise ValueError("--normalize-std must be positive.")
    if args.max_depth < 0:
        raise ValueError("--max-depth must be >= 0.")
    if args.crop_eps < 0:
        raise ValueError("--crop-eps must be >= 0.")
    if args.dense_points <= 0:
        raise ValueError("--dense-points must be positive.")
    if int(getattr(args, "recursive_bbox_batch_size", 1)) <= 0:
        raise ValueError("--recursive-bbox-batch-size must be positive.")
    if int(getattr(args, "recursive_code_batch_size", 1)) <= 0:
        raise ValueError("--recursive-code-batch-size must be positive.")


def main():
    parser = argparse.ArgumentParser(description="Recursive bbox inference for CADRec checkpoints")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().with_name("cadrec_config.yaml")))
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to trained checkpoint directory")
    parser.add_argument("--output-dir", type=str, default="./recursive_bbox_output")
    parser.add_argument("--npy-path", type=str, default=None, help="Normalized xyz+normal npy input")
    parser.add_argument("--ply-path", type=str, default=None, help="Normalized ply input")
    parser.add_argument("--stl-path", type=str, default=None, help="STL input; script will sample and normalize it")
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
    parser.add_argument(
        "--recursive-bbox-batch-size",
        type=int,
        default=128,
        help="Batch size for sibling bbox-node generation inside one recursive sample. Lower this if you hit OOM.",
    )
    parser.add_argument(
        "--recursive-code-batch-size",
        type=int,
        default=128,
        help="Batch size for leaf code generation inside one recursive sample. Lower this if you hit OOM.",
    )
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--crop-eps", type=float, default=0.01)
    parser.add_argument("--dense-points", type=int, default=32768)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    args = parser.parse_args()

    args = _apply_yaml_defaults(args)
    _validate_args(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    print("Loading model...")
    model, model_load_kwargs, checkpoint_hints, attempt_index = load_model_with_fallbacks(args)
    model.eval()

    runtime = resolve_runtime_settings(model, args)
    args.use_point_normals = bool(runtime["pc_encoder_type"] == "utonia" and runtime["utonia_use_normal"])

    input_path, root_dense = _build_root_dense_cloud(args, use_point_normals=args.use_point_normals)
    input_stem = Path(input_path).stem

    runtime_summary = {
        "task_mode": runtime["task_mode"],
        "pc_encoder_type": runtime["pc_encoder_type"],
        "n_point_tokens": runtime["n_point_tokens"],
        "part_points_per_bbox": runtime["part_points_per_bbox"],
        "bbox_grounding_enabled": bool(getattr(model, "bbox_grounding_enabled", False)),
        "contrastive_enabled": bool(getattr(model, "contrastive_enabled", False)),
        "utonia_use_normal": runtime["utonia_use_normal"],
        "use_point_normals": args.use_point_normals,
        "normalize_std": args.normalize_std,
        "load_attempt": attempt_index + 1,
        "processor_path": resolve_processor_path(args),
        "execution_device": str(_resolve_execution_device(model)),
    }
    print("Resolved runtime:", json.dumps(runtime_summary, ensure_ascii=False))
    if checkpoint_hints["config_found"]:
        print("Checkpoint config detected and used for auto hints when available.")
    if model_load_kwargs.get("n_point_tokens") == 512 and args.n_point_tokens is None and checkpoint_hints["n_point_tokens"] is None:
        print("Auto fallback applied: loading model with --n-point-tokens 512 because checkpoint config did not provide it.")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        resolve_processor_path(args),
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side="left",
        use_fast=True,
    )

    summary = {
        "total_nodes": 0,
        "split_nodes": 0,
        "stop_nodes": 0,
        "invalid_nodes": 0,
        "max_depth_stops": 0,
        "leaf_codes": 0,
        "bbox_visualizations": 0,
    }
    tree = infer_recursive_tree(
        model=model,
        processor=processor,
        runtime=runtime,
        dense_cloud=root_dense,
        node_id="root",
        depth=0,
        args=args,
        summary=summary)

    result = {
        "input_path": str(Path(input_path).resolve()),
        "output_dir": str(output_dir),
        "summary": summary,
        "runtime": runtime_summary,
        "tree": tree,
    }
    split_visualization = _save_aggregate_split_visualization(output_dir, root_dense.points, tree)
    result["split_visualization"] = split_visualization
    summary["bbox_visualizations"] = int(split_visualization.get("count", 0) > 0)
    mesh_export = export_and_merge_leaf_meshes(output_dir, tree)
    result["mesh_export"] = mesh_export
    summary["leaf_meshes"] = int(mesh_export["leaf_mesh_count"])
    summary["merge_errors"] = int(len(mesh_export["merge_errors"]))
    result_path = output_dir / f"{input_stem}_recursive_tree.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved recursive tree: {result_path}")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

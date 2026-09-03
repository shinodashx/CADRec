"""
Single geometry inference script for cadgen0311 checkpoints.

Supports:
- multitask checkpoints: bbox + code
- two-stage checkpoints: bbox + step_pc code
- STL mesh input sampling
- PLY point-cloud input with FPS downsampling

The script can auto-read model-side settings from the checkpoint config saved by
cadgen0311.Cadrille, and also supports explicit CLI overrides when needed.
"""

import argparse
import json
import os

import numpy as np
import torch
import trimesh
from pytorch3d.ops import sample_farthest_points
from transformers import AutoProcessor

from cadgen0311 import Cadrille, collate
from cadgendataset0311 import mesh_to_point_cloud

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TASK_DESCRIPTIONS = {
    "code": "Generate cadquery code",
    "bbox": "Generate cad parts bounding box json",
    "step_pc": "generate cadquery code by step point cloud",
}

TASK_MODE_ALIASES = {
    "auto": "auto",
    "multitask": "multitask",
    "multi_task": "multitask",
    "multi-task": "multitask",
    "twostage": "two_stage",
    "two_stage": "two_stage",
    "two-stage": "two_stage",
}

PC_ENCODER_ALIASES = {
    "auto": "auto",
    "pointbert": "pointbert",
    "utonia": "utonia",
}


def str2bool(value):
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def normalize_optional_bool(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str2bool(value)


def normalize_task_mode(value, allow_auto=True):
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_")
    normalized = TASK_MODE_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"Unsupported task mode: {value}")
    if normalized == "auto" and not allow_auto:
        raise ValueError("task mode cannot be auto here.")
    return normalized


def normalize_pc_encoder_type(value, allow_auto=True):
    if value is None:
        return None
    key = str(value).strip().lower()
    normalized = PC_ENCODER_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"Unsupported pc encoder type: {value}")
    if normalized == "auto" and not allow_auto:
        raise ValueError("pc encoder type cannot be auto here.")
    return normalized


def _load_json_if_exists(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_config(checkpoint_path):
    config_path = os.path.join(checkpoint_path, "config.json")
    config = _load_json_if_exists(config_path)
    if not isinstance(config, dict):
        return {}
    return config


def _config_get(config, *keys, default=None):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def resolve_checkpoint_hints(args):
    config = load_checkpoint_config(args.checkpoint_path)
    hints = {
        "task_mode": None,
        "pc_encoder_type": None,
        "n_point_tokens": None,
        "part_points_per_bbox": None,
        "utonia_scale": None,
        "utonia_normalize_coord": None,
        "utonia_use_normal": None,
        "contrastive_enabled": None,
        "bbox_grounding_enabled": None,
    }

    two_stage_value = _config_get(config, "two_stage", "cadgen_two_stage", default=None)
    if two_stage_value is not None:
        hints["task_mode"] = "two_stage" if normalize_optional_bool(two_stage_value) else "multitask"

    task_mode_value = _config_get(config, "task_mode", "cadgen_task_mode", default=None)
    if task_mode_value is not None:
        hints["task_mode"] = normalize_task_mode(task_mode_value, allow_auto=False)

    encoder_value = _config_get(config, "pc_encoder_type", "cadgen_pc_encoder_type", default=None)
    if encoder_value is not None:
        hints["pc_encoder_type"] = normalize_pc_encoder_type(encoder_value, allow_auto=False)

    hints["n_point_tokens"] = _config_get(config, "n_point_tokens", "cadgen_n_point_tokens", default=None)
    hints["part_points_per_bbox"] = _config_get(
        config, "part_points_per_bbox", "cadgen_part_points_per_bbox", default=None)
    hints["utonia_scale"] = _config_get(config, "cadgen_utonia_scale", "utonia_scale", default=None)
    hints["utonia_normalize_coord"] = _config_get(
        config, "cadgen_utonia_normalize_coord", "utonia_normalize_coord", default=None)
    hints["utonia_use_normal"] = _config_get(
        config, "cadgen_utonia_use_normal", "utonia_use_normal", default=None)
    contrastive_enabled = _config_get(
        config, "contrastive_enabled", "cadgen_contrastive_enabled", default=None)
    if contrastive_enabled is not None:
        hints["contrastive_enabled"] = normalize_optional_bool(contrastive_enabled)
    bbox_grounding_enabled = _config_get(
        config, "bbox_grounding_enabled", "cadgen_bbox_grounding_enabled", default=None)
    if bbox_grounding_enabled is not None:
        hints["bbox_grounding_enabled"] = normalize_optional_bool(bbox_grounding_enabled)
    hints["config_found"] = bool(config)
    return hints


def _dedupe_preserve_order(values):
    seen = set()
    ordered = []
    for value in values:
        marker = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, dict) else value
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(value)
    return ordered


def _extract_json_block(text):
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if not stripped:
        return None
    try:
        obj, _ = decoder.raw_decode(stripped)
        return obj
    except json.JSONDecodeError:
        pass
    for idx, char in enumerate(stripped):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def normalize_point_cloud_train0311(points, normalize_std=100.0):
    points = np.asarray(points, dtype=np.float32)
    return (points / float(normalize_std)).astype(np.float32)


def normalize_point_cloud_cadrecode_test(points):
    points = np.asarray(points, dtype=np.float32)
    return ((points - 0.5) * 2.0).astype(np.float32)


def normalize_point_cloud_legacy(points, target_range=100.0, normalize_std=100.0, eps=1e-8):
    points = np.asarray(points, dtype=np.float32)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    half_range = np.max(maxs - mins) / 2.0
    if half_range < eps:
        normalized = np.zeros_like(points, dtype=np.float32)
    else:
        normalized = (points - center) / half_range * float(target_range)
    return (normalized / float(normalize_std)).astype(np.float32)


def apply_point_normalization(points, mode, normalize_std):
    if mode == "train0311":
        return normalize_point_cloud_train0311(points, normalize_std=normalize_std)
    if mode == "cadrecode_test":
        return normalize_point_cloud_cadrecode_test(points)
    if mode == "legacy_center_scale":
        return normalize_point_cloud_legacy(points, normalize_std=normalize_std)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _to_vec3(value):
    if isinstance(value, dict):
        if all(axis in value for axis in ("x", "y", "z")):
            return np.array([value["x"], value["y"], value["z"]], dtype=np.float32)
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def _box_dict_to_min_max(box):
    if not isinstance(box, dict):
        return None
    if "min" in box and "max" in box:
        bbox_min = _to_vec3(box["min"])
        bbox_max = _to_vec3(box["max"])
    elif "lower" in box and "upper" in box:
        bbox_min = _to_vec3(box["lower"])
        bbox_max = _to_vec3(box["upper"])
    else:
        return None
    if bbox_min is None or bbox_max is None:
        return None
    lo = np.minimum(bbox_min, bbox_max).astype(np.float32)
    hi = np.maximum(bbox_min, bbox_max).astype(np.float32)
    return lo, hi


def extract_part_boxes(parsed_bbox, normalize_std=100.0):
    part_items = []
    if not isinstance(parsed_bbox, dict):
        return part_items

    if isinstance(parsed_bbox.get("model_box"), dict):
        parts = parsed_bbox["model_box"].get("parts")
        if isinstance(parts, dict):
            for name, box in parts.items():
                part_items.append((name, box))

    if not part_items and isinstance(parsed_bbox.get("parts"), dict):
        for name, box in parsed_bbox["parts"].items():
            part_items.append((name, box))

    if not part_items:
        for key, value in parsed_bbox.items():
            if isinstance(value, dict) and "name" in value and ("min" in value and "max" in value):
                part_items.append((str(value["name"]), value))
            elif isinstance(value, dict) and ("min" in value and "max" in value):
                part_items.append((key, value))

    part_boxes = []
    for name, box in part_items:
        min_max = _box_dict_to_min_max(box)
        if min_max is None:
            continue
        bbox_min, bbox_max = min_max
        part_boxes.append({
            "name": str(name),
            "min": bbox_min / float(normalize_std),
            "max": bbox_max / float(normalize_std),
        })
    return part_boxes


def _sample_indices(points, candidate_indices, target_count, center):
    if candidate_indices.size == 0:
        dists = np.linalg.norm(points - center[None, :], axis=1)
        nearest_idx = int(np.argmin(dists))
        return np.full((target_count,), nearest_idx, dtype=np.int64)

    if candidate_indices.size < target_count:
        repeats = (target_count + candidate_indices.size - 1) // candidate_indices.size
        tiled = np.tile(candidate_indices, repeats)[:target_count]
        return tiled.astype(np.int64)

    part_points = points[candidate_indices]
    _, local_ids = sample_farthest_points(
        torch.tensor(part_points, dtype=torch.float32).unsqueeze(0),
        K=target_count,
    )
    return candidate_indices[local_ids[0].cpu().numpy()].astype(np.int64)


def _extract_points_and_normals_from_geometry(geometry, use_point_normals=False):
    if isinstance(geometry, trimesh.Scene):
        point_chunks = []
        normal_chunks = []
        all_have_normals = True
        for item in geometry.geometry.values():
            pts, nrm = _extract_points_and_normals_from_geometry(
                item,
                use_point_normals=use_point_normals,
            )
            if pts.shape[0] == 0:
                continue
            point_chunks.append(pts)
            if use_point_normals:
                if nrm is None:
                    all_have_normals = False
                else:
                    normal_chunks.append(nrm)
        if not point_chunks:
            raise ValueError("PLY scene has no valid point vertices.")
        points = np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
        normals = None
        if use_point_normals and all_have_normals and len(normal_chunks) == len(point_chunks):
            normals = np.concatenate(normal_chunks, axis=0).astype(np.float32, copy=False)
        return points, normals

    if not hasattr(geometry, "vertices"):
        raise ValueError(f"Unsupported PLY geometry type: {type(geometry)}")
    points = np.asarray(geometry.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("PLY vertex array must have shape [N, >=3].")
    points = points[:, :3]
    normals = None
    if use_point_normals and hasattr(geometry, "vertex_normals"):
        raw_normals = np.asarray(geometry.vertex_normals, dtype=np.float32)
        if raw_normals.ndim == 2 and raw_normals.shape[0] == points.shape[0] and raw_normals.shape[1] >= 3:
            normals = raw_normals[:, :3]
    return points, normals


def fps_downsample_points(points, target_count, normals=None):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Points must have shape [N, >=3].")
    points = points[:, :3]
    if target_count <= 0:
        raise ValueError("target_count must be a positive integer.")

    n_points = int(points.shape[0])
    if n_points <= 0:
        raise ValueError("Cannot downsample empty point cloud.")

    if n_points < target_count:
        repeats = (target_count + n_points - 1) // n_points
        sample_indices = np.tile(np.arange(n_points, dtype=np.int64), repeats)[:target_count]
    else:
        _, sample_ids = sample_farthest_points(
            torch.tensor(points, dtype=torch.float32).unsqueeze(0),
            K=target_count,
        )
        sample_indices = sample_ids[0].cpu().numpy().astype(np.int64)

    sampled_points = points[sample_indices]
    sampled_normals = None
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.ndim != 2 or normals.shape[0] != n_points or normals.shape[1] < 3:
            raise ValueError("Normals must have shape [N, >=3] and match points.")
        sampled_normals = normals[sample_indices, :3]
    return sampled_points.astype(np.float32), sampled_normals


def load_ply_point_cloud(ply_path, target_count, use_point_normals=False):
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    geometry = trimesh.load(ply_path, process=False)
    points, normals = _extract_points_and_normals_from_geometry(
        geometry,
        use_point_normals=use_point_normals,
    )
    sampled_points, sampled_normals = fps_downsample_points(
        points,
        target_count=target_count,
        normals=normals if use_point_normals else None,
    )
    return sampled_points, sampled_normals


def normalize_to_bbox_range(points, target_range=100.0, eps=1e-8):
    points = np.asarray(points, dtype=np.float32)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    half_range = np.max(maxs - mins) / 2.0
    if half_range < eps:
        return np.zeros_like(points, dtype=np.float32)
    return ((points - center) / half_range * float(target_range)).astype(np.float32)


def build_step_part_indices(point_cloud, parsed_bbox, n_points_per_part, normalize_std=100.0, eps=0.01):
    part_boxes = extract_part_boxes(parsed_bbox, normalize_std=normalize_std)
    if not part_boxes:
        return np.zeros((0, n_points_per_part), dtype=np.int64)

    points = point_cloud.astype(np.float32)
    part_indices = []
    for part in part_boxes:
        bbox_min = part["min"]
        bbox_max = part["max"]
        mask = np.all(
            (points >= bbox_min - eps) & (points <= bbox_max + eps),
            axis=1,
        )
        candidate_indices = np.nonzero(mask)[0]
        center = (bbox_min + bbox_max) / 2.0
        sampled = _sample_indices(points, candidate_indices, n_points_per_part, center)
        part_indices.append(sampled)

    if not part_indices:
        return np.zeros((0, n_points_per_part), dtype=np.int64)
    return np.stack(part_indices, axis=0).astype(np.int64)


def build_model_load_kwargs(args):
    hints = resolve_checkpoint_hints(args)
    model_kwargs = {}

    task_mode = normalize_task_mode(args.task_mode, allow_auto=True)
    if task_mode == "auto":
        task_mode = hints["task_mode"]
    if task_mode is not None:
        model_kwargs["two_stage"] = (task_mode == "two_stage")

    pc_encoder_type = normalize_pc_encoder_type(args.pc_encoder_type, allow_auto=True)
    if pc_encoder_type == "auto":
        pc_encoder_type = hints["pc_encoder_type"]
    if pc_encoder_type is not None:
        model_kwargs["pc_encoder_type"] = pc_encoder_type

    n_point_tokens = args.n_point_tokens if args.n_point_tokens is not None else hints["n_point_tokens"]
    if n_point_tokens is not None:
        model_kwargs["n_point_tokens"] = int(n_point_tokens)

    part_points_per_bbox = (
        args.part_points_per_bbox
        if args.part_points_per_bbox is not None
        else hints["part_points_per_bbox"]
    )
    if part_points_per_bbox is not None:
        model_kwargs["part_points_per_bbox"] = int(part_points_per_bbox)

    utonia_scale = args.utonia_scale if args.utonia_scale is not None else hints["utonia_scale"]
    if utonia_scale is not None:
        model_kwargs["utonia_scale"] = float(utonia_scale)

    utonia_normalize_coord = (
        args.utonia_normalize_coord
        if args.utonia_normalize_coord is not None
        else hints["utonia_normalize_coord"]
    )
    if utonia_normalize_coord is not None:
        model_kwargs["utonia_normalize_coord"] = normalize_optional_bool(utonia_normalize_coord)

    utonia_use_normal = (
        args.utonia_use_normal
        if args.utonia_use_normal is not None
        else hints["utonia_use_normal"]
    )
    if utonia_use_normal is not None:
        model_kwargs["utonia_use_normal"] = normalize_optional_bool(utonia_use_normal)

    contrastive_enabled = getattr(args, "contrastive_enabled", None)
    if contrastive_enabled is None:
        contrastive_enabled = hints["contrastive_enabled"]
    if contrastive_enabled is not None:
        model_kwargs["contrastive_enabled"] = normalize_optional_bool(contrastive_enabled)

    bbox_grounding_enabled = getattr(args, "bbox_grounding_enabled", None)
    if bbox_grounding_enabled is None:
        bbox_grounding_enabled = hints["bbox_grounding_enabled"]
    if bbox_grounding_enabled is not None:
        model_kwargs["bbox_grounding_enabled"] = normalize_optional_bool(bbox_grounding_enabled)

    return model_kwargs, hints


def load_model_with_fallbacks(args):
    base_model_kwargs, hints = build_model_load_kwargs(args)
    attempts = [dict(base_model_kwargs)]

    if (
        "pc_encoder_type" not in base_model_kwargs
        and args.pc_encoder_type == "auto"
        and hints["pc_encoder_type"] is None
    ):
        utonia_retry = dict(base_model_kwargs)
        utonia_retry["pc_encoder_type"] = "utonia"
        if "utonia_scale" not in utonia_retry and args.utonia_scale is None and hints["utonia_scale"] is None:
            utonia_retry["utonia_scale"] = 2.0
        if (
            "utonia_normalize_coord" not in utonia_retry
            and args.utonia_normalize_coord is None
            and hints["utonia_normalize_coord"] is None
        ):
            utonia_retry["utonia_normalize_coord"] = True
        if (
            "utonia_use_normal" not in utonia_retry
            and args.utonia_use_normal is None
            and hints["utonia_use_normal"] is None
        ):
            utonia_retry["utonia_use_normal"] = True
        attempts.append(utonia_retry)

    expanded_attempts = []
    for attempt in attempts:
        expanded_attempts.append(dict(attempt))
        if (
            "n_point_tokens" not in attempt
            and args.n_point_tokens is None
            and hints["n_point_tokens"] is None
        ):
            retry_kwargs = dict(attempt)
            retry_kwargs["n_point_tokens"] = 512
            expanded_attempts.append(retry_kwargs)

    attempts = _dedupe_preserve_order(expanded_attempts)
    last_exc = None
    for idx, model_kwargs in enumerate(attempts):
        try:
            model = Cadrille.from_pretrained(
                args.checkpoint_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation,
                device_map=args.device_map,
                **model_kwargs,
            )
            return model, model_kwargs, hints, idx
        except Exception as exc:
            last_exc = exc

    hint_lines = [
        "Failed to load checkpoint with current example1.py settings.",
        f"checkpoint: {args.checkpoint_path}",
    ]
    if args.task_mode == "auto" and hints["task_mode"] is None:
        hint_lines.append("task mode was not stored in checkpoint config, so pass --task-mode multitask or --task-mode two_stage explicitly.")
    if args.pc_encoder_type == "auto" and hints["pc_encoder_type"] is None:
        hint_lines.append("pc encoder type was not stored in checkpoint config, so pass --pc-encoder-type pointbert or --pc-encoder-type utonia explicitly if needed.")
    if args.n_point_tokens is None and hints["n_point_tokens"] is None:
        hint_lines.append("n-point-tokens was not stored in checkpoint config. Current traincadgen0311.py often uses --n-point-tokens 512; pass it explicitly if your checkpoint was trained that way.")
    if args.part_points_per_bbox is None and hints["part_points_per_bbox"] is None:
        hint_lines.append("If this is a two-stage checkpoint, also confirm --part-points-per-bbox matches training, usually 64.")
    raise RuntimeError(" ".join(hint_lines)) from last_exc


def resolve_processor_path(args):
    if args.processor_path != "auto":
        return args.processor_path

    checkpoint_processor_files = (
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    if any(os.path.isfile(os.path.join(args.checkpoint_path, name)) for name in checkpoint_processor_files):
        return args.checkpoint_path
    return "Qwen/Qwen2-VL-2B-Instruct"


def resolve_runtime_settings(model, args):
    point_encoder = getattr(model, "point_encoder", None)
    pc_encoder_type = str(getattr(model, "pc_encoder_type", "pointbert")).lower()
    task_mode = args.task_mode
    if task_mode == "auto":
        task_mode = "two_stage" if bool(getattr(model, "two_stage", False)) else "multitask"
    task_mode = normalize_task_mode(task_mode, allow_auto=False)

    normalization_mode = args.normalization_mode
    if normalization_mode == "auto":
        normalization_mode = "train0311"

    return {
        "task_mode": task_mode,
        "pc_encoder_type": pc_encoder_type,
        "n_point_tokens": int(getattr(model, "n_point_tokens", 256)),
        "part_points_per_bbox": int(getattr(model, "part_points_per_bbox", 64)),
        "utonia_scale": float(getattr(point_encoder, "scale", getattr(model.config, "cadgen_utonia_scale", 1.0))),
        "utonia_normalize_coord": bool(getattr(
            point_encoder,
            "normalize_coord",
            getattr(model.config, "cadgen_utonia_normalize_coord", True))),
        "utonia_use_normal": bool(getattr(
            point_encoder,
            "use_normal",
            getattr(model.config, "cadgen_utonia_use_normal", False))),
        "normalization_mode": normalization_mode,
    }


def generate_text(model, processor, batch, runtime, n_points, max_new_tokens):
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

    generate_kwargs = {
        "input_ids": model_inputs["input_ids"].to(model.device),
        "attention_mask": model_inputs["attention_mask"].to(model.device),
        "point_clouds": model_inputs["point_clouds"].to(model.device),
        "is_pc": model_inputs["is_pc"].to(model.device),
        "is_img": model_inputs["is_img"].to(model.device),
        "max_new_tokens": max_new_tokens,
    }
    if model_inputs.get("part_point_indices") is not None:
        generate_kwargs["part_point_indices"] = model_inputs["part_point_indices"].to(model.device)
    if model_inputs.get("n_parts") is not None:
        generate_kwargs["n_parts"] = model_inputs["n_parts"].to(model.device)
    if model_inputs.get("is_step_pc") is not None:
        generate_kwargs["is_step_pc"] = model_inputs["is_step_pc"].to(model.device)
    if model_inputs.get("pixel_values_videos") is not None:
        generate_kwargs["pixel_values_videos"] = model_inputs["pixel_values_videos"].to(model.device)
    if model_inputs.get("video_grid_thw") is not None:
        generate_kwargs["video_grid_thw"] = model_inputs["video_grid_thw"].to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    outputs = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return outputs[0]


def main():
    parser = argparse.ArgumentParser(description="Single geometry inference for cadgen0311 checkpoints")
    parser.add_argument("--stl-path", type=str, default=None, help="Path to input STL file")
    parser.add_argument("--ply", action="store_true", help="Use PLY point-cloud input mode.")
    parser.add_argument("--ply-path", type=str, default=None, help="Path to input PLY file when --ply is set.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to trained checkpoint directory")
    parser.add_argument("--output-dir", type=str, default="./exampleply", help="Directory to save output files")
    parser.add_argument(
        "--processor-path",
        type=str,
        default="auto",
        help="auto prefers processor files saved inside the checkpoint, otherwise falls back to Qwen/Qwen2-VL-2B-Instruct.",
    )
    parser.add_argument("--n-points", type=int, default=8192, help="Point cloud size")
    parser.add_argument("--normalize-std", type=float, default=100.0, help="Normalization divisor")
    parser.add_argument(
        "--normalization-mode",
        type=str,
        choices=["auto", "train0311", "cadrecode_test", "legacy_center_scale"],
        default="auto",
        help="Point normalization mode. auto defaults to train0311, which matches traincadgen0311.py.",
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        default="auto",
        help="Checkpoint task mode: auto, multitask, multi-task, two_stage, two-stage, or twostage.",
    )
    parser.add_argument(
        "--pc-encoder-type",
        type=str,
        default="pointbert",
        help="Override point encoder type: auto, pointbert, or utonia.",
    )
    parser.add_argument(
        "--n-point-tokens",
        type=int,
        default=512,
        help="Override number of global point tokens. Pass this explicitly if checkpoint config did not save it.",
    )
    parser.add_argument(
        "--part-points-per-bbox",
        type=int,
        default=64,
        help="Override points sampled per bbox part for two-stage step_pc inference.",
    )
    parser.add_argument(
        "--utonia-scale",
        type=float,
        default=2.0,
        help="Optional Utonia scale override. Current traincadgen0311.py default is 2.0.",
    )
    parser.add_argument(
        "--utonia-normalize-coord",
        type=str2bool,
        default=True,
        help="Optional Utonia normalize_coord override",
    )
    parser.add_argument(
        "--utonia-use-normal",
        type=str2bool,
        default=True,
        help="Optional Utonia use_normal override",
    )
    parser.add_argument(
        "--utonia-use-fourier-fusion",
        type=str2bool,
        default=True,
        help="Kept for CLI compatibility only. example1.py ignores this because cadgen0311.Cadrille does not accept it.",
    )
    parser.add_argument("--max-new-tokens-bbox", type=int, default=3768)
    parser.add_argument("--max-new-tokens-code", type=int, default=3768)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    args = parser.parse_args()

    if args.ply:
        if not args.ply_path:
            raise ValueError("--ply-path is required when --ply is set.")
        if not os.path.isfile(args.ply_path):
            raise FileNotFoundError(f"PLY file not found: {args.ply_path}")
    else:
        if not args.stl_path:
            raise ValueError("--stl-path is required when --ply is not set.")
        if not os.path.isfile(args.stl_path):
            raise FileNotFoundError(f"STL file not found: {args.stl_path}")
    if args.n_points <= 0:
        raise ValueError("--n-points must be a positive integer.")
    if args.normalize_std <= 0:
        raise ValueError("--normalize-std must be a positive number.")
    if args.n_point_tokens is not None and args.n_point_tokens <= 0:
        raise ValueError("--n-point-tokens must be a positive integer.")
    if args.part_points_per_bbox is not None and args.part_points_per_bbox <= 0:
        raise ValueError("--part-points-per-bbox must be a positive integer.")
    args.task_mode = normalize_task_mode(args.task_mode, allow_auto=True)
    args.pc_encoder_type = normalize_pc_encoder_type(args.pc_encoder_type, allow_auto=True)

    os.makedirs(args.output_dir, exist_ok=True)
    input_path = args.ply_path if args.ply else args.stl_path
    stem = os.path.splitext(os.path.basename(input_path))[0]

    print("Loading model...")
    model, model_load_kwargs, checkpoint_hints, attempt_index = load_model_with_fallbacks(args)
    model.eval()

    runtime = resolve_runtime_settings(model, args)
    print(
        "Resolved runtime:",
        json.dumps(
            {
                "task_mode": runtime["task_mode"],
                "pc_encoder_type": runtime["pc_encoder_type"],
                "n_point_tokens": runtime["n_point_tokens"],
                "part_points_per_bbox": runtime["part_points_per_bbox"],
                "utonia_use_normal": runtime["utonia_use_normal"],
                "normalization_mode": runtime["normalization_mode"],
                "processor_path": resolve_processor_path(args),
                "load_attempt": attempt_index + 1,
            },
            ensure_ascii=False,
        ),
    )
    if checkpoint_hints["config_found"]:
        print("Checkpoint config detected and used for auto hints when available.")
    if args.utonia_use_fourier_fusion is not None:
        print("Note: --utonia-use-fourier-fusion is ignored because cadgen0311.Cadrille has no such init argument.")
    if model_load_kwargs.get("n_point_tokens") == 512 and args.n_point_tokens is None and checkpoint_hints["n_point_tokens"] is None:
        print("Auto fallback applied: loading model with --n-point-tokens 512 because checkpoint config did not provide it.")

    print("Loading processor...")
    processor_path = resolve_processor_path(args)
    processor = AutoProcessor.from_pretrained(
        processor_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side="left",
        use_fast=True,
    )

    use_point_normals = bool(runtime["pc_encoder_type"] == "utonia" and runtime["utonia_use_normal"])
    if args.ply:
        print(f"Loading PLY point cloud and FPS downsampling to --n-points={args.n_points}...")
        point_cloud, point_normal = load_ply_point_cloud(
            ply_path=args.ply_path,
            target_count=args.n_points,
            use_point_normals=use_point_normals,
        )
    else:
        print("Sampling point cloud from STL...")
        mesh = trimesh.load(args.stl_path)
        sampled = mesh_to_point_cloud(
            mesh,
            n_points=args.n_points,
            return_normals=use_point_normals,
        )
        if use_point_normals:
            point_cloud, point_normal = sampled
        else:
            point_cloud = sampled
            point_normal = None

    if args.normalization_mode not in ("auto", "legacy_center_scale"):
        print(
            "Warning: to keep STL/PLY preprocessing consistent, this script uses legacy_center_scale "
            "(normalize to [-100,100] then divide by --normalize-std) for both inputs."
        )
    point_cloud = normalize_point_cloud_legacy(
        point_cloud,
        target_range=100.0,
        normalize_std=args.normalize_std,
    )
    effective_n_points = int(point_cloud.shape[0])
    print(f"Point cloud shape: {point_cloud.shape}")

    base_item = {
        "point_cloud": point_cloud,
        "file_name": stem,
    }
    if point_normal is not None:
        base_item["point_normal"] = point_normal.astype(np.float32)

    print("Generating bbox output...")
    bbox_batch = [{
        **base_item,
        "description": TASK_DESCRIPTIONS["bbox"],
        "task_type": "bbox",
    }]
    bbox_text = generate_text(
        model=model,
        processor=processor,
        batch=bbox_batch,
        runtime=runtime,
        n_points=effective_n_points,
        max_new_tokens=args.max_new_tokens_bbox,
    )
    bbox_text_path = os.path.join(args.output_dir, f"{stem}_bbox.txt")
    with open(bbox_text_path, "w", encoding="utf-8") as f:
        f.write(bbox_text.strip())

    bbox_json_path = os.path.join(args.output_dir, f"{stem}_bbox.json")
    parsed_bbox = _extract_json_block(bbox_text)
    with open(bbox_json_path, "w", encoding="utf-8") as f:
        if parsed_bbox is not None:
            json.dump(parsed_bbox, f, ensure_ascii=False, indent=2)
        else:
            f.write(bbox_text.strip())

    print("Generating code output...")
    if runtime["task_mode"] == "two_stage":
        if parsed_bbox is None:
            print("Warning: bbox output could not be parsed as JSON. Two-stage step_pc inference will continue with zero or fallback part indices.")
        part_point_indices = build_step_part_indices(
            point_cloud=point_cloud,
            parsed_bbox=parsed_bbox,
            n_points_per_part=runtime["part_points_per_bbox"],
            normalize_std=args.normalize_std,
            eps=0.01,
        )
        code_batch = [{
            **base_item,
            "description": TASK_DESCRIPTIONS["step_pc"],
            "task_type": "step_pc",
            "part_point_indices": part_point_indices,
            "n_parts": int(part_point_indices.shape[0]),
        }]
    else:
        code_batch = [{
            **base_item,
            "description": TASK_DESCRIPTIONS["code"],
        }]

    code_text = generate_text(
        model=model,
        processor=processor,
        batch=code_batch,
        runtime=runtime,
        n_points=effective_n_points,
        max_new_tokens=args.max_new_tokens_code,
    )
    code_output_path = os.path.join(args.output_dir, f"{stem}_generated.py")
    with open(code_output_path, "w", encoding="utf-8") as f:
        f.write(code_text)

    print(f"Saved bbox text: {bbox_text_path}")
    print(f"Saved bbox json: {bbox_json_path}")
    print(f"Saved code file: {code_output_path}")


if __name__ == "__main__":
    main()

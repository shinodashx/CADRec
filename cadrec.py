import sys
import os
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch3d.ops import sample_farthest_points
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

# Make sure project local packages (utils, models, tools, ...) shadow site-packages entries.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


BBOX_GROUNDING_SPECIAL_TOKENS = (
    '<SPLIT>', '<STOP>', '<BOX>', '<Bs>', '<Be>', '<END_BBOX>')

_UTONIA_TRANSFORM_CACHE = {}
_TASK_TYPE_TO_ID = {
    'code': 0,
    'bbox': 1,
    'step_pc': 2,
}


def _stable_hash_to_int64(text):
    text = str(text)
    digest = hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, byteorder='big', signed=True)


def _infer_source_key_for_collate(item):
    source_key = item.get('source_key', None)
    if isinstance(source_key, str) and source_key:
        return source_key

    file_name = item.get('file_name', '')
    if isinstance(file_name, str) and file_name:
        source_key = file_name
        for suffix in ('_code', '_bbox', '_step_pc', '_step', '_full', '_crop'):
            if source_key.endswith(suffix):
                source_key = source_key[:-len(suffix)]
        if source_key:
            return source_key
    return ''


def _task_type_to_id(task_type):
    return int(_TASK_TYPE_TO_ID.get(str(task_type or 'code').lower(), 0))


def _bbox_value_to_array(value):
    if isinstance(value, dict):
        if 'bbox' in value:
            return _bbox_value_to_array(value['bbox'])
        if 'box' in value:
            return _bbox_value_to_array(value['box'])
        if 'bounds' in value:
            return _bbox_value_to_array(value['bounds'])
        if 'min' in value and 'max' in value:
            return _bbox_value_to_array([value['min'], value['max']])
        if 'lower' in value and 'upper' in value:
            return _bbox_value_to_array([value['lower'], value['upper']])
        if 'center' in value and 'size' in value:
            center = _bbox_value_to_xyz(value['center'])
            size = _bbox_value_to_xyz(value['size'])
            if center is None or size is None:
                return None
            half = size / 2.0
            return np.concatenate([center - half, center + half], axis=0)
        keys = ('xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax')
        if all(k in value for k in keys):
            return np.array([value[k] for k in keys], dtype=np.float32)
    if isinstance(value, (list, tuple)):
        if len(value) == 6 and not isinstance(value[0], (dict, list, tuple)):
            return np.asarray(value, dtype=np.float32)
        if len(value) == 2:
            lower = _bbox_value_to_xyz(value[0])
            upper = _bbox_value_to_xyz(value[1])
            if lower is not None and upper is not None:
                return np.concatenate([lower, upper], axis=0)
    return None


def _bbox_value_to_xyz(value):
    if isinstance(value, dict):
        if all(k in value for k in ('x', 'y', 'z')):
            return np.array([value['x'], value['y'], value['z']], dtype=np.float32)
        if all(k in value for k in ('0', '1', '2')):
            return np.array([value['0'], value['1'], value['2']], dtype=np.float32)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return np.asarray(value[:3], dtype=np.float32)
    return None


def _canonicalize_bbox_array_raw(box):
    box = np.asarray(box, dtype=np.float32).reshape(-1)
    if box.shape[0] != 6:
        return None
    lower = np.minimum(box[:3], box[3:])
    upper = np.maximum(box[:3], box[3:])
    return np.concatenate([lower, upper], axis=0).astype(np.float32, copy=False)


def _canonicalize_bbox_array(box, normalize_std=100.0):
    out = _canonicalize_bbox_array_raw(box)
    if out is None:
        return None
    scale = float(normalize_std) if normalize_std is not None else 1.0
    if scale > 0:
        out = out / scale
    return out


def _iter_bbox_container_values(container):
    if isinstance(container, dict):
        return list(container.values())
    if isinstance(container, (list, tuple)):
        return list(container)
    return []


def _parse_bbox_grounding_answer(answer, normalize_std=100.0, max_boxes=0):
    if not isinstance(answer, str) or not answer.strip():
        return None
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    label = str(data.get('label', '')).strip().upper()
    container = None
    if label == 'SPLIT':
        container = data.get('parts', None)
    elif label == 'STOP':
        container = data.get('steps', None)

    if container is None and 'parts' in data:
        label = 'SPLIT'
        container = data.get('parts', None)
    if container is None and 'steps' in data:
        label = 'STOP'
        container = data.get('steps', None)
    if container is None and isinstance(data.get('model_box', None), dict):
        model_box = data['model_box']
        if 'parts' in model_box:
            label = 'SPLIT'
            container = model_box.get('parts', None)

    if label not in ('SPLIT', 'STOP') or container is None:
        return None

    boxes = []
    raw_boxes = []
    for value in _iter_bbox_container_values(container):
        box = _bbox_value_to_array(value)
        raw_box = _canonicalize_bbox_array_raw(box) if box is not None else None
        norm_box = _canonicalize_bbox_array(
            box,
            normalize_std=normalize_std) if box is not None else None
        if raw_box is not None and norm_box is not None:
            raw_boxes.append(raw_box)
            boxes.append(norm_box)

    if max_boxes is not None and int(max_boxes) > 0:
        boxes = boxes[:int(max_boxes)]
        raw_boxes = raw_boxes[:int(max_boxes)]
    if len(boxes) == 0:
        return None

    return {
        'label': label,
        'boxes': np.stack(boxes, axis=0).astype(np.float32, copy=False),
        'raw_boxes': np.stack(raw_boxes, axis=0).astype(np.float32, copy=False),
    }


def _format_bbox_number(value):
    value = float(value)
    if abs(value) < 5e-7:
        value = 0.0
    text = f'{value:.6f}'.rstrip('0').rstrip('.')
    return text if text else '0'


def _format_bbox_span(box):
    numbers = ','.join(_format_bbox_number(v) for v in box)
    return f'<Bs> {numbers} <Be>'


def _format_bbox_grounding_answer(record):
    label_token = '<SPLIT>' if record['label'] == 'SPLIT' else '<STOP>'
    raw_boxes = record.get('raw_boxes', record['boxes'])
    box_spans = ' '.join(_format_bbox_span(box) for box in raw_boxes)
    if box_spans:
        return f'{label_token} {box_spans} <END_BBOX>'
    return f'{label_token} <END_BBOX>'


def _prepare_bbox_grounding_batch(batch, enabled=False, normalize_std=100.0, max_boxes=0):
    if not enabled:
        return batch, None
    out_batch = []
    records = []
    for item in batch:
        item_out = dict(item)
        record = None
        if item_out.get('task_type') == 'bbox':
            record = _parse_bbox_grounding_answer(
                item_out.get('answer', ''),
                normalize_std=normalize_std,
                max_boxes=max_boxes)
            if record is not None:
                item_out['answer'] = _format_bbox_grounding_answer(record)
        out_batch.append(item_out)
        records.append(record)
    return out_batch, records


def _attach_bbox_grounding_targets(inputs, records):
    if records is None:
        return
    max_boxes = max(
        (int(record['boxes'].shape[0]) for record in records if record is not None),
        default=0)
    if max_boxes <= 0:
        return

    batch_size = len(records)
    boxes = torch.zeros(batch_size, max_boxes, 6, dtype=torch.float32)
    box_mask = torch.zeros(batch_size, max_boxes, dtype=torch.bool)
    sample_mask = torch.zeros(batch_size, dtype=torch.bool)
    for idx, record in enumerate(records):
        if record is None:
            continue
        n_box = int(record['boxes'].shape[0])
        boxes[idx, :n_box] = torch.as_tensor(record['boxes'], dtype=torch.float32)
        box_mask[idx, :n_box] = True
        sample_mask[idx] = True

    inputs['bbox_grounding_boxes'] = boxes
    inputs['bbox_grounding_box_mask'] = box_mask
    inputs['bbox_grounding_sample_mask'] = sample_mask


def _import_utonia_for_collate():
    try:
        import utonia  # noqa: F401
        return utonia
    except ImportError:
        local_root = PROJECT_ROOT / 'Utonia-main'
        local_root_str = str(local_root)
        if local_root.exists() and local_root_str not in sys.path:
            sys.path.insert(0, local_root_str)
        import utonia  # noqa: F401
        return utonia


def _get_utonia_transform_for_collate(scale, normalize_coord):
    key = (float(scale), bool(normalize_coord))
    if key not in _UTONIA_TRANSFORM_CACHE:
        utonia = _import_utonia_for_collate()
        _UTONIA_TRANSFORM_CACHE[key] = utonia.transform.default(
            scale=float(scale),
            apply_z_positive=True,
            normalize_coord=bool(normalize_coord))
    return _UTONIA_TRANSFORM_CACHE[key]


def _build_utonia_data_dict_from_tensors(point_cloud_tensors, scale, normalize_coord, use_normal):
    transform = _get_utonia_transform_for_collate(scale=scale, normalize_coord=normalize_coord)
    samples = []
    for point_tensor in point_cloud_tensors:
        point_np = point_tensor.numpy().astype(np.float32, copy=False)
        coord_np = point_np[:, :3]
        zeros = np.zeros_like(coord_np, dtype=np.float32)
        if bool(use_normal) and point_np.shape[1] >= 6:
            normal_np = point_np[:, 3:6]
        else:
            normal_np = zeros
        sample = transform({
            'coord': coord_np,
            'color': zeros,
            'normal': normal_np,
        })
        samples.append(sample)

    utonia = _import_utonia_for_collate()
    return utonia.data.collate_fn(samples)


def _infer_point_feature_dim(batch, default_dim=3):
    dims = []
    for item in batch:
        if 'point_cloud' not in item:
            continue
        point_cloud = np.asarray(item['point_cloud'])
        if point_cloud.ndim == 2 and point_cloud.shape[1] > 0:
            dim = int(point_cloud.shape[1])
        else:
            dim = int(default_dim)
        if 'point_normal' in item:
            point_normal = np.asarray(item['point_normal'])
            if point_normal.ndim == 2 and point_normal.shape[1] > 0:
                dim += int(point_normal.shape[1])
            else:
                dim += 3
        dims.append(dim)
    return max(dims, default=default_dim)


def _compose_point_cloud_tensor(item, n_points, feature_dim):
    if 'point_cloud' not in item:
        return torch.zeros(n_points, feature_dim, dtype=torch.float32)

    point_cloud = torch.as_tensor(item['point_cloud'], dtype=torch.float32)
    features = [point_cloud]
    if 'point_normal' in item:
        point_normal = torch.as_tensor(item['point_normal'], dtype=torch.float32)
        features.append(point_normal)
    point_tensor = torch.cat(features, dim=-1)

    if point_tensor.shape[1] < feature_dim:
        pad = torch.zeros(
            point_tensor.shape[0],
            feature_dim - point_tensor.shape[1],
            dtype=point_tensor.dtype)
        point_tensor = torch.cat([point_tensor, pad], dim=-1)
    elif point_tensor.shape[1] > feature_dim:
        point_tensor = point_tensor[:, :feature_dim]
    return point_tensor


def _collate_two_stage_single_task(
    batch,
    processor,
    n_point_tokens,
    n_type_tokens=1,
    n_part_point_tokens=64,
    eval=False,
    pc_encoder_type='pointbert',
    utonia_scale=1.0,
    utonia_normalize_coord=True,
    utonia_use_normal=False,
    build_utonia_data_dict=True,
    bbox_grounding_enabled=False,
    bbox_grounding_normalize_std=100.0,
    bbox_grounding_max_boxes=0):
    batch, bbox_grounding_records = _prepare_bbox_grounding_batch(
        batch,
        enabled=bbox_grounding_enabled,
        normalize_std=bbox_grounding_normalize_std,
        max_boxes=bbox_grounding_max_boxes)
    messages = []
    for m in batch:
        if eval:
            message = [{
                'role': 'user',
                'content': [{'type': 'text', 'text': m['description']}]
            }]
        else:
            message = [{
                'role': 'user',
                'content': [{'type': 'text', 'text': m['description']}]
            }, {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': m['answer']}]
            }]
        messages.append(message)

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=eval)
        for msg in messages
    ]

    for i, m in enumerate(batch):
        if m.get('task_type') == 'step_pc':
            n_parts = int(m.get('n_parts', 0))
            n_part_tokens = n_parts * n_part_point_tokens + max(n_parts - 1, 0)
            n_tokens = n_type_tokens + n_point_tokens + 2 + n_part_tokens
        else:
            n_tokens = n_type_tokens + n_point_tokens
        prefix = ''.join(n_tokens * [processor.tokenizer.pad_token])
        texts[i] = prefix + texts[i]

    inputs = processor(
        text=texts,
        padding=True,
        return_tensors='pt')

    point_feature_dim = _infer_point_feature_dim(batch)
    point_cloud_tensors = [
        _compose_point_cloud_tensor(m, n_points=0, feature_dim=point_feature_dim)
        for m in batch
    ]
    inputs['point_clouds'] = torch.stack(point_cloud_tensors, dim=0)
    if build_utonia_data_dict and str(pc_encoder_type).lower() == 'utonia':
        inputs['utonia_data_dict'] = _build_utonia_data_dict_from_tensors(
            point_cloud_tensors=point_cloud_tensors,
            scale=utonia_scale,
            normalize_coord=utonia_normalize_coord,
            use_normal=utonia_use_normal)
    inputs['is_pc'] = torch.ones(len(batch), dtype=torch.bool)
    inputs['is_img'] = torch.zeros(len(batch), dtype=torch.bool)
    inputs['pc_type_ids'] = torch.tensor(
        [int(m.get('pc_type_id', 0)) for m in batch],
        dtype=torch.long)
    inputs['source_ids'] = torch.tensor(
        [
            _stable_hash_to_int64(source_key) if source_key else -1
            for source_key in (_infer_source_key_for_collate(m) for m in batch)
        ],
        dtype=torch.long)
    inputs['task_ids'] = torch.tensor(
        [_task_type_to_id(m.get('task_type', 'code')) for m in batch],
        dtype=torch.long)
    inputs['is_step_pc'] = torch.tensor(
        [m.get('task_type') == 'step_pc' for m in batch], dtype=torch.bool)

    n_parts_list = [int(m.get('n_parts', 0)) for m in batch]
    max_parts = max(n_parts_list) if len(n_parts_list) > 0 else 0
    part_indices = torch.zeros(
        (len(batch), max_parts, n_part_point_tokens), dtype=torch.long)
    for i, m in enumerate(batch):
        if m.get('task_type') != 'step_pc':
            continue
        n_parts = n_parts_list[i]
        if n_parts <= 0:
            continue
        indices = torch.as_tensor(m['part_point_indices'], dtype=torch.long)
        n_points_this_part = min(indices.size(1), n_part_point_tokens)
        part_indices[i, :n_parts, :n_points_this_part] = \
            indices[:n_parts, :n_points_this_part]
    inputs['part_point_indices'] = part_indices
    inputs['n_parts'] = torch.tensor(n_parts_list, dtype=torch.long)
    _attach_bbox_grounding_targets(inputs, bbox_grounding_records)

    if not eval:
        input_ids_lists = inputs['input_ids'].tolist()
        labels_list = []
        for ids_list in input_ids_lists:
            label_ids = [-100] * len(ids_list)
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0] + 2: begin_end_indexs[1] + 1] = \
                    ids_list[begin_end_indexs[0] + 2: begin_end_indexs[1] + 1]
            labels_list.append(label_ids)
        inputs['labels'] = torch.tensor(labels_list, dtype=torch.int64)
    else:
        inputs['file_name'] = [m['file_name'] for m in batch]

    return inputs


def _collate_two_stage(
    batch,
    processor,
    n_point_tokens,
    n_type_tokens=1,
    n_part_point_tokens=64,
    eval=False,
    pc_encoder_type='pointbert',
    utonia_scale=1.0,
    utonia_normalize_coord=True,
    utonia_use_normal=False,
    build_utonia_data_dict=True,
    bbox_grounding_enabled=False,
    bbox_grounding_normalize_std=100.0,
    bbox_grounding_max_boxes=0):
    bbox_batch = []
    step_batch = []
    for m in batch:
        bbox_item = {
            'description': m['bbox_description'],
            'answer': m.get('bbox_answer', ''),
            'point_cloud': m['point_cloud'],
            'task_type': 'bbox',
            'n_parts': 0,
            'part_point_indices': [],
            'pc_type_id': int(m.get('pc_type_id', 0)),
            'file_name': m.get('file_name', ''),
            'source_key': m.get('source_key', m.get('file_name', '')),
        }
        if 'point_normal' in m:
            bbox_item['point_normal'] = m['point_normal']
        bbox_batch.append(bbox_item)

        step_item = {
            'description': m['step_description'],
            'answer': m.get('step_answer', ''),
            'point_cloud': m['point_cloud'],
            'task_type': 'step_pc',
            'n_parts': int(m.get('n_parts', 0)),
            'part_point_indices': m.get('part_point_indices', []),
            'pc_type_id': int(m.get('pc_type_id', 0)),
            'file_name': m.get('file_name', ''),
            'source_key': m.get('source_key', m.get('file_name', '')),
        }
        if 'point_normal' in m:
            step_item['point_normal'] = m['point_normal']
        step_batch.append(step_item)

    return {
        'bbox_inputs': _collate_two_stage_single_task(
            bbox_batch,
            processor=processor,
            n_point_tokens=n_point_tokens,
            n_type_tokens=n_type_tokens,
            n_part_point_tokens=n_part_point_tokens,
            eval=eval,
            pc_encoder_type=pc_encoder_type,
            utonia_scale=utonia_scale,
            utonia_normalize_coord=utonia_normalize_coord,
            utonia_use_normal=utonia_use_normal,
            build_utonia_data_dict=build_utonia_data_dict,
            bbox_grounding_enabled=bbox_grounding_enabled,
            bbox_grounding_normalize_std=bbox_grounding_normalize_std,
            bbox_grounding_max_boxes=bbox_grounding_max_boxes),
        'step_inputs': _collate_two_stage_single_task(
            step_batch,
            processor=processor,
            n_point_tokens=n_point_tokens,
            n_type_tokens=n_type_tokens,
            n_part_point_tokens=n_part_point_tokens,
            eval=eval,
            pc_encoder_type=pc_encoder_type,
            utonia_scale=utonia_scale,
            utonia_normalize_coord=utonia_normalize_coord,
            utonia_use_normal=utonia_use_normal,
            build_utonia_data_dict=build_utonia_data_dict,
            bbox_grounding_enabled=bbox_grounding_enabled,
            bbox_grounding_normalize_std=bbox_grounding_normalize_std,
            bbox_grounding_max_boxes=bbox_grounding_max_boxes),
    }


def collate(
    batch,
    processor,
    n_points,
    n_point_tokens=256,
    n_type_tokens=1,
    n_part_point_tokens=64,
    eval=False,
    pc_encoder_type='pointbert',
    utonia_scale=1.0,
    utonia_normalize_coord=True,
    utonia_use_normal=False,
    build_utonia_data_dict=True,
    bbox_grounding_enabled=False,
    bbox_grounding_normalize_std=100.0,
    bbox_grounding_max_boxes=0):
    if n_point_tokens <= 0:
        raise ValueError('n_point_tokens must be a positive integer.')
    if n_type_tokens < 0:
        raise ValueError('n_type_tokens must be >= 0.')
    if n_part_point_tokens <= 0:
        raise ValueError('n_part_point_tokens must be a positive integer.')

    if len(batch) > 0 and batch[0].get('task_type') in ('bbox', 'step_pc'):
        return _collate_two_stage_single_task(
            batch=batch,
            processor=processor,
            n_point_tokens=n_point_tokens,
            n_type_tokens=n_type_tokens,
            n_part_point_tokens=n_part_point_tokens,
            eval=eval,
            pc_encoder_type=pc_encoder_type,
            utonia_scale=utonia_scale,
            utonia_normalize_coord=utonia_normalize_coord,
            utonia_use_normal=utonia_use_normal,
            build_utonia_data_dict=build_utonia_data_dict,
            bbox_grounding_enabled=bbox_grounding_enabled,
            bbox_grounding_normalize_std=bbox_grounding_normalize_std,
            bbox_grounding_max_boxes=bbox_grounding_max_boxes)

    if len(batch) > 0 and 'bbox_description' in batch[0] and 'step_description' in batch[0]:
        return _collate_two_stage(
            batch=batch,
            processor=processor,
            n_point_tokens=n_point_tokens,
            n_type_tokens=n_type_tokens,
            n_part_point_tokens=n_part_point_tokens,
            eval=eval,
            pc_encoder_type=pc_encoder_type,
            utonia_scale=utonia_scale,
            utonia_normalize_coord=utonia_normalize_coord,
            utonia_use_normal=utonia_use_normal,
            build_utonia_data_dict=build_utonia_data_dict,
            bbox_grounding_enabled=bbox_grounding_enabled,
            bbox_grounding_normalize_std=bbox_grounding_normalize_std,
            bbox_grounding_max_boxes=bbox_grounding_max_boxes)

    batch, bbox_grounding_records = _prepare_bbox_grounding_batch(
        batch,
        enabled=bbox_grounding_enabled,
        normalize_std=bbox_grounding_normalize_std,
        max_boxes=bbox_grounding_max_boxes)

    messages = []
    is_pc = [0] * len(batch)
    is_img = [0] * len(batch)
    if not eval:
        for i, m in enumerate(batch):
            if 'video' in m.keys():
                is_img[i] = 1
                message = [{
                        'role': 'user',
                        'content': [
                            {'type': 'video', 'video': m['video'], 'fps': 1.0},
                            {'type': 'text', 'text': m['description']}
                        ]
                    },
                    {
                        'role': 'sassistant',
                        'content': [
                            {'type': 'text', 'text': m['answer']}
                        ]
                    }]
            else:
                if 'point_cloud' in m.keys():
                    is_pc[i] = 1
                message = [{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': m['description']}
                        ]
                    },
                    {
                        'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': m['answer']}
                        ]
                    }]
            messages.append(message)
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages]
    else:
        for i, m in enumerate(batch):
            if 'video' in m.keys():
                is_img[i] = 1
                message = [{
                        'role': 'user',
                        'content': [
                            {'type': 'video', 'video': m['video'], 'fps': 1.0},
                            {'type': 'text', 'text': m['description']}
                        ]
                    }]
            else:
                if 'point_cloud' in m.keys():
                    is_pc[i] = 1
                message = [{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': m['description']}
                        ]
                    }]
            messages.append(message)
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages]


    points_inputs = ''.join((n_type_tokens + n_point_tokens) * [processor.tokenizer.pad_token])

    for i in range(len(texts)):
        if is_pc[i]:
            texts[i] = points_inputs + texts[i]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt')

    point_feature_dim = _infer_point_feature_dim(batch)
    point_cloud_tensors = []
    for i, m in enumerate(batch):
        if is_pc[i]:
            point_cloud_tensors.append(
                _compose_point_cloud_tensor(m, n_points=n_points, feature_dim=point_feature_dim))
        else:
            point_cloud_tensors.append(torch.zeros(n_points, point_feature_dim, dtype=torch.float32))
    inputs['point_clouds'] = torch.stack(point_cloud_tensors, dim=0)
    if build_utonia_data_dict and str(pc_encoder_type).lower() == 'utonia':
        inputs['utonia_data_dict'] = _build_utonia_data_dict_from_tensors(
            point_cloud_tensors=point_cloud_tensors,
            scale=utonia_scale,
            normalize_coord=utonia_normalize_coord,
            use_normal=utonia_use_normal)
    inputs['is_pc'] = torch.tensor(is_pc, dtype=torch.bool)
    inputs['is_img'] = torch.tensor(is_img, dtype=torch.bool)
    inputs['pc_type_ids'] = torch.tensor(
        [int(m.get('pc_type_id', 0)) if is_pc[i] else 0 for i, m in enumerate(batch)],
        dtype=torch.long)
    inputs['source_ids'] = torch.tensor(
        [
            (_stable_hash_to_int64(source_key) if source_key else -1)
            if is_pc[i] else -1
            for i, source_key in enumerate(_infer_source_key_for_collate(m) for m in batch)
        ],
        dtype=torch.long)
    inputs['task_ids'] = torch.tensor(
        [_task_type_to_id(m.get('task_type', 'code')) if is_pc[i] else 0
         for i, m in enumerate(batch)],
        dtype=torch.long)
    _attach_bbox_grounding_targets(inputs, bbox_grounding_records)

    if 'pixel_values_videos' in inputs.keys():
        pixel_values_videos = inputs['pixel_values_videos'].new_zeros((
            len(batch), torch.prod(inputs['video_grid_thw'][0]),
            inputs['pixel_values_videos'].shape[1]))
        pixel_values_videos[inputs['is_img']] = torch.stack(
            torch.chunk(inputs['pixel_values_videos'],
            chunks=sum(inputs['is_img'])))
        inputs['pixel_values_videos'] = pixel_values_videos

        video_grid_thw = inputs['video_grid_thw'].new_zeros((len(batch), 3))
        video_grid_thw[inputs['is_img']] = inputs['video_grid_thw']
        inputs['video_grid_thw'] = video_grid_thw

    if not eval:
        input_ids_lists = inputs['input_ids'].tolist()
        assert len(messages) == len(input_ids_lists)

        labels_list = []
        for ids_list in input_ids_lists:
            label_ids = [-100] * len(ids_list)
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0] + 2: begin_end_indexs[1] + 1] = \
                    ids_list[begin_end_indexs[0] + 2: begin_end_indexs[1] + 1]
            labels_list.append(label_ids)
        labels_ids = torch.tensor(labels_list, dtype=torch.int64)
        inputs['labels'] = labels_ids
    else:
        inputs['file_name'] = [m['file_name'] for m in batch]
    return inputs


def find_assistant_content_sublist_indexes(l):
    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 1):
        # Check if the current and next element form the start sequence
        if l[i] == 151644 and l[i + 1] == 77091:
            start_indexes.append(i)
            # Now look for the first 151645 after the start
            for j in range(i + 2, len(l)):
                if l[j] == 151645:
                    end_indexes.append(j)
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=8, logspace=True, include_input=True, include_pi=False):
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies *= torch.pi
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.include_input = include_input

    def forward(self, x):
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class KNNInterpolator(nn.Module):
    def __init__(self, k=3, temperature=0.1):
        super().__init__()
        self.k = k
        self.temperature = temperature

    def forward(self, points, patch_centers, patch_features):
        dist = torch.cdist(points, patch_centers)
        topk_dist, topk_idx = torch.topk(dist, k=self.k, dim=-1, largest=False)
        weights = F.softmax(-topk_dist / self.temperature, dim=-1)

        bsz, n_points, k = topk_idx.shape
        _, n_patches, feat_dim = patch_features.shape

        patch_features_expanded = patch_features.unsqueeze(1).expand(bsz, n_points, n_patches, feat_dim)
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(bsz, n_points, k, feat_dim)
        topk_features = torch.gather(patch_features_expanded, dim=2, index=topk_idx_expanded)
        return (topk_features * weights.unsqueeze(-1)).sum(dim=2)


class PointFeatureFusion(nn.Module):
    def __init__(self, pointbert_dim=384, fourier_dim=51, branch_dim=256, fusion_dim=512):
        super().__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs=8, include_pi=False)

        # Balance two branches before fusion:
        # PointBERT: 384 -> 256, Fourier: 51 -> 256
        self.pointbert_proj = nn.Sequential(
            nn.Linear(pointbert_dim, branch_dim),
            nn.GELU(),
            nn.LayerNorm(branch_dim),
        )
        self.fourier_proj = nn.Sequential(
            nn.Linear(fourier_dim, branch_dim),
            nn.GELU(),
            nn.LayerNorm(branch_dim),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(branch_dim * 2, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, fusion_dim),
        )

    def forward(self, points, pointbert_features):
        fourier_coords = self.fourier_embedder(points)
        target_dtype = next(self.fusion_mlp.parameters()).dtype
        pointbert_embed = self.pointbert_proj(pointbert_features.to(target_dtype))
        fourier_embed = self.fourier_proj(fourier_coords.to(target_dtype))
        combined = torch.cat([pointbert_embed, fourier_embed], dim=-1)
        return self.fusion_mlp(combined)

class PointBertEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_tokens=256,
        fusion_dim=512,
        k_neighbors=3,
        pointbert_root=None,
        config_path=None,
        ckpt_path=None,
        dvae_ckpt_path=None):
        super().__init__()

        self.pointbert_root = Path(
            pointbert_root if pointbert_root is not None
            else os.environ.get('POINTBERT_ROOT', PROJECT_ROOT / 'Point-BERT')
        ).expanduser().resolve()
        self.config_path = Path(
            config_path if config_path is not None
            else os.environ.get('POINTBERT_CONFIG', self.pointbert_root / 'cfgs/Mixup_models/Point-BERT.yaml')
        ).expanduser().resolve()
        self.dvae_ckpt_path = Path(
            dvae_ckpt_path if dvae_ckpt_path is not None
            else os.environ.get('POINTBERT_DVAE_CHECKPOINT', self.pointbert_root / 'ckpt/dVAE.pth')
        ).expanduser().resolve()
        self.ckpt_path = Path(
            ckpt_path if ckpt_path is not None
            else os.environ.get('POINTBERT_CHECKPOINT', self.pointbert_root / 'ckpt/Point-BERT.pth')
        ).expanduser().resolve()

        for path in (self.config_path, self.dvae_ckpt_path, self.ckpt_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f'Point-BERT asset not found: {path}. Set POINTBERT_ROOT or the '
                    'POINTBERT_CONFIG / POINTBERT_DVAE_CHECKPOINT / POINTBERT_CHECKPOINT overrides.')

        if str(self.pointbert_root) not in sys.path:
            sys.path.insert(0, str(self.pointbert_root))
        import yaml
        from easydict import EasyDict
        from tools import builder

        # Only the model is needed; dataset _base_ paths belong to Point-BERT training.
        with self.config_path.open(encoding='utf-8') as stream:
            model_config = EasyDict(yaml.safe_load(stream)['model'])
        model_config.dvae_config.ckpt = str(self.dvae_ckpt_path)

        pointbert_model = builder.model_builder(model_config)
        self._load_pointbert_weights(pointbert_model, self.ckpt_path)
        pointbert_model.eval()

        self.group_divider = pointbert_model.group_divider
        self.transformer = pointbert_model.transformer_q
        self.pointbert_dim = self.transformer.trans_dim
        self.num_groups = getattr(pointbert_model, 'num_group', model_config.dvae_config.num_group)

        self.knn_interpolator = KNNInterpolator(k=k_neighbors, temperature=0.1)
        self.feature_fusion = PointFeatureFusion(
            pointbert_dim=self.pointbert_dim,
            fourier_dim=51,
            fusion_dim=fusion_dim,
        )
        self.projection_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Linear(fusion_dim * 2, hidden_size),
        )

        self.n_stage1_tokens = int(num_tokens)
        self.num_tokens = self.n_stage1_tokens
        self.fusion_dim = fusion_dim
        self._freeze_and_fix_dtype()

    def _freeze_and_fix_dtype(self):
        if any(param.is_meta for param in self.parameters()):
            return

        self.group_divider.eval()
        self.transformer.eval()

        # PointBERT 核心部分保持 float32
        for module in (self.group_divider, self.transformer):
            module.to(dtype=torch.float32)
            for param in module.parameters():
                param.requires_grad_(False)
                param.data = param.data.float()
            for buffer_name, buffer in module.named_buffers():
                if buffer is not None:
                    buffer.data = buffer.data.float()

        # 新增融合与投影模块用 bf16 以匹配 Qwen
        for module in (self.feature_fusion, self.projection_mlp):
            module.to(dtype=torch.bfloat16)

    def _load_pointbert_weights(self, pointbert_model, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Point-BERT checkpoint not found: {ckpt_path}')

        raw_state = torch.load(ckpt_path, map_location='cpu')
        state_dict = raw_state
        if isinstance(raw_state, dict):
            for key in ('model', 'base_model', 'state_dict'):
                if isinstance(raw_state.get(key, None), dict):
                    state_dict = raw_state[key]
                    break

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        try:
            pointbert_model.load_state_dict(state_dict, strict=True)
            return state_dict
        except RuntimeError:
            pass

        def _remap_keys(sd):
            remapped = {}
            for k, v in sd.items():
                new_k = k
                if new_k.startswith('point_encoder.'):
                    new_k = new_k[len('point_encoder.'):]
                if new_k.startswith('transformer.'):
                    new_k = 'transformer_q.' + new_k[len('transformer.'):]
                remapped[new_k] = v
            return remapped

        remapped_state = _remap_keys(state_dict)
        incompatible = pointbert_model.load_state_dict(remapped_state, strict=False)

        total_params = len(list(pointbert_model.state_dict().keys()))
        if len(incompatible.missing_keys) > total_params * 0.5:
            raise RuntimeError(
                f'Failed to load Point-BERT weights: more than 50% of parameters are missing. '
                f'This usually means the checkpoint format is incompatible.')

        return remapped_state

    def reload_backbone_weights(self):
        import yaml
        from easydict import EasyDict
        from tools import builder

        with self.config_path.open(encoding='utf-8') as stream:
            model_config = EasyDict(yaml.safe_load(stream)['model'])
        model_config.dvae_config.ckpt = str(self.dvae_ckpt_path)

        pointbert_model = builder.model_builder(model_config)
        self._load_pointbert_weights(pointbert_model, self.ckpt_path)
        pointbert_model.eval()

        self.group_divider.load_state_dict(pointbert_model.group_divider.state_dict())
        self.transformer.load_state_dict(pointbert_model.transformer_q.state_dict())
        self._freeze_and_fix_dtype()
        return self

    @staticmethod
    def _normalize_points(points):
        centroid = points.mean(dim=1, keepdim=True)
        points = points - centroid
        scale = points.norm(dim=-1).amax(dim=1, keepdim=True)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return points / scale.unsqueeze(-1)

    @staticmethod
    def _gather_features(features, indices):
        # features: [B, N, C], indices: [B, ...]
        bsz, _, feat_dim = features.shape
        flat_indices = indices.reshape(bsz, -1)
        gather_index = flat_indices.unsqueeze(-1).expand(-1, -1, feat_dim)
        gathered = torch.gather(features, dim=1, index=gather_index)
        out_shape = list(indices.shape) + [feat_dim]
        return gathered.view(*out_shape)

    def _encode_fused_features(self, points):
        with torch.no_grad():
            pts = self._normalize_points(points)

            neighborhood, center = self.group_divider(pts)

            group_tokens = self.transformer.encoder(neighborhood)
            group_tokens = self.transformer.reduce_dim(group_tokens)

            cls_tokens = self.transformer.cls_token.expand(group_tokens.size(0), -1, -1)
            cls_pos = self.transformer.cls_pos.expand(group_tokens.size(0), -1, -1)
            pos_embed = self.transformer.pos_embed(center)

            x = torch.cat((cls_tokens, group_tokens), dim=1)
            pos = torch.cat((cls_pos, pos_embed), dim=1)
            x = self.transformer.blocks(x, pos)
            x = self.transformer.norm(x)
            patch_features = x[:, 1:, :]

        interpolated_features = self.knn_interpolator(pts, center, patch_features)
        return self.feature_fusion(points, interpolated_features)

    def encode_global_and_part_tokens(self, points, part_point_indices=None, return_token_coords=False):
        # 确保输入是 float32（训练框架可能自动转为 bf16）
        points = points[..., :3].float()

        if points.size(1) == 0:
            global_tokens = torch.zeros(
                points.size(0),
                self.n_stage1_tokens,
                self.projection_mlp[-1].out_features,
                device=points.device,
                dtype=torch.bfloat16)
            if part_point_indices is None:
                if return_token_coords:
                    empty_coords = torch.zeros(
                        points.size(0),
                        self.n_stage1_tokens,
                        3,
                        device=points.device,
                        dtype=torch.float32)
                    return global_tokens, None, empty_coords
                return global_tokens, None
            empty_part = torch.zeros(
                part_point_indices.size(0),
                part_point_indices.size(1),
                part_point_indices.size(2),
                self.projection_mlp[-1].out_features,
                device=points.device,
                dtype=torch.bfloat16)
            if return_token_coords:
                empty_coords = torch.zeros(
                    points.size(0),
                    self.n_stage1_tokens,
                    3,
                    device=points.device,
                    dtype=torch.float32)
                return global_tokens, empty_part, empty_coords
            return global_tokens, empty_part

        fused_features = self._encode_fused_features(points)
        _, indices = sample_farthest_points(points, K=self.n_stage1_tokens)
        sampled_features = self._gather_features(fused_features, indices)
        global_tokens = self.projection_mlp(sampled_features.bfloat16())
        token_coords = self._gather_features(points, indices) if return_token_coords else None

        if part_point_indices is None:
            if return_token_coords:
                return global_tokens, None, token_coords
            return global_tokens, None

        max_valid_index = points.size(1) - 1
        clipped_indices = part_point_indices.long().clamp(0, max_valid_index)
        part_features = self._gather_features(fused_features, clipped_indices)
        part_tokens = self.projection_mlp(part_features.bfloat16())
        if return_token_coords:
            return global_tokens, part_tokens, token_coords
        return global_tokens, part_tokens

    def forward(self, points):
        global_tokens, _ = self.encode_global_and_part_tokens(points, part_point_indices=None)
        return global_tokens

    def to(self, *args, **kwargs):
        # 只移动设备，不改变 dtype
        device = kwargs.get('device', None)
        for arg in args:
            if isinstance(arg, torch.device):
                device = arg
                break

        if device is not None:
            self.group_divider.to(device=device)
            self.transformer.to(device=device)
            self.knn_interpolator.to(device=device)
            self.feature_fusion.to(device=device)
            self.projection_mlp.to(device=device)

        return self

    def train(self, mode: bool = True):
        super().train(False)
        self.group_divider.eval()
        self.transformer.eval()
        self.knn_interpolator.train(mode)
        self.feature_fusion.train(mode)
        self.projection_mlp.train(mode)
        return self


class UtoniaEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_tokens=256,
        model_name='utonia',
        repo_id='Pointcept/Utonia',
        download_root=None,
        ckpt_path=None,
        freeze_backbone=True,
        scale=1.0,
        normalize_coord=True,
        use_normal=False):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_stage1_tokens = int(num_tokens)
        self.num_tokens = self.n_stage1_tokens
        if ckpt_path is None:
            ckpt_path = os.environ.get('UTONIA_CHECKPOINT')
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path).expanduser().resolve()
            if not ckpt_path.is_file():
                raise FileNotFoundError(f'Utonia checkpoint not found: {ckpt_path}')
        if download_root is None:
            download_root = os.environ.get('UTONIA_DOWNLOAD_ROOT')
        self.model_name = str(ckpt_path) if ckpt_path is not None else str(model_name)
        self.repo_id = repo_id
        self.download_root = (
            str(Path(download_root).expanduser().resolve()) if download_root is not None else None)
        self.freeze_backbone = bool(freeze_backbone)
        self.scale = float(scale)
        self.normalize_coord = bool(normalize_coord)
        self.use_normal = bool(use_normal)
        self.utonia = self._import_utonia()
        self.utonia_transform = self.utonia.transform.default(
            scale=self.scale,
            apply_z_positive=True,
            normalize_coord=self.normalize_coord)
        # Official README explicitly uses range(4) for quantitative features and
        # range(2) only for visualization. Downstream training should keep the
        # quantitative setting instead of the demo visualization shortcut.
        self.demo_concat_upcast_levels = 4

        self.backbone, self.backbone_config = self._build_backbone_and_config()
        self.utonia_in_channels = int(self.backbone_config.get('in_channels', 6))
        self.backbone_out_dim = self._infer_output_dim(self.backbone_config)
        self.feature_fusion = PointFeatureFusion(
            pointbert_dim=self.backbone_out_dim,
            fourier_dim=51,
            fusion_dim=self.backbone_out_dim,
        )
        self.projection_mlp = nn.Sequential(
            nn.Linear(self.backbone_out_dim, self.backbone_out_dim * 2),
            nn.GELU(),
            nn.Linear(self.backbone_out_dim * 2, self.hidden_size),
        )
        self._freeze_and_fix_dtype()

    @staticmethod
    def _insert_local_utonia_path():
        local_root = PROJECT_ROOT / 'Utonia-main'
        if local_root.exists():
            local_root_str = str(local_root)
            if local_root_str not in sys.path:
                sys.path.insert(0, local_root_str)

    @classmethod
    def _import_utonia(cls):
        try:
            import utonia  # noqa: F401
            return utonia
        except ImportError:
            cls._insert_local_utonia_path()
            import utonia  # noqa: F401
            return utonia

    def _build_backbone_and_config(self):
        load_kwargs = {}
        if self.download_root is not None:
            load_kwargs['download_root'] = self.download_root

        ckpt = self.utonia.model.load(
            self.model_name,
            repo_id=self.repo_id,
            ckpt_only=True,
            **load_kwargs)
        config = dict(ckpt.get('config', {}))
        model = self.utonia.model.load(
            self.model_name,
            repo_id=self.repo_id,
            **load_kwargs)
        model.eval()
        return model, config

    def _infer_output_dim(self, config):
        if bool(config.get('enc_mode', False)):
            enc_channels = config.get('enc_channels', None)
            if isinstance(enc_channels, (list, tuple)) and len(enc_channels) > 0:
                concat_levels = min(self.demo_concat_upcast_levels, max(len(enc_channels) - 1, 0))
                start = len(enc_channels) - (concat_levels + 1)
                return int(sum(int(c) for c in enc_channels[start:]))
        dec_channels = config.get('dec_channels', None)
        if isinstance(dec_channels, (list, tuple)) and len(dec_channels) > 0:
            return int(dec_channels[0])
        raise RuntimeError('Unable to infer Utonia output feature dimension from checkpoint config.')

    def _freeze_and_fix_dtype(self):
        if any(param.is_meta for param in self.parameters()):
            return

        self.backbone.eval()
        self.backbone.to(dtype=torch.float32)
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        for module in (self.feature_fusion, self.projection_mlp):
            module.to(dtype=torch.bfloat16)

    @staticmethod
    def _gather_features(features, indices):
        bsz, _, feat_dim = features.shape
        flat_indices = indices.reshape(bsz, -1)
        gather_index = flat_indices.unsqueeze(-1).expand(-1, -1, feat_dim)
        gathered = torch.gather(features, dim=1, index=gather_index)
        out_shape = list(indices.shape) + [feat_dim]
        return gathered.view(*out_shape)

    def _build_input_feature(self, coord):
        in_channels = max(self.utonia_in_channels, 1)
        feat = coord.new_zeros((coord.size(0), in_channels))
        n_copy = min(3, in_channels)
        feat[:, :n_copy] = coord[:, :n_copy]
        return feat

    def _to_utonia_data_dict(self, points):
        points_list = []
        for i in range(points.size(0)):
            point_np = points[i].detach().cpu().numpy().astype(np.float32, copy=False)
            coord_np = point_np[:, :3]
            zeros = np.zeros_like(coord_np, dtype=np.float32)
            if self.use_normal and point_np.shape[1] >= 6:
                normal_np = point_np[:, 3:6]
            else:
                normal_np = zeros
            sample = {
                'coord': coord_np,
                'color': zeros,
                'normal': normal_np,
            }
            sample = self.utonia_transform(sample)
            points_list.append(sample)

        data_dict = self.utonia.data.collate_fn(points_list)
        for key, value in list(data_dict.items()):
            if torch.is_tensor(value):
                data_dict[key] = value.to(points.device, non_blocking=True)
        return data_dict

    def _upcast_demo_style(self, point):
        for _ in range(self.demo_concat_upcast_levels):
            if 'pooling_parent' not in point.keys() or 'pooling_inverse' not in point.keys():
                raise RuntimeError(
                    'Utonia hierarchy mismatch with demo-style upcast: '
                    'missing pooling_parent/pooling_inverse during concat stage.')
            parent = point.pop('pooling_parent')
            inverse = point.pop('pooling_inverse')
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        while 'pooling_parent' in point.keys():
            if 'pooling_inverse' not in point.keys():
                raise RuntimeError(
                    'Utonia hierarchy mismatch with demo-style upcast: '
                    'missing pooling_inverse during projection stage.')
            parent = point.pop('pooling_parent')
            inverse = point.pop('pooling_inverse')
            parent.feat = point.feat[inverse]
            point = parent
        return point

    def _move_data_dict_to_device(self, data_dict, device):
        moved = {}
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    @staticmethod
    def _project_sparse_feat_to_dense(point, bsz, n_points):
        if 'inverse' not in point.keys():
            raise RuntimeError('Utonia transform output missing inverse for projection to original points.')

        inverse = point.inverse.long()
        expected_dense = int(bsz) * int(n_points)
        if inverse.numel() != expected_dense:
            raise RuntimeError(
                f'Utonia inverse size mismatch: expected {expected_dense}, got {inverse.numel()}.')

        if bsz <= 1:
            return point.feat[inverse]

        if 'offset' not in point.keys():
            raise RuntimeError('Utonia point output missing offset for batched dense projection.')

        sparse_offset = point.offset.long()
        if sparse_offset.numel() != bsz:
            raise RuntimeError(
                f'Utonia sparse offset size mismatch: expected {bsz}, got {sparse_offset.numel()}.')

        sparse_start = torch.cat([sparse_offset.new_zeros(1), sparse_offset[:-1]], dim=0)
        sample_ids = torch.arange(bsz, device=inverse.device, dtype=torch.long).repeat_interleave(n_points)
        adjusted_inverse = inverse + sparse_start[sample_ids]
        return point.feat[adjusted_inverse]

    def _encode_point_features(self, points, utonia_data_dict=None):
        bsz, n_points, _ = points.shape
        with torch.no_grad():
            if utonia_data_dict is None:
                data_dict = self._to_utonia_data_dict(points)
            else:
                data_dict = self._move_data_dict_to_device(utonia_data_dict, points.device)
            point = self.backbone(data_dict)
            point = self._upcast_demo_style(point)

            feat = self._project_sparse_feat_to_dense(point, bsz=bsz, n_points=n_points)

        if feat.size(-1) != self.backbone_out_dim:
            raise RuntimeError(
                f'Utonia feature dim mismatch after demo-style upcast: '
                f'expected {self.backbone_out_dim}, got {feat.size(-1)}.')

        expected_rows = bsz * n_points
        if feat.size(0) != expected_rows:
            raise RuntimeError(
                f'Utonia projected point count mismatch: expected {expected_rows}, got {feat.size(0)}.')
        return feat.view(bsz, n_points, -1)

    def _encode_fused_features(self, points, utonia_data_dict=None):
        coords = points[..., :3].float()
        point_features = self._encode_point_features(points, utonia_data_dict=utonia_data_dict)
        return self.feature_fusion(coords, point_features)

    def reload_backbone_weights(self):
        backbone, config = self._build_backbone_and_config()
        new_out_dim = self._infer_output_dim(config)
        if new_out_dim != self.backbone_out_dim:
            raise RuntimeError(
                f'Utonia output dim changed from {self.backbone_out_dim} to {new_out_dim}; '
                'refusing to reload to avoid projection mismatch.')
        self.backbone.load_state_dict(backbone.state_dict(), strict=True)
        self.backbone_config = config
        self.utonia_in_channels = int(self.backbone_config.get('in_channels', self.utonia_in_channels))
        self._freeze_and_fix_dtype()
        return self

    def encode_global_and_part_tokens(
        self,
        points,
        part_point_indices=None,
        utonia_data_dict=None,
        return_token_coords=False):
        points = points.float()
        coords = points[..., :3]

        if points.size(1) == 0:
            global_tokens = torch.zeros(
                points.size(0),
                self.n_stage1_tokens,
                self.hidden_size,
                device=points.device,
                dtype=torch.bfloat16)
            if part_point_indices is None:
                if return_token_coords:
                    empty_coords = torch.zeros(
                        points.size(0),
                        self.n_stage1_tokens,
                        3,
                        device=points.device,
                        dtype=torch.float32)
                    return global_tokens, None, empty_coords
                return global_tokens, None
            empty_part = torch.zeros(
                part_point_indices.size(0),
                part_point_indices.size(1),
                part_point_indices.size(2),
                self.hidden_size,
                device=points.device,
                dtype=torch.bfloat16)
            if return_token_coords:
                empty_coords = torch.zeros(
                    points.size(0),
                    self.n_stage1_tokens,
                    3,
                    device=points.device,
                    dtype=torch.float32)
                return global_tokens, empty_part, empty_coords
            return global_tokens, empty_part

        fused_features = self._encode_fused_features(points, utonia_data_dict=utonia_data_dict)
        _, indices = sample_farthest_points(coords, K=self.n_stage1_tokens)
        sampled_features = self._gather_features(fused_features, indices)
        global_tokens = self.projection_mlp(sampled_features.bfloat16())
        token_coords = self._gather_features(coords, indices) if return_token_coords else None

        if part_point_indices is None:
            if return_token_coords:
                return global_tokens, None, token_coords
            return global_tokens, None

        max_valid_index = points.size(1) - 1
        clipped_indices = part_point_indices.long().clamp(0, max_valid_index)
        part_features = self._gather_features(fused_features, clipped_indices)
        part_tokens = self.projection_mlp(part_features.bfloat16())
        if return_token_coords:
            return global_tokens, part_tokens, token_coords
        return global_tokens, part_tokens

    def forward(self, points, utonia_data_dict=None):
        global_tokens, _ = self.encode_global_and_part_tokens(
            points,
            part_point_indices=None,
            utonia_data_dict=utonia_data_dict)
        return global_tokens

    def to(self, *args, **kwargs):
        device = kwargs.get('device', None)
        for arg in args:
            if isinstance(arg, torch.device):
                device = arg
                break

        if device is not None:
            self.backbone.to(device=device)
            self.feature_fusion.to(device=device)
            self.projection_mlp.to(device=device)
        return self

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        self.feature_fusion.train(mode)
        self.projection_mlp.train(mode)
        return self


class BBoxGroundingHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.box_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 6),
        )
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.point_proj = nn.Linear(hidden_size, hidden_size)
        self.mask_bias = nn.Linear(hidden_size, 1)

    def forward(self, box_hidden, point_embeds):
        head_dtype = self.box_mlp[1].weight.dtype
        box_hidden = box_hidden.to(dtype=head_dtype)
        point_embeds = point_embeds.to(dtype=head_dtype)
        raw_boxes = self.box_mlp(box_hidden).float()
        lower = torch.minimum(raw_boxes[..., :3], raw_boxes[..., 3:])
        upper = torch.maximum(raw_boxes[..., :3], raw_boxes[..., 3:])
        pred_boxes = torch.cat([lower, upper], dim=-1)
        query = F.normalize(self.query_proj(box_hidden), dim=-1)
        point = F.normalize(self.point_proj(point_embeds), dim=-1)
        logits = torch.einsum('bkh,bnh->bkn', query, point)
        logits = logits + self.mask_bias(box_hidden)
        return pred_boxes, logits.float()


class Cadrille(Qwen2VLForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        user_output_loading_info = kwargs.pop("output_loading_info", False)

        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            output_loading_info=True,
            **kwargs,
        )

        # HF from_pretrained 可能覆盖点云编码器 backbone，显式重载确保权重正确。
        if isinstance(model, cls):
            model.reload_point_encoder_weights()

        if user_output_loading_info:
            return model, loading_info
        return model

    def __init__(
        self,
        config,
        two_stage=False,
        n_point_tokens=256,
        part_points_per_bbox=64,
        pc_encoder_type='pointbert',
        utonia_scale=1.0,
        utonia_normalize_coord=True,
        utonia_use_normal=False,
        contrastive_enabled=False,
        contrastive_lambda=0.05,
        contrastive_temperature=0.07,
        contrastive_proj_dim=128,
        contrastive_queue_size=4096,
        contrastive_momentum=0.999,
        contrastive_interval=1,
        contrastive_max_negatives=1024,
        bbox_grounding_enabled=False,
        bbox_grounding_lambda=0.1,
        bbox_grounding_lambda_box=1.0,
        bbox_grounding_lambda_mask=0.25,
        bbox_grounding_match_mask_weight=0.5,
        bbox_grounding_mask_eps=0.01):
        super().__init__(config)

        n_point_tokens = int(n_point_tokens)
        part_points_per_bbox = int(part_points_per_bbox)
        if n_point_tokens <= 0:
            raise ValueError('n_point_tokens must be a positive integer.')
        if part_points_per_bbox <= 0:
            raise ValueError('part_points_per_bbox must be a positive integer.')

        torch.set_default_dtype(torch.float32)
        self.pc_encoder_type = str(pc_encoder_type).lower()
        if self.pc_encoder_type == 'pointbert':
            self.point_encoder = PointBertEncoder(
                config.hidden_size,
                num_tokens=n_point_tokens)
        elif self.pc_encoder_type == 'utonia':
            self.point_encoder = UtoniaEncoder(
                config.hidden_size,
                num_tokens=n_point_tokens,
                scale=utonia_scale,
                normalize_coord=utonia_normalize_coord,
                use_normal=utonia_use_normal)
        else:
            raise ValueError(
                f'Unsupported pc_encoder_type: {pc_encoder_type}. '
                "Expected one of ['pointbert', 'utonia'].")
        self.n_point_tokens = self.point_encoder.num_tokens
        self.two_stage = bool(two_stage)
        self.part_points_per_bbox = part_points_per_bbox
        self.model_pc_type_embedding = nn.Embedding(2, config.hidden_size)
        self.model_pc_start_embedding = nn.Parameter(torch.randn(config.hidden_size))
        self.model_pc_end_embedding = nn.Parameter(torch.randn(config.hidden_size))

        self.bbox_grounding_enabled = bool(bbox_grounding_enabled)
        self.bbox_grounding_lambda = float(bbox_grounding_lambda)
        self.bbox_grounding_lambda_box = float(bbox_grounding_lambda_box)
        self.bbox_grounding_lambda_mask = float(bbox_grounding_lambda_mask)
        self.bbox_grounding_match_mask_weight = float(bbox_grounding_match_mask_weight)
        self.bbox_grounding_mask_eps = float(bbox_grounding_mask_eps)
        self.bbox_grounding_box_token_id = -1
        self.bbox_grounding_span_start_token_id = -1
        self.bbox_grounding_span_end_token_id = -1
        self.bbox_grounding_split_token_id = -1
        self.bbox_grounding_stop_token_id = -1
        if self.bbox_grounding_enabled:
            self.bbox_grounding_head = BBoxGroundingHead(config.hidden_size)
        else:
            self.bbox_grounding_head = None

        self.contrastive_enabled = bool(contrastive_enabled)
        self.contrastive_lambda = float(contrastive_lambda)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_proj_dim = int(contrastive_proj_dim)
        self.contrastive_queue_size = int(contrastive_queue_size)
        self.contrastive_momentum = float(contrastive_momentum)
        self.contrastive_interval = max(int(contrastive_interval), 1)
        self.contrastive_max_negatives = int(contrastive_max_negatives)
        self.register_buffer(
            'contrastive_forward_counter',
            torch.zeros(1, dtype=torch.long),
            persistent=False)

        if self.contrastive_enabled:
            if self.contrastive_proj_dim <= 0:
                raise ValueError('contrastive_proj_dim must be a positive integer.')
            if self.contrastive_queue_size <= 0:
                raise ValueError('contrastive_queue_size must be a positive integer.')
            if self.contrastive_temperature <= 0:
                raise ValueError('contrastive_temperature must be positive.')
            if not (0.0 <= self.contrastive_momentum < 1.0):
                raise ValueError('contrastive_momentum must be in [0, 1).')

            proj_in_dim = config.hidden_size * 2
            self.contrastive_q_proj = nn.Sequential(
                nn.Linear(proj_in_dim, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, self.contrastive_proj_dim),
            )
            self.contrastive_k_proj = nn.Sequential(
                nn.Linear(proj_in_dim, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, self.contrastive_proj_dim),
            )
            self.contrastive_k_proj.load_state_dict(self.contrastive_q_proj.state_dict())
            for param in self.contrastive_k_proj.parameters():
                param.requires_grad = False

            self.register_buffer(
                'contrastive_queue',
                torch.zeros(self.contrastive_queue_size, self.contrastive_proj_dim),
                persistent=False)
            self.register_buffer(
                'contrastive_queue_source_ids',
                torch.full((self.contrastive_queue_size,), -1, dtype=torch.long),
                persistent=False)
            self.register_buffer(
                'contrastive_queue_task_ids',
                torch.full((self.contrastive_queue_size,), -1, dtype=torch.long),
                persistent=False)
            self.register_buffer(
                'contrastive_queue_type_ids',
                torch.full((self.contrastive_queue_size,), -1, dtype=torch.long),
                persistent=False)
            self.register_buffer(
                'contrastive_queue_ptr',
                torch.zeros(1, dtype=torch.long),
                persistent=False)
            self.register_buffer(
                'contrastive_queue_filled',
                torch.zeros(1, dtype=torch.long),
                persistent=False)
        else:
            self.contrastive_q_proj = None
            self.contrastive_k_proj = None

        self.latest_ce_loss = None
        self.latest_contrastive_loss = None
        self.latest_weighted_contrastive_loss = None
        self.latest_bbox_grounding_loss = None
        self.latest_bbox_grounding_box_loss = None
        self.latest_bbox_grounding_mask_loss = None
        self.latest_bbox_grounding_predictions = None
        self.latest_bbox_grounding_predictions_raw = None
        self.latest_total_loss = None
        self.enable_point_activation_vis = False
        self.enable_bbox_activation_vis = False
        self.latest_point_embeds = None
        self.latest_point_token_xyz = None
        self.latest_point_token_input_positions = None
        torch.set_default_dtype(torch.bfloat16)

    def reload_point_encoder_weights(self, device=None):
        if hasattr(self.point_encoder, 'reload_backbone_weights'):
            self.point_encoder.reload_backbone_weights()

        # 移动到正确的设备
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        self.point_encoder.to(device=device)

        return self

    @torch.no_grad()
    def _momentum_update_contrastive_key_encoder(self):
        if not self.contrastive_enabled:
            return
        momentum = self.contrastive_momentum
        for param_q, param_k in zip(
            self.contrastive_q_proj.parameters(),
            self.contrastive_k_proj.parameters()):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def _enqueue_contrastive(
        self,
        keys,
        source_ids,
        task_ids,
        pc_type_ids,
        valid_mask):
        if not self.contrastive_enabled:
            return
        if keys is None or keys.numel() == 0:
            return

        if valid_mask is None:
            valid_mask = torch.ones(keys.size(0), dtype=torch.bool, device=keys.device)
        valid_mask = valid_mask.bool()
        if valid_mask.sum().item() == 0:
            return

        keys = keys[valid_mask].detach()
        source_ids = source_ids[valid_mask].detach()
        task_ids = task_ids[valid_mask].detach()
        pc_type_ids = pc_type_ids[valid_mask].detach()

        num_new = int(keys.size(0))
        queue_size = int(self.contrastive_queue_size)
        if num_new >= queue_size:
            keys = keys[-queue_size:]
            source_ids = source_ids[-queue_size:]
            task_ids = task_ids[-queue_size:]
            pc_type_ids = pc_type_ids[-queue_size:]
            num_new = queue_size

        ptr = int(self.contrastive_queue_ptr.item())
        end = ptr + num_new

        if end <= queue_size:
            self.contrastive_queue[ptr:end] = keys
            self.contrastive_queue_source_ids[ptr:end] = source_ids
            self.contrastive_queue_task_ids[ptr:end] = task_ids
            self.contrastive_queue_type_ids[ptr:end] = pc_type_ids
        else:
            first = queue_size - ptr
            second = end - queue_size
            self.contrastive_queue[ptr:] = keys[:first]
            self.contrastive_queue_source_ids[ptr:] = source_ids[:first]
            self.contrastive_queue_task_ids[ptr:] = task_ids[:first]
            self.contrastive_queue_type_ids[ptr:] = pc_type_ids[:first]
            self.contrastive_queue[:second] = keys[first:]
            self.contrastive_queue_source_ids[:second] = source_ids[first:]
            self.contrastive_queue_task_ids[:second] = task_ids[first:]
            self.contrastive_queue_type_ids[:second] = pc_type_ids[first:]

        self.contrastive_queue_ptr[0] = end % queue_size
        self.contrastive_queue_filled[0] = min(
            queue_size,
            int(self.contrastive_queue_filled.item()) + num_new)

    def _compute_contrastive_loss(
        self,
        point_embeds,
        pc_type_ids,
        source_ids,
        task_ids,
        is_pc,
        compute_loss=True):
        if not self.contrastive_enabled:
            return None
        if point_embeds is None:
            return None

        device = point_embeds.device
        batch_size = int(point_embeds.size(0))

        if is_pc is None:
            is_pc = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            is_pc = is_pc.to(device=device, dtype=torch.bool)

        if pc_type_ids is None:
            pc_type_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            pc_type_ids = pc_type_ids.to(device=device, dtype=torch.long).clamp(0, 1)

        if source_ids is None:
            source_ids = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        else:
            source_ids = source_ids.to(device=device, dtype=torch.long)

        if task_ids is None:
            task_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            task_ids = task_ids.to(device=device, dtype=torch.long)

        # `-1` is the only sentinel for "invalid source id".
        # Real hashed source ids can be negative, so do not filter by `>= 0`.
        valid_mask = is_pc & (source_ids != -1)
        if valid_mask.sum().item() == 0:
            return None

        global_features = point_embeds.float().mean(dim=1)
        type_features = self.model_pc_type_embedding(pc_type_ids).float()
        contrastive_inputs = torch.cat([global_features, type_features], dim=-1)
        z_q = None
        if compute_loss:
            z_q = F.normalize(self.contrastive_q_proj(contrastive_inputs), dim=-1)

        with torch.no_grad():
            self._momentum_update_contrastive_key_encoder()
            z_k = F.normalize(self.contrastive_k_proj(contrastive_inputs.detach()), dim=-1)

        queue_filled = int(self.contrastive_queue_filled.item())
        if queue_filled > 0:
            # Snapshot queue tensors so later enqueue (in-place ring-buffer updates)
            # cannot invalidate autograd's saved values for this forward pass.
            queue_features = self.contrastive_queue[:queue_filled].detach().clone()
            queue_sources = self.contrastive_queue_source_ids[:queue_filled].detach().clone()
            queue_tasks = self.contrastive_queue_task_ids[:queue_filled].detach().clone()
            queue_types = self.contrastive_queue_type_ids[:queue_filled].detach().clone()
        else:
            queue_features = None
            queue_sources = None
            queue_tasks = None
            queue_types = None

        contrastive_loss = None
        if compute_loss and queue_filled > 0:
            losses = []
            valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)
            for idx in valid_indices.tolist():
                src_id = source_ids[idx]
                task_id = task_ids[idx]
                type_id = pc_type_ids[idx]
                opposite_type = 1 - type_id

                pos_from_batch_mask = (
                    valid_mask
                    & (source_ids == src_id)
                    & (task_ids == task_id)
                    & (pc_type_ids == opposite_type))
                pos_from_batch_mask[idx] = False
                pos_indices = pos_from_batch_mask.nonzero(as_tuple=False).squeeze(-1)
                if pos_indices.numel() > 0:
                    pos_vec = z_k[int(pos_indices[0].item())]
                else:
                    pos_from_queue_mask = (
                        (queue_sources == src_id)
                        & (queue_tasks == task_id)
                        & (queue_types == opposite_type))
                    pos_queue_indices = pos_from_queue_mask.nonzero(as_tuple=False).squeeze(-1)
                    if pos_queue_indices.numel() == 0:
                        continue
                    pos_vec = queue_features[int(pos_queue_indices[0].item())]

                neg_mask = (
                    (queue_tasks == task_id)
                    & (queue_types == opposite_type)
                    & (queue_sources != src_id)
                    & (queue_sources != -1))
                neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(-1)
                if neg_indices.numel() == 0:
                    continue
                if self.contrastive_max_negatives > 0 and neg_indices.numel() > self.contrastive_max_negatives:
                    perm = torch.randperm(neg_indices.numel(), device=neg_indices.device)
                    neg_indices = neg_indices[perm[:self.contrastive_max_negatives]]
                neg_vecs = queue_features[neg_indices]

                pos_logit = torch.sum(z_q[idx] * pos_vec, dim=-1, keepdim=True)
                neg_logits = torch.matmul(neg_vecs, z_q[idx])
                logits = torch.cat([pos_logit, neg_logits], dim=0)
                logits = logits.unsqueeze(0) / self.contrastive_temperature
                target = torch.zeros(1, dtype=torch.long, device=device)
                losses.append(F.cross_entropy(logits, target))

            if losses:
                contrastive_loss = torch.stack(losses).mean()

        with torch.no_grad():
            self._enqueue_contrastive(
                keys=z_k,
                source_ids=source_ids,
                task_ids=task_ids,
                pc_type_ids=pc_type_ids,
                valid_mask=valid_mask)

        return contrastive_loss

    @staticmethod
    def _boxes_to_point_masks(boxes, token_coords, eps=0.01):
        lower = boxes[:, :3].unsqueeze(1)
        upper = boxes[:, 3:].unsqueeze(1)
        coords = token_coords.unsqueeze(0)
        masks = ((coords >= lower - eps) & (coords <= upper + eps)).all(dim=-1)
        return masks.float()

    @staticmethod
    def _dice_loss_with_logits(logits, targets, eps=1e-6):
        probs = torch.sigmoid(logits)
        numerator = 2.0 * (probs * targets).sum(dim=-1)
        denominator = probs.sum(dim=-1) + targets.sum(dim=-1)
        return 1.0 - (numerator + eps) / (denominator + eps)

    @staticmethod
    def _greedy_match(cost_matrix):
        cost = cost_matrix.detach().cpu()
        n_pred, n_gt = cost.shape
        used_pred = set()
        used_gt = set()
        pairs = []
        for _ in range(min(n_pred, n_gt)):
            best_pair = None
            best_value = None
            for pred_idx in range(n_pred):
                if pred_idx in used_pred:
                    continue
                for gt_idx in range(n_gt):
                    if gt_idx in used_gt:
                        continue
                    value = float(cost[pred_idx, gt_idx].item())
                    if best_value is None or value < best_value:
                        best_value = value
                        best_pair = (pred_idx, gt_idx)
            if best_pair is None:
                break
            used_pred.add(best_pair[0])
            used_gt.add(best_pair[1])
            pairs.append(best_pair)
        if not pairs:
            return [], []
        rows, cols = zip(*pairs)
        return list(rows), list(cols)

    def _match_boxes(self, cost_matrix):
        if linear_sum_assignment is None:
            return self._greedy_match(cost_matrix)
        rows, cols = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        return rows.tolist(), cols.tolist()

    def _find_bbox_spans(self, input_ids_1d):
        start_id = int(getattr(self, 'bbox_grounding_span_start_token_id', -1))
        end_id = int(getattr(self, 'bbox_grounding_span_end_token_id', -1))
        if start_id < 0 or end_id < 0:
            return []
        starts = (input_ids_1d == start_id).nonzero(as_tuple=False).squeeze(-1)
        spans = []
        for start_tensor in starts:
            start = int(start_tensor.item())
            if start + 1 >= int(input_ids_1d.numel()):
                continue
            end_offsets = (input_ids_1d[start + 1:] == end_id).nonzero(as_tuple=False).squeeze(-1)
            if end_offsets.numel() == 0:
                continue
            end = start + 1 + int(end_offsets[0].item())
            spans.append((start, end))
        return spans

    def _pool_bbox_span_hidden(self, hidden_states_1d, start, end):
        if end > start + 1:
            return hidden_states_1d[start + 1:end].mean(dim=0)
        return hidden_states_1d[end]

    def _get_bbox_query_hidden(self, hidden_states_1d, input_ids_1d):
        spans = self._find_bbox_spans(input_ids_1d)
        if len(spans) > 0:
            hidden = torch.stack([
                self._pool_bbox_span_hidden(hidden_states_1d, start, end)
                for start, end in spans
            ], dim=0)
            positions = torch.as_tensor(
                [start for start, _ in spans],
                dtype=torch.long,
                device=hidden_states_1d.device)
            return hidden, positions

        box_token_id = int(getattr(self, 'bbox_grounding_box_token_id', -1))
        if box_token_id < 0:
            return None, None
        positions = (input_ids_1d == box_token_id).nonzero(as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            return None, None
        return hidden_states_1d[positions], positions

    def _compute_bbox_grounding_loss(
        self,
        hidden_states,
        input_ids,
        point_embeds,
        token_coords,
        gt_boxes,
        gt_box_mask,
        sample_mask):
        if not self.bbox_grounding_enabled or self.bbox_grounding_head is None:
            return None, None, None
        if input_ids is None or point_embeds is None or token_coords is None:
            return None, None, None
        if gt_boxes is None or gt_box_mask is None or sample_mask is None:
            return None, None, None

        device = hidden_states.device
        gt_boxes = gt_boxes.to(device=device, dtype=torch.float32)
        gt_box_mask = gt_box_mask.to(device=device, dtype=torch.bool)
        sample_mask = sample_mask.to(device=device, dtype=torch.bool)
        token_coords = token_coords.to(device=device, dtype=torch.float32)
        point_embeds = point_embeds.to(device=device)

        batch_size = int(hidden_states.size(0))
        box_losses = []
        mask_losses = []
        for batch_idx in range(batch_size):
            if not bool(sample_mask[batch_idx].item()):
                continue
            box_hidden, _ = self._get_bbox_query_hidden(
                hidden_states[batch_idx],
                input_ids[batch_idx])
            gt_indices = gt_box_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
            if box_hidden is None or gt_indices.numel() == 0:
                continue

            pred_boxes, pred_mask_logits = self.bbox_grounding_head(
                box_hidden.unsqueeze(0),
                point_embeds[batch_idx:batch_idx + 1])
            pred_boxes = pred_boxes[0]
            pred_mask_logits = pred_mask_logits[0]

            target_boxes = gt_boxes[batch_idx, gt_indices]
            target_masks = self._boxes_to_point_masks(
                target_boxes,
                token_coords[batch_idx],
                eps=self.bbox_grounding_mask_eps)

            box_cost = torch.cdist(pred_boxes.float(), target_boxes.float(), p=1) / 6.0
            mask_cost = F.binary_cross_entropy_with_logits(
                pred_mask_logits[:, None, :].expand(-1, target_masks.size(0), -1),
                target_masks[None, :, :].expand(pred_mask_logits.size(0), -1, -1),
                reduction='none').mean(dim=-1)
            cost = box_cost + self.bbox_grounding_match_mask_weight * mask_cost
            pred_match, gt_match = self._match_boxes(cost)
            if len(pred_match) == 0:
                continue

            pred_match = torch.as_tensor(pred_match, dtype=torch.long, device=device)
            gt_match = torch.as_tensor(gt_match, dtype=torch.long, device=device)
            matched_pred_boxes = pred_boxes[pred_match]
            matched_target_boxes = target_boxes[gt_match]
            matched_pred_masks = pred_mask_logits[pred_match]
            matched_target_masks = target_masks[gt_match]

            box_losses.append(F.smooth_l1_loss(
                matched_pred_boxes,
                matched_target_boxes,
                reduction='mean'))
            bce = F.binary_cross_entropy_with_logits(
                matched_pred_masks,
                matched_target_masks,
                reduction='mean')
            dice = self._dice_loss_with_logits(
                matched_pred_masks,
                matched_target_masks).mean()
            mask_losses.append(bce + dice)

        if not box_losses:
            return None, None, None
        box_loss = torch.stack(box_losses).mean()
        mask_loss = torch.stack(mask_losses).mean() if mask_losses else box_loss.new_zeros(())
        total = (
            self.bbox_grounding_lambda_box * box_loss
            + self.bbox_grounding_lambda_mask * mask_loss)
        return total, box_loss, mask_loss

    def _predict_bbox_grounding_outputs(self, hidden_states, input_ids, point_embeds):
        if not self.bbox_grounding_enabled or self.bbox_grounding_head is None:
            return None
        if hidden_states is None or input_ids is None or point_embeds is None:
            return None

        predictions = []
        raw_predictions = [] if bool(getattr(self, 'enable_bbox_activation_vis', False)) else None
        for batch_idx in range(int(hidden_states.size(0))):
            box_hidden, positions = self._get_bbox_query_hidden(
                hidden_states[batch_idx],
                input_ids[batch_idx])
            if box_hidden is None:
                predictions.append({
                    'boxes': hidden_states.new_zeros((0, 6), dtype=torch.float32),
                    'mask_logits': hidden_states.new_zeros((0, self.n_point_tokens), dtype=torch.float32),
                    'box_positions': hidden_states.new_zeros((0,), dtype=torch.long),
                })
                if raw_predictions is not None:
                    raw_predictions.append({
                        'boxes': hidden_states.new_zeros((0, 6), dtype=torch.float32),
                        'mask_logits': hidden_states.new_zeros((0, self.n_point_tokens), dtype=torch.float32),
                        'box_positions': hidden_states.new_zeros((0,), dtype=torch.long),
                    })
                continue
            pred_boxes, pred_mask_logits = self.bbox_grounding_head(
                box_hidden.unsqueeze(0),
                point_embeds[batch_idx:batch_idx + 1])
            predictions.append({
                'boxes': pred_boxes[0].detach(),
                'mask_logits': pred_mask_logits[0].detach(),
                'box_positions': positions.detach(),
            })
            if raw_predictions is not None:
                raw_predictions.append({
                    'boxes': pred_boxes[0],
                    'mask_logits': pred_mask_logits[0],
                    'box_positions': positions,
                })
        if raw_predictions is not None:
            self.latest_bbox_grounding_predictions_raw = raw_predictions
        return predictions

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        point_clouds=None,
        is_pc=None,
        is_img=None,
        pc_type_ids=None,
        source_ids=None,
        task_ids=None,
        part_point_indices=None,
        n_parts=None,
        is_step_pc=None,
        utonia_data_dict=None,
        bbox_grounding_boxes=None,
        bbox_grounding_box_mask=None,
        bbox_grounding_sample_mask=None,
        bbox_grounding_infer=False):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.latest_ce_loss = None
        self.latest_contrastive_loss = None
        self.latest_weighted_contrastive_loss = None
        self.latest_bbox_grounding_loss = None
        self.latest_bbox_grounding_box_loss = None
        self.latest_bbox_grounding_mask_loss = None
        self.latest_bbox_grounding_predictions = None
        self.latest_bbox_grounding_predictions_raw = None
        self.latest_total_loss = None
        self.latest_point_embeds = None
        self.latest_point_token_xyz = None
        self.latest_point_token_input_positions = None

        point_embeds_for_contrastive = None
        point_embeds_for_grounding = None
        token_coords_for_grounding = None
        has_bbox_grounding_targets = (
            bbox_grounding_boxes is not None
            and bbox_grounding_box_mask is not None
            and bbox_grounding_sample_mask is not None)
        need_bbox_grounding = (
            self.bbox_grounding_enabled
            and (has_bbox_grounding_targets or bool(bbox_grounding_infer)))
        need_point_activation_vis = bool(getattr(self, 'enable_point_activation_vis', False))
        need_point_token_coords = bool(need_bbox_grounding or need_point_activation_vis)

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if is_img.sum() > 0 and pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos[is_img]
                pixel_values_videos = pixel_values_videos.view(-1, pixel_values_videos.shape[-1])
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_grid_thw = video_grid_thw[is_img]
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # add point cloud embeddings
            if past_key_values is None or past_key_values.get_seq_length() == 0:
                if self.two_stage and is_step_pc is not None and is_step_pc.sum() > 0:
                    if self.pc_encoder_type == 'utonia':
                        encoded = self.point_encoder.encode_global_and_part_tokens(
                            point_clouds,
                            part_point_indices=part_point_indices,
                            utonia_data_dict=utonia_data_dict,
                            return_token_coords=need_point_token_coords)
                    else:
                        encoded = self.point_encoder.encode_global_and_part_tokens(
                            point_clouds,
                            part_point_indices=part_point_indices,
                            return_token_coords=need_point_token_coords)
                    if need_point_token_coords:
                        point_embeds, part_embeds, token_coords_for_grounding = encoded
                    else:
                        point_embeds, part_embeds = encoded
                else:
                    if self.pc_encoder_type == 'utonia':
                        if need_point_token_coords:
                            point_embeds, _, token_coords_for_grounding = \
                                self.point_encoder.encode_global_and_part_tokens(
                                    point_clouds,
                                    part_point_indices=None,
                                    utonia_data_dict=utonia_data_dict,
                                    return_token_coords=True)
                        else:
                            point_embeds = self.point_encoder(
                                point_clouds,
                                utonia_data_dict=utonia_data_dict)
                    else:
                        if need_point_token_coords:
                            point_embeds, _, token_coords_for_grounding = \
                                self.point_encoder.encode_global_and_part_tokens(
                                    point_clouds,
                                    part_point_indices=None,
                                    return_token_coords=True)
                        else:
                            point_embeds = self.point_encoder(point_clouds)
                    part_embeds = None
                point_embeds_for_contrastive = point_embeds
                if need_bbox_grounding:
                    point_embeds_for_grounding = point_embeds
                if need_point_activation_vis:
                    if point_embeds.requires_grad:
                        point_embeds.retain_grad()
                    self.latest_point_embeds = point_embeds
                    self.latest_point_token_xyz = token_coords_for_grounding

                start_idxs = (attention_mask.shape[1] - attention_mask.sum(dim=1)).tolist()
                step_mask = is_step_pc
                if step_mask is None:
                    step_mask = torch.zeros(
                        point_embeds.size(0),
                        dtype=torch.bool,
                        device=point_embeds.device)
                if pc_type_ids is None:
                    pc_type_ids = torch.zeros(
                        point_embeds.size(0),
                        dtype=torch.long,
                        device=point_embeds.device)
                else:
                    pc_type_ids = pc_type_ids.to(
                        device=point_embeds.device,
                        dtype=torch.long).clamp(0, 1)
                model_pc_start = self.model_pc_start_embedding.to(inputs_embeds.dtype)
                model_pc_end = self.model_pc_end_embedding.to(inputs_embeds.dtype)
                pc_type_embeds = self.model_pc_type_embedding(pc_type_ids).to(inputs_embeds.dtype)
                point_token_positions = []

                for i, start in enumerate(start_idxs):
                    pc_type_token = pc_type_embeds[i].unsqueeze(0)
                    is_step_sample = bool(step_mask[i].item()) if torch.is_tensor(step_mask[i]) else bool(step_mask[i])
                    if self.two_stage and is_step_sample and part_embeds is not None:
                        n_part = int(n_parts[i].item()) if n_parts is not None else 0
                        pieces = [
                            pc_type_token,
                            model_pc_start.unsqueeze(0),
                            point_embeds[i],
                            model_pc_end.unsqueeze(0),
                        ]
                        for p in range(n_part):
                            pieces.append(part_embeds[i, p])
                            if p < n_part - 1:
                                pieces.append(model_pc_end.unsqueeze(0))
                        seq = torch.cat(pieces, dim=0)
                        inputs_embeds[i, start:start + seq.shape[0], :] = seq
                        point_positions_i = torch.arange(
                            start + 2,
                            start + 2 + point_embeds.size(1),
                            device=inputs_embeds.device,
                            dtype=torch.long,
                        )
                    else:
                        seq = torch.cat([pc_type_token, point_embeds[i]], dim=0)
                        inputs_embeds[i, start:start + seq.shape[0], :] = seq
                        point_positions_i = torch.arange(
                            start + 1,
                            start + 1 + point_embeds.size(1),
                            device=inputs_embeds.device,
                            dtype=torch.long,
                        )
                    point_token_positions.append(point_positions_i)
                if need_point_activation_vis and point_token_positions:
                    self.latest_point_token_input_positions = torch.stack(point_token_positions, dim=0)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        ce_loss = None
        contrastive_loss = None
        bbox_grounding_loss = None
        bbox_grounding_box_loss = None
        bbox_grounding_mask_loss = None
        weighted_ctr = None
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ce_loss = loss_fct(shift_logits, shift_labels)
            loss = ce_loss

        if self.contrastive_enabled and self.training and point_embeds_for_contrastive is not None:
            self.contrastive_forward_counter += 1
            current_forward = int(self.contrastive_forward_counter.item())
            compute_contrastive = ((current_forward - 1) % self.contrastive_interval) == 0
            contrastive_loss = self._compute_contrastive_loss(
                point_embeds=point_embeds_for_contrastive,
                pc_type_ids=pc_type_ids,
                source_ids=source_ids,
                task_ids=task_ids,
                is_pc=is_pc,
                compute_loss=compute_contrastive)
            if contrastive_loss is not None:
                weighted_ctr = self.contrastive_lambda * contrastive_loss
                loss = weighted_ctr if loss is None else (loss + weighted_ctr)

        if need_bbox_grounding and bool(bbox_grounding_infer):
            self.latest_bbox_grounding_predictions = self._predict_bbox_grounding_outputs(
                hidden_states=hidden_states,
                input_ids=input_ids,
                point_embeds=point_embeds_for_grounding)

        if need_bbox_grounding and has_bbox_grounding_targets:
            (
                bbox_grounding_loss,
                bbox_grounding_box_loss,
                bbox_grounding_mask_loss,
            ) = self._compute_bbox_grounding_loss(
                hidden_states=hidden_states,
                input_ids=input_ids,
                point_embeds=point_embeds_for_grounding,
                token_coords=token_coords_for_grounding,
                gt_boxes=bbox_grounding_boxes,
                gt_box_mask=bbox_grounding_box_mask,
                sample_mask=bbox_grounding_sample_mask)
            if bbox_grounding_loss is not None:
                weighted_bbox_grounding = self.bbox_grounding_lambda * bbox_grounding_loss
                loss = weighted_bbox_grounding if loss is None else (loss + weighted_bbox_grounding)

        if ce_loss is not None:
            self.latest_ce_loss = ce_loss.detach()
        if contrastive_loss is not None:
            self.latest_contrastive_loss = contrastive_loss.detach()
        if weighted_ctr is not None:
            self.latest_weighted_contrastive_loss = weighted_ctr.detach()
        if bbox_grounding_loss is not None:
            self.latest_bbox_grounding_loss = bbox_grounding_loss.detach()
        if bbox_grounding_box_loss is not None:
            self.latest_bbox_grounding_box_loss = bbox_grounding_box_loss.detach()
        if bbox_grounding_mask_loss is not None:
            self.latest_bbox_grounding_mask_loss = bbox_grounding_mask_loss.detach()
        if loss is not None:
            self.latest_total_loss = loss.detach()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        target_device = kwargs.get('device', None)
        if target_device is None:
            for arg in args:
                if isinstance(arg, torch.device):
                    target_device = arg
                    break
                if torch.is_tensor(arg):
                    target_device = arg.device
                    break
        if target_device is None:
            try:
                target_device = next(self.model.parameters()).device
            except StopIteration:
                target_device = torch.device('cpu')

        self.point_encoder.to(device=target_device)

        return model

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs['point_clouds'] = kwargs['point_clouds']
        model_inputs['is_pc'] = kwargs['is_pc']
        model_inputs['is_img'] = kwargs['is_img']
        model_inputs['pc_type_ids'] = kwargs.get('pc_type_ids', None)
        model_inputs['source_ids'] = kwargs.get('source_ids', None)
        model_inputs['task_ids'] = kwargs.get('task_ids', None)
        model_inputs['part_point_indices'] = kwargs.get('part_point_indices', None)
        model_inputs['n_parts'] = kwargs.get('n_parts', None)
        model_inputs['is_step_pc'] = kwargs.get('is_step_pc', None)
        model_inputs['utonia_data_dict'] = kwargs.get('utonia_data_dict', None)
        model_inputs['bbox_grounding_boxes'] = kwargs.get('bbox_grounding_boxes', None)
        model_inputs['bbox_grounding_box_mask'] = kwargs.get('bbox_grounding_box_mask', None)
        model_inputs['bbox_grounding_sample_mask'] = kwargs.get('bbox_grounding_sample_mask', None)
        model_inputs['bbox_grounding_infer'] = kwargs.get('bbox_grounding_infer', False)
        return model_inputs

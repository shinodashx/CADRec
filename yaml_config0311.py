import copy
import os
from datetime import datetime
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    'common': {
        'model': {
            'base_model_path': 'Qwen/Qwen2-VL-2B-Instruct',
            'pc_encoder_type': 'pointbert',
        },
        'point_cloud': {
            'n_points': 8192,
            'n_point_tokens': 512,
            'part_points_per_bbox': 64,
        },
        'utonia': {
            'scale': 2.0,
            'normalize_coord': True,
            'use_normal': True,
        },
    },
    'train': {
        'paths': {
            'data_path': '/data/songhx24/Dataset',
            'log_path': './work_dirs_utonia0319',
            'resume': None,
            'model_path': None,
        },
        'task': {
            'mode': 'pc',
            'use_text': False,
            'multi_task': False,
            'balanced': False,
            'deepcad': False,
            'two_stage': False,
        },
        'dataset': {
            'name': 'auto',
            'train_split': 'train',
            'eval_split': 'val',
            'deepcad_root_subdir': 'data/cad_cadqueryV1.2',
            'deepcadv2_root_subdir': 'data/cad_cadqueryV2',
            'cadrecode_v15our_root_subdir': '/nas1/songhx24/Dataset/CAARECODE/cad-recode-v1.5-process-new',
            'deepcad_split_json': 'train_val_test_split_V1.2.json',
            'cad_recode_subdir': 'cad-recode-v1.5',
            'text2cad_subdir': 'text2cad',
            'normalize_std_pc': 100.0,
            'train_noise_scale_pc': 0.01,
            'eval_noise_scale_pc': None,
            'img_size': 128,
            'normalize_std_img': 200.0,
            'noise_scale_img': -1,
            'num_imgs': 4,
        },
        'dataloader': {
            'batch_size': None,
            'gradient_accumulation_steps': None,
            'num_workers': 8,
        },
        'trainer': {
            'dataloader_num_workers': 8,
            'max_steps': 120000,
            'learning_rate': 2e-4,
            'warmup_steps': 1000,
            'weight_decay': 0.01,
            'logging_steps': 100,
            'save_total_limit': 20,
            'save_steps': 10000,
            'eval_steps': 5000,
            'per_device_train_batch_size': None,
            'gradient_accumulation_steps': None,
        },
        'contrastive': {
            'enabled': False,
            'lambda': 0.05,
            'temperature': 0.07,
            'proj_dim': 128,
            'queue_size': 4096,
            'momentum': 0.999,
            'interval': 1,
            'max_negatives': 1024,
        },
    },
    'runtime': {
        'save_config_name': 'run_config.yaml',
        'save_timestamped_copy': True,
    },
}


def get_default_config():
    return copy.deepcopy(DEFAULT_CONFIG)


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Config file must contain a YAML mapping: {config_path}')
    return data


def load_config_with_defaults(config_path=None):
    config = get_default_config()
    if config_path is None:
        return config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    user_config = load_yaml_config(config_path)
    return deep_update(config, user_config)


def dump_yaml_config(config, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def get_nested(config, keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def resolve_log_dir_from_checkpoint(checkpoint_path):
    checkpoint = Path(checkpoint_path)
    if checkpoint.name.startswith('checkpoint-'):
        return checkpoint.parent
    return checkpoint


def find_latest_saved_config(log_dir, save_config_name='run_config.yaml'):
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None

    exact = log_dir / save_config_name
    if exact.exists():
        return exact

    stem = Path(save_config_name).stem
    suffix = Path(save_config_name).suffix or '.yaml'
    timestamped = sorted(
        log_dir.glob(f'{stem}_*{suffix}'),
        key=lambda p: p.stat().st_mtime,
        reverse=True)
    if timestamped:
        return timestamped[0]

    any_yaml = sorted(
        list(log_dir.glob('*.yaml')) + list(log_dir.glob('*.yml')),
        key=lambda p: p.stat().st_mtime,
        reverse=True)
    if any_yaml:
        return any_yaml[0]
    return None


def save_training_config_snapshot(
    config,
    log_dir,
    save_config_name='run_config.yaml',
    save_timestamped_copy=True,
    source_config_path=None):
    os.makedirs(log_dir, exist_ok=True)
    snapshot = copy.deepcopy(config)
    snapshot.setdefault('meta', {})
    snapshot['meta']['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if source_config_path is not None:
        snapshot['meta']['source_config'] = str(Path(source_config_path).resolve())

    save_path = Path(log_dir) / save_config_name
    dump_yaml_config(snapshot, save_path)

    timestamped_path = None
    if save_timestamped_copy:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_name = f'{save_path.stem}_{timestamp}{save_path.suffix}'
        timestamped_path = Path(log_dir) / timestamped_name
        dump_yaml_config(snapshot, timestamped_path)
    return save_path, timestamped_path

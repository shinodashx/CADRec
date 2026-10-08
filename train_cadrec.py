import copy
import os
from functools import partial
from argparse import ArgumentParser
from pathlib import Path

import torch
import contextlib
from torch.utils.data import ConcatDataset
from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback

from cadrec import (
    BBOX_GROUNDING_SPECIAL_TOKENS,
    Cadrille,
    collate,
    _build_utonia_data_dict_from_tensors,
)
from cadrec_dataset import (
    Text2CADDataset,
    CadRecodeDataset,
    DeepCADDataset,
    DeepCADMultiTaskDataset,
    DeepCADV2MultiTaskDataset,
    DeepCADV22MultiTaskDataset,
    CadRecodeV15OURDataset,
    DeepCADTwoStageDataset,
    BalancedMultiTaskSampler,
    HybridBalancedContrastiveSampler)
from yaml_config import (
    load_config_with_defaults,
    get_nested,
    save_training_config_snapshot,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if text in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise ValueError(f'Invalid boolean value: {value}')


def _resolve_project_path(path_value):
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parent / path).resolve())


def _resolve_data_subpath(data_path, path_value):
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(Path(data_path) / path)


def _normalize_train_dataset_name(dataset_name):
    normalized = str(dataset_name or 'auto').strip().lower()
    aliases = {
        'auto': 'auto',
        'cadrecode': 'cadrecode',
        'cad_recode': 'cadrecode',
        'deepcad': 'deepcad',
        'deepcad_multitask': 'deepcad_multitask',
        'deepcad-multitask': 'deepcad_multitask',
        'deepcadv2_multitask': 'deepcadv2_multitask',
        'deepcadv2-multitask': 'deepcadv2_multitask',
        'deepcad_v2_multitask': 'deepcadv2_multitask',
        'deepcad-v2-multitask': 'deepcadv2_multitask',
        'deepcadv22_multitask': 'deepcadv22_multitask',
        'deepcadv22-multitask': 'deepcadv22_multitask',
        'deepcad_v22_multitask': 'deepcadv22_multitask',
        'deepcad-v22-multitask': 'deepcadv22_multitask',
        'cadrecode_v15our': 'cadrecode_v15our',
        'cadrecode-v15our': 'cadrecode_v15our',
        'cadrecode_v15our_multitask': 'cadrecode_v15our',
        'cadrecode-v15our-multitask': 'cadrecode_v15our',
        'deepcadv2.2_multitask': 'deepcadv22_multitask',
        'deepcadv2.2-multitask': 'deepcadv22_multitask',
        'deepcad_v2.2_multitask': 'deepcadv22_multitask',
        'deepcad-v2.2-multitask': 'deepcadv22_multitask',
        'deepcad_two_stage': 'deepcad_two_stage',
        'deepcad-two-stage': 'deepcad_two_stage',
    }
    if normalized not in aliases:
        raise ValueError(
            f'Unsupported train.dataset.name: {dataset_name}. '
            "Expected one of ['auto', 'cadrecode', 'deepcad', 'deepcad_multitask', 'deepcadv2_multitask', 'deepcadv22_multitask', 'cadrecode_v15our', 'deepcad_two_stage'].")
    return aliases[normalized]


def _resolve_train_task_flags(dataset_name, deepcad, multi_task, two_stage):
    normalized_name = _normalize_train_dataset_name(dataset_name)
    if normalized_name == 'auto':
        if two_stage:
            normalized_name = 'deepcad_two_stage'
        elif deepcad and multi_task:
            normalized_name = 'deepcad_multitask'
        elif deepcad:
            normalized_name = 'deepcad'
        else:
            normalized_name = 'cadrecode'
        return {
            'dataset_name': normalized_name,
            'deepcad': bool(deepcad),
            'multi_task': bool(multi_task),
            'two_stage': bool(two_stage),
        }

    expected_flags = {
        'cadrecode': {'deepcad': False, 'multi_task': False, 'two_stage': False},
        'deepcad': {'deepcad': True, 'multi_task': False, 'two_stage': False},
        'deepcad_multitask': {'deepcad': True, 'multi_task': True, 'two_stage': False},
        'deepcadv2_multitask': {'deepcad': True, 'multi_task': True, 'two_stage': False},
        'deepcadv22_multitask': {'deepcad': True, 'multi_task': True, 'two_stage': False},
        'cadrecode_v15our': {'deepcad': True, 'multi_task': True, 'two_stage': False},
        'deepcad_two_stage': {'deepcad': True, 'multi_task': False, 'two_stage': True},
    }[normalized_name]
    actual_flags = {
        'deepcad': bool(deepcad),
        'multi_task': bool(multi_task),
        'two_stage': bool(two_stage),
    }
    if actual_flags != expected_flags:
        raise ValueError(
            f'train.dataset.name={normalized_name} conflicts with train.task flags: '
            f'{actual_flags}. Expected {expected_flags}.')
    return {
        'dataset_name': normalized_name,
        **expected_flags,
    }


def resolve_training_setup(
    data_path,
    mode,
    use_text,
    multi_task,
    balanced,
    deepcad,
    two_stage,
    n_points,
    part_points_per_bbox,
    pc_encoder_type='pointbert',
    utonia_use_normal=True,
    dataset_config=None,
    dataloader_config=None,
    trainer_overrides=None):
    dataset_config = dataset_config or {}
    dataloader_config = dataloader_config or {}
    trainer_overrides = trainer_overrides or {}

    task_flags = _resolve_train_task_flags(
        dataset_name=dataset_config.get('name', 'auto'),
        deepcad=deepcad,
        multi_task=multi_task,
        two_stage=two_stage)
    deepcad = task_flags['deepcad']
    multi_task = task_flags['multi_task']
    two_stage = task_flags['two_stage']

    use_point_normals = bool(str(pc_encoder_type).lower() == 'utonia' and utonia_use_normal)

    if two_stage:
        if not deepcad:
            raise ValueError('Two-stage training currently supports DeepCAD only. Set train.task.deepcad=true.')
        if mode != 'pc':
            raise ValueError('Two-stage training requires train.task.mode=pc.')
        if use_text:
            raise ValueError('Two-stage training does not support train.task.use_text=true.')
        if balanced:
            raise ValueError('Two-stage training does not support train.task.balanced=true.')
        if multi_task:
            raise ValueError('Two-stage training uses its own paired bbox/step format; keep train.task.multi_task=false.')

    if multi_task and not deepcad:
        raise ValueError(
            'CadRecode multi-task branch has been removed. '
            'For multi-task training, set train.task.deepcad=true.')

    if balanced:
        if not multi_task:
            raise ValueError('Balanced sampling is only available when train.task.multi_task=true.')
        if use_text:
            raise ValueError('Balanced sampling is not supported together with train.task.use_text=true.')
    if task_flags['dataset_name'] in ('deepcadv2_multitask', 'deepcadv22_multitask', 'cadrecode_v15our') and mode != 'pc':
        raise ValueError('DeepCAD V2/V2.2 and CadRecode V15 OUR multitask currently support train.task.mode=pc only.')

    normalize_std_pc = float(dataset_config.get('normalize_std_pc', 100.0))
    train_noise_scale_pc = dataset_config.get('train_noise_scale_pc', 0.01)
    if train_noise_scale_pc is not None:
        train_noise_scale_pc = float(train_noise_scale_pc)
    eval_noise_scale_pc = dataset_config.get('eval_noise_scale_pc', None)
    if eval_noise_scale_pc is not None:
        eval_noise_scale_pc = float(eval_noise_scale_pc)
    img_size = int(dataset_config.get('img_size', 128))
    normalize_std_img = float(dataset_config.get('normalize_std_img', 200.0))
    noise_scale_img = dataset_config.get('noise_scale_img', -1)
    if noise_scale_img is not None:
        noise_scale_img = float(noise_scale_img)
    num_imgs = int(dataset_config.get('num_imgs', 4))
    train_split = str(dataset_config.get('train_split', 'train'))
    eval_split = str(dataset_config.get('eval_split', 'val'))

    batch_size_override = dataloader_config.get(
        'batch_size',
        trainer_overrides.get('per_device_train_batch_size', None))
    if batch_size_override is not None:
        batch_size_override = int(batch_size_override)
        if batch_size_override <= 0:
            raise ValueError('train.dataloader.batch_size must be a positive integer.')

    accumulation_override = dataloader_config.get(
        'gradient_accumulation_steps',
        trainer_overrides.get('gradient_accumulation_steps', None))
    if accumulation_override is not None:
        accumulation_override = int(accumulation_override)
        if accumulation_override <= 0:
            raise ValueError('train.dataloader.gradient_accumulation_steps must be a positive integer.')

    dataloader_num_workers = int(
        dataloader_config.get(
            'num_workers',
            trainer_overrides.get('dataloader_num_workers', 8)))
    if dataloader_num_workers < 0:
        raise ValueError('train.dataloader.num_workers must be >= 0.')

    if deepcad:
        split_json_path = _resolve_project_path(
            dataset_config.get('deepcad_split_json', 'train_val_test_split_V1.2.json'))
        if task_flags['dataset_name'] == 'deepcadv2_multitask':
            dataset_cls = DeepCADV2MultiTaskDataset
            dataset_name = 'deepcadv2_multitask'
            deepcad_root = _resolve_data_subpath(
                data_path,
                dataset_config.get('deepcadv2_root_subdir', 'data/cad_cadqueryV2'))
        elif task_flags['dataset_name'] == 'deepcadv22_multitask':
            dataset_cls = DeepCADV22MultiTaskDataset
            dataset_name = 'deepcadv22_multitask'
            deepcad_root = _resolve_data_subpath(
                data_path,
                dataset_config.get('deepcadv2_root_subdir', 'data/cad_cadqueryV2'))
        elif task_flags['dataset_name'] == 'cadrecode_v15our':
            dataset_cls = CadRecodeV15OURDataset
            dataset_name = 'cadrecode_v15our'
            deepcad_root = _resolve_data_subpath(
                data_path,
                dataset_config.get(
                    'cadrecode_v15our_root_subdir',
                    '/nas1/songhx24/Dataset/CAARECODE/cad-recode-v1.5-process-new'))
            split_json_path = None
        else:
            deepcad_root = _resolve_data_subpath(
                data_path,
                dataset_config.get('deepcad_root_subdir', 'data/cad_cadqueryV1.2'))
            dataset_cls = None

        if two_stage:
            dataset_cls = DeepCADTwoStageDataset
            dataset_name = 'deepcad_two_stage'
        elif dataset_cls is not None:
            pass
        elif multi_task:
            dataset_cls = DeepCADMultiTaskDataset
            dataset_name = 'deepcad_multitask'
        else:
            dataset_cls = DeepCADDataset
            dataset_name = 'deepcad'

        train_dataset_kwargs = dict(
            root_dir=deepcad_root,
            split_json_path=split_json_path,
            split=train_split,
            n_points=int(n_points),
            normalize_std_pc=normalize_std_pc,
            noise_scale_pc=train_noise_scale_pc,
            img_size=img_size,
            normalize_std_img=normalize_std_img,
            noise_scale_img=noise_scale_img,
            num_imgs=num_imgs,
            mode=mode,
            use_point_normals=use_point_normals)
        eval_dataset_kwargs = dict(
            root_dir=deepcad_root,
            split_json_path=split_json_path,
            split=eval_split,
            n_points=int(n_points),
            normalize_std_pc=normalize_std_pc,
            noise_scale_pc=eval_noise_scale_pc,
            img_size=img_size,
            normalize_std_img=normalize_std_img,
            noise_scale_img=noise_scale_img,
            num_imgs=num_imgs,
            mode=mode,
            use_point_normals=use_point_normals)
        if two_stage:
            train_dataset_kwargs['n_points_per_part'] = int(part_points_per_bbox)
            eval_dataset_kwargs['n_points_per_part'] = int(part_points_per_bbox)
            default_batch_size = 2
            default_accumulation_steps = 8
        else:
            default_batch_size = 4
            default_accumulation_steps = 4
        text_dataset_kwargs = None
    else:
        dataset_cls = CadRecodeDataset
        dataset_name = 'cadrecode'
        cad_recode_path = _resolve_data_subpath(
            data_path,
            dataset_config.get('cad_recode_subdir', 'cad-recode-v1.5'))
        train_dataset_kwargs = dict(
            root_dir=cad_recode_path,
            split=train_split,
            n_points=int(n_points),
            normalize_std_pc=normalize_std_pc,
            noise_scale_pc=train_noise_scale_pc,
            img_size=img_size,
            normalize_std_img=normalize_std_img,
            noise_scale_img=noise_scale_img,
            num_imgs=num_imgs,
            mode=mode,
            use_point_normals=use_point_normals)
        eval_dataset_kwargs = dict(
            root_dir=cad_recode_path,
            split=eval_split,
            n_points=int(n_points),
            normalize_std_pc=normalize_std_pc,
            noise_scale_pc=eval_noise_scale_pc,
            img_size=img_size,
            normalize_std_img=normalize_std_img,
            noise_scale_img=noise_scale_img,
            num_imgs=num_imgs,
            mode=mode,
            use_point_normals=use_point_normals)
        default_batch_size = 8 if use_text else 4
        default_accumulation_steps = 4
        text_dataset_kwargs = None
        if use_text:
            text_dataset_kwargs = dict(
                root_dir=_resolve_data_subpath(
                    data_path,
                    dataset_config.get('text2cad_subdir', 'text2cad')),
                split=train_split)

    batch_size = batch_size_override if batch_size_override is not None else default_batch_size
    accumulation_steps = accumulation_override if accumulation_override is not None else default_accumulation_steps

    return {
        'task_flags': {
            'deepcad': deepcad,
            'multi_task': multi_task,
            'two_stage': two_stage,
        },
        'dataset_name': dataset_name,
        'train_dataset_cls': dataset_cls,
        'train_dataset_kwargs': train_dataset_kwargs,
        'eval_dataset_cls': dataset_cls,
        'eval_dataset_kwargs': eval_dataset_kwargs,
        'text_dataset_kwargs': text_dataset_kwargs,
        'batch_size': batch_size,
        'accumulation_steps': accumulation_steps,
        'dataloader_num_workers': dataloader_num_workers,
        'resolved_task': {
            'mode': mode,
            'use_text': bool(use_text),
            'balanced': bool(balanced),
            'deepcad': deepcad,
            'multi_task': multi_task,
            'two_stage': two_stage,
        },
        'resolved_dataset': {
            'name': dataset_name,
            'train_dataset_class': dataset_cls.__name__,
            'eval_dataset_class': dataset_cls.__name__,
            'train_dataset_kwargs': copy.deepcopy(train_dataset_kwargs),
            'eval_dataset_kwargs': copy.deepcopy(eval_dataset_kwargs),
            'text_dataset_kwargs': copy.deepcopy(text_dataset_kwargs),
            'use_point_normals': use_point_normals,
        },
        'resolved_dataloader': {
            'batch_size': batch_size,
            'gradient_accumulation_steps': accumulation_steps,
            'num_workers': dataloader_num_workers,
        },
    }


class PrintToFileCallback(TrainerCallback):
    def on_init_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            os.makedirs(args.logging_dir, exist_ok=True)

    def on_log(self, args, state, control, logs, **kwargs):
        if state.is_world_process_zero:
            with open(os.path.join(args.logging_dir, 'log.txt'), 'a') as f:
                f.write(str(logs) + '\n')


class BalancedTrainer(Trainer):
    def __init__(self, *args, balanced_sampling=False, contrastive_sampling=False, **kwargs):
        self.balanced_sampling = balanced_sampling
        self.contrastive_sampling = contrastive_sampling
        self._pending_aux_loss_logs = None
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self):
        if not self.balanced_sampling:
            return super()._get_train_sampler()
        if not isinstance(self.train_dataset, (DeepCADMultiTaskDataset, DeepCADV2MultiTaskDataset)):
            return super()._get_train_sampler()
        seed = self.args.seed + self.args.process_index
        if self.contrastive_sampling:
            return HybridBalancedContrastiveSampler(
                dataset=self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                seed=seed)
        return BalancedMultiTaskSampler(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            seed=seed)

    def _get_wrapped_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _get_utonia_encoder(self):
        model = self._get_wrapped_model()
        if getattr(model, 'pc_encoder_type', None) != 'utonia':
            return None
        return getattr(model, 'point_encoder', None)

    @staticmethod
    def _to_loss_scalar(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().item())
        return float(value)

    def _collect_model_loss_components(self):
        model = self._get_wrapped_model()
        ce_loss = self._to_loss_scalar(getattr(model, 'latest_ce_loss', None))
        ctr_loss = self._to_loss_scalar(getattr(model, 'latest_contrastive_loss', None))
        bbox_ground_loss = self._to_loss_scalar(getattr(model, 'latest_bbox_grounding_loss', None))
        bbox_box_loss = self._to_loss_scalar(getattr(model, 'latest_bbox_grounding_box_loss', None))
        bbox_mask_loss = self._to_loss_scalar(getattr(model, 'latest_bbox_grounding_mask_loss', None))
        total_loss = self._to_loss_scalar(getattr(model, 'latest_total_loss', None))
        return {
            'loss_ce': ce_loss,
            'loss_ctr': ctr_loss,
            'loss_bbox_ground': bbox_ground_loss,
            'loss_bbox_box': bbox_box_loss,
            'loss_bbox_mask': bbox_mask_loss,
            'loss_total': total_loss,
        }

    def _queue_aux_loss_logs(
        self,
        ce_loss=None,
        ctr_loss=None,
        bbox_ground_loss=None,
        bbox_box_loss=None,
        bbox_mask_loss=None,
        total_loss=None):
        logs = {}
        if ce_loss is not None:
            logs['loss_ce'] = ce_loss
        if ctr_loss is not None:
            logs['loss_ctr'] = ctr_loss
        if bbox_ground_loss is not None:
            logs['loss_bbox_ground'] = bbox_ground_loss
        if bbox_box_loss is not None:
            logs['loss_bbox_box'] = bbox_box_loss
        if bbox_mask_loss is not None:
            logs['loss_bbox_mask'] = bbox_mask_loss
        if total_loss is not None:
            logs['loss_total'] = total_loss
        self._pending_aux_loss_logs = logs if logs else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

        if model.training:
            components = self._collect_model_loss_components()
            self._queue_aux_loss_logs(
                ce_loss=components['loss_ce'],
                ctr_loss=components['loss_ctr'],
                bbox_ground_loss=components['loss_bbox_ground'],
                bbox_box_loss=components['loss_bbox_box'],
                bbox_mask_loss=components['loss_bbox_mask'],
                total_loss=components['loss_total'])

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs, *args, **kwargs):
        pending = self._pending_aux_loss_logs
        if pending:
            merged = dict(logs)
            merged.update(pending)
            logs = merged
            self._pending_aux_loss_logs = None
        return super().log(logs, *args, **kwargs)

    @staticmethod
    def _point_cloud_batches_match(lhs, rhs):
        return (
            torch.is_tensor(lhs)
            and torch.is_tensor(rhs)
            and lhs.shape == rhs.shape
            and torch.equal(lhs, rhs)
        )

    def _build_utonia_data_dict(self, point_clouds, encoder):
        if not torch.is_tensor(point_clouds):
            return None
        point_cloud_tensors = [sample.detach().cpu() for sample in point_clouds]
        return _build_utonia_data_dict_from_tensors(
            point_cloud_tensors=point_cloud_tensors,
            scale=encoder.scale,
            normalize_coord=encoder.normalize_coord,
            use_normal=encoder.use_normal)

    def _maybe_attach_utonia_data_dict(self, model_inputs, encoder):
        if not isinstance(model_inputs, dict):
            return
        if 'utonia_data_dict' in model_inputs or 'point_clouds' not in model_inputs:
            return
        data_dict = self._build_utonia_data_dict(model_inputs['point_clouds'], encoder)
        if data_dict is not None:
            model_inputs['utonia_data_dict'] = data_dict

    def _prepare_inputs(self, inputs):
        encoder = self._get_utonia_encoder()
        if encoder is None:
            return super()._prepare_inputs(inputs)

        if 'bbox_inputs' in inputs and 'step_inputs' in inputs:
            bbox_inputs = inputs['bbox_inputs']
            step_inputs = inputs['step_inputs']
            bbox_points = bbox_inputs.get('point_clouds', None)
            step_points = step_inputs.get('point_clouds', None)
            if self._point_cloud_batches_match(bbox_points, step_points):
                shared_data_dict = self._build_utonia_data_dict(bbox_points, encoder)
                if shared_data_dict is not None:
                    bbox_inputs.setdefault('utonia_data_dict', shared_data_dict)
                    step_inputs.setdefault('utonia_data_dict', shared_data_dict)
            else:
                self._maybe_attach_utonia_data_dict(bbox_inputs, encoder)
                self._maybe_attach_utonia_data_dict(step_inputs, encoder)
        else:
            self._maybe_attach_utonia_data_dict(inputs, encoder)
        return super()._prepare_inputs(inputs)


class TwoStageTrainer(BalancedTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if 'bbox_inputs' not in inputs or 'step_inputs' not in inputs:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch)

        bbox_outputs = model(**inputs['bbox_inputs'])
        bbox_components = self._collect_model_loss_components() if model.training else None

        step_outputs = model(**inputs['step_inputs'])
        step_components = self._collect_model_loss_components() if model.training else None
        loss = bbox_outputs.loss + step_outputs.loss

        if model.training:
            ce_losses = []
            ctr_losses = []
            bbox_ground_losses = []
            bbox_box_losses = []
            bbox_mask_losses = []
            total_losses = []
            for components in (bbox_components, step_components):
                if components is None:
                    continue
                if components['loss_ce'] is not None:
                    ce_losses.append(components['loss_ce'])
                if components['loss_ctr'] is not None:
                    ctr_losses.append(components['loss_ctr'])
                if components['loss_bbox_ground'] is not None:
                    bbox_ground_losses.append(components['loss_bbox_ground'])
                if components['loss_bbox_box'] is not None:
                    bbox_box_losses.append(components['loss_bbox_box'])
                if components['loss_bbox_mask'] is not None:
                    bbox_mask_losses.append(components['loss_bbox_mask'])
                if components['loss_total'] is not None:
                    total_losses.append(components['loss_total'])
            self._queue_aux_loss_logs(
                ce_loss=sum(ce_losses) if ce_losses else None,
                ctr_loss=sum(ctr_losses) if ctr_losses else None,
                bbox_ground_loss=sum(bbox_ground_losses) if bbox_ground_losses else None,
                bbox_box_loss=sum(bbox_box_losses) if bbox_box_losses else None,
                bbox_mask_loss=sum(bbox_mask_losses) if bbox_mask_losses else None,
                total_loss=sum(total_losses) if total_losses else self._to_loss_scalar(loss))

        if return_outputs:
            return loss, {'bbox_outputs': bbox_outputs, 'step_outputs': step_outputs}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if 'bbox_inputs' not in inputs or 'step_inputs' not in inputs:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                bbox_outputs = model(**inputs['bbox_inputs'])
                step_outputs = model(**inputs['step_inputs'])
                loss = bbox_outputs.loss + step_outputs.loss
        loss = loss.mean().detach()
        return (loss, None, None)


def run(
    data_path,
    log_path,
    mode,
    use_text,
    multi_task,
    balanced,
    resume,
    model_path,
    deepcad,
    two_stage,
    pc_encoder_type='pointbert',
    n_points=8192,
    n_point_tokens=256,
    part_points_per_bbox=64,
    utonia_scale=1.0,
    utonia_normalize_coord=True,
    utonia_use_normal=True,
    trainer_overrides=None,
    dataset_config=None,
    dataloader_config=None,
    contrastive_config=None,
    bbox_grounding_config=None,
    resolved_setup=None):
    """
    model_path: base pretrained weights path (do NOT point this at a Trainer checkpoint).
    resume: Trainer checkpoint dir to continue from (e.g., work_dirs/checkpoint-40000).
    """
    # 8192点经过PointBERT+Fourier融合后，FPS下采样为256个输入token
    n_points = int(n_points)
    n_point_tokens = int(n_point_tokens)
    part_points_per_bbox = int(part_points_per_bbox)
    trainer_overrides = trainer_overrides or {}
    dataset_config = dataset_config or {}
    dataloader_config = dataloader_config or {}
    contrastive_config = contrastive_config or {}
    bbox_grounding_config = bbox_grounding_config or {}
    if n_points <= 0:
        raise ValueError('n_points must be a positive integer.')
    if n_point_tokens <= 0:
        raise ValueError('n_point_tokens must be a positive integer.')
    if part_points_per_bbox <= 0:
        raise ValueError('part_points_per_bbox must be a positive integer.')
    if utonia_scale <= 0:
        raise ValueError('utonia_scale must be a positive number.')
    debug_detect_anomaly = bool(contrastive_config.get('debug_detect_anomaly', False))

    dataloader_num_workers = int(trainer_overrides.get('dataloader_num_workers', 8))
    max_steps = int(trainer_overrides.get('max_steps', 120000))
    learning_rate = float(trainer_overrides.get('learning_rate', 2e-4))
    warmup_steps = int(trainer_overrides.get('warmup_steps', 1000))
    weight_decay = float(trainer_overrides.get('weight_decay', 0.01))
    logging_steps = int(trainer_overrides.get('logging_steps', 100))
    save_total_limit = int(trainer_overrides.get('save_total_limit', 20))
    save_steps = int(trainer_overrides.get('save_steps', 10000))
    eval_steps = int(trainer_overrides.get('eval_steps', 5000))
    resolved_setup = resolved_setup or resolve_training_setup(
        data_path=data_path,
        mode=mode,
        use_text=use_text,
        multi_task=multi_task,
        balanced=balanced,
        deepcad=deepcad,
        two_stage=two_stage,
        n_points=n_points,
        part_points_per_bbox=part_points_per_bbox,
        pc_encoder_type=pc_encoder_type,
        utonia_use_normal=utonia_use_normal,
        dataset_config=dataset_config,
        dataloader_config=dataloader_config,
        trainer_overrides=trainer_overrides)

    task_flags = resolved_setup['task_flags']
    deepcad = bool(task_flags['deepcad'])
    multi_task = bool(task_flags['multi_task'])
    two_stage = bool(task_flags['two_stage'])
    batch_size = int(resolved_setup['batch_size'])
    accumulation_steps = int(resolved_setup['accumulation_steps'])
    dataloader_num_workers = int(resolved_setup['dataloader_num_workers'])

    train_dataset = resolved_setup['train_dataset_cls'](**resolved_setup['train_dataset_kwargs'])
    eval_dataset = resolved_setup['eval_dataset_cls'](**resolved_setup['eval_dataset_kwargs'])
    if resolved_setup['text_dataset_kwargs'] is not None:
        text_dataset = Text2CADDataset(**resolved_setup['text_dataset_kwargs'])
        train_dataset = ConcatDataset([train_dataset, text_dataset])

    # 默认使用 Qwen2-VL-2B-Instruct，也可以指定预训练好的权重
    base_model_path = model_path if model_path else 'Qwen/Qwen2-VL-2B-Instruct'
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side='left',
        use_fast=True)
    bbox_grounding_enabled = bool(bbox_grounding_config.get('enabled', False))
    if bbox_grounding_enabled:
        processor.tokenizer.add_special_tokens({
            'additional_special_tokens': list(BBOX_GROUNDING_SPECIAL_TOKENS),
        })

    # 加载 PointBERT 版本的 Cadrille 模型
    model = Cadrille.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        ignore_mismatched_sizes=bbox_grounding_enabled,
        two_stage=two_stage,
        pc_encoder_type=pc_encoder_type,
        n_point_tokens=n_point_tokens,
        part_points_per_bbox=part_points_per_bbox,
        utonia_scale=utonia_scale,
        utonia_normalize_coord=utonia_normalize_coord,
        utonia_use_normal=utonia_use_normal,
        contrastive_enabled=bool(contrastive_config.get('enabled', False)),
        contrastive_lambda=float(contrastive_config.get('lambda', 0.05)),
        contrastive_temperature=float(contrastive_config.get('temperature', 0.07)),
        contrastive_proj_dim=int(contrastive_config.get('proj_dim', 128)),
        contrastive_queue_size=int(contrastive_config.get('queue_size', 4096)),
        contrastive_momentum=float(contrastive_config.get('momentum', 0.999)),
        contrastive_interval=int(contrastive_config.get('interval', 1)),
        contrastive_max_negatives=int(contrastive_config.get('max_negatives', 1024)),
        bbox_grounding_enabled=bbox_grounding_enabled,
        bbox_grounding_lambda=float(bbox_grounding_config.get('lambda', 0.1)),
        bbox_grounding_lambda_box=float(bbox_grounding_config.get('lambda_box', 1.0)),
        bbox_grounding_lambda_mask=float(bbox_grounding_config.get('lambda_mask', 0.25)),
        bbox_grounding_match_mask_weight=float(bbox_grounding_config.get('match_mask_weight', 0.5)),
        bbox_grounding_mask_eps=float(bbox_grounding_config.get('mask_eps', 0.01)))
    if bbox_grounding_enabled:
        model.resize_token_embeddings(len(processor.tokenizer))
        model.bbox_grounding_box_token_id = processor.tokenizer.convert_tokens_to_ids('<BOX>')
        model.bbox_grounding_span_start_token_id = processor.tokenizer.convert_tokens_to_ids('<Bs>')
        model.bbox_grounding_span_end_token_id = processor.tokenizer.convert_tokens_to_ids('<Be>')
        model.bbox_grounding_split_token_id = processor.tokenizer.convert_tokens_to_ids('<SPLIT>')
        model.bbox_grounding_stop_token_id = processor.tokenizer.convert_tokens_to_ids('<STOP>')

    trainer_cls = TwoStageTrainer if two_stage else BalancedTrainer
    contrastive_enabled = bool(contrastive_config.get('enabled', False))
    trainer = trainer_cls(
        model=model,
        args=TrainingArguments(
            output_dir=log_path,
            per_device_train_batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            max_steps=max_steps,
            lr_scheduler_type='cosine',
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=accumulation_steps,
            remove_unused_columns=False,
            logging_steps=logging_steps,
            save_total_limit=save_total_limit,
            save_strategy='steps',
            save_steps=save_steps,
            eval_strategy='steps',
            eval_steps=eval_steps,
            load_best_model_at_end=True,
            report_to=None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(
            collate,
            processor=processor,
            n_points=n_points,
            n_point_tokens=n_point_tokens,
            n_part_point_tokens=part_points_per_bbox,
            pc_encoder_type=pc_encoder_type,
            utonia_scale=utonia_scale,
            utonia_normalize_coord=utonia_normalize_coord,
            utonia_use_normal=utonia_use_normal,
            build_utonia_data_dict=True,
            bbox_grounding_enabled=bbox_grounding_enabled,
            bbox_grounding_normalize_std=float(bbox_grounding_config.get(
                'normalize_std', dataset_config.get('normalize_std_pc', 100.0))),
            bbox_grounding_max_boxes=int(bbox_grounding_config.get('max_boxes', 0))),
        processing_class=processor,
        callbacks=[PrintToFileCallback()],
        balanced_sampling=balanced,
        contrastive_sampling=contrastive_enabled)
    anomaly_ctx = (
        torch.autograd.detect_anomaly()
        if debug_detect_anomaly else contextlib.nullcontext()
    )
    with anomaly_ctx:
        trainer.train(resume_from_checkpoint=resume if resume else None)


if __name__ == '__main__':
    parser = ArgumentParser()
    default_config_path = str(Path(__file__).resolve().with_name('cadrec_config.yaml'))
    parser.add_argument('--config', type=str, default=default_config_path,
                        help='YAML config path for all train/test parameters.')
    args = parser.parse_args()

    config = load_config_with_defaults(args.config)
    common_cfg = get_nested(config, ['common'], {}) or {}
    common_model_cfg = get_nested(common_cfg, ['model'], {}) or {}
    common_pc_cfg = get_nested(common_cfg, ['point_cloud'], {}) or {}
    common_utonia_cfg = get_nested(common_cfg, ['utonia'], {}) or {}

    train_cfg = get_nested(config, ['train'], {}) or {}
    train_paths_cfg = get_nested(train_cfg, ['paths'], {}) or {}
    train_task_cfg = get_nested(train_cfg, ['task'], {}) or {}
    train_dataset_cfg = get_nested(train_cfg, ['dataset'], {}) or {}
    train_dataloader_cfg = get_nested(train_cfg, ['dataloader'], {}) or {}
    train_trainer_cfg = get_nested(train_cfg, ['trainer'], {}) or {}
    train_contrastive_cfg = get_nested(train_cfg, ['contrastive'], {}) or {}
    train_bbox_grounding_cfg = get_nested(train_cfg, ['bbox_grounding'], {}) or {}
    runtime_cfg = get_nested(config, ['runtime'], {}) or {}

    log_path = train_paths_cfg.get('log_path')
    if not log_path:
        raise ValueError('train.paths.log_path must be set in YAML config.')
    data_path = train_paths_cfg.get('data_path', '/data/songhx24/Dataset')
    resolved_setup = resolve_training_setup(
        data_path=data_path,
        mode=train_task_cfg.get('mode', 'pc'),
        use_text=bool(train_task_cfg.get('use_text', False)),
        multi_task=bool(train_task_cfg.get('multi_task', False)),
        balanced=bool(train_task_cfg.get('balanced', False)),
        deepcad=bool(train_task_cfg.get('deepcad', False)),
        two_stage=bool(train_task_cfg.get('two_stage', False)),
        n_points=common_pc_cfg.get('n_points', 8192),
        part_points_per_bbox=common_pc_cfg.get('part_points_per_bbox', 64),
        pc_encoder_type=common_model_cfg.get('pc_encoder_type', 'pointbert'),
        utonia_use_normal=bool(common_utonia_cfg.get('use_normal', True)),
        dataset_config=train_dataset_cfg,
        dataloader_config=train_dataloader_cfg,
        trainer_overrides=train_trainer_cfg)

    config_to_save = copy.deepcopy(config)
    resolved_cfg = config_to_save.setdefault('resolved', {})
    resolved_train_cfg = resolved_cfg.setdefault('train', {})
    resolved_train_cfg['task'] = resolved_setup['resolved_task']
    resolved_train_cfg['dataset'] = resolved_setup['resolved_dataset']
    resolved_train_cfg['dataloader'] = resolved_setup['resolved_dataloader']
    resolved_train_cfg['contrastive'] = copy.deepcopy(train_contrastive_cfg)
    resolved_train_cfg['bbox_grounding'] = copy.deepcopy(train_bbox_grounding_cfg)
    save_name = str(runtime_cfg.get('save_config_name', 'run_config.yaml'))
    save_timestamped = bool(runtime_cfg.get('save_timestamped_copy', True))
    save_path, timestamped_path = save_training_config_snapshot(
        config=config_to_save,
        log_dir=log_path,
        save_config_name=save_name,
        save_timestamped_copy=save_timestamped,
        source_config_path=args.config)
    print(f'[Config] Saved current training config: {save_path}')
    if timestamped_path is not None:
        print(f'[Config] Saved timestamped snapshot: {timestamped_path}')

    model_path = train_paths_cfg.get('model_path', None)
    if model_path is None:
        model_path = common_model_cfg.get('base_model_path', None)

    run(
        data_path=data_path,
        log_path=log_path,
        mode=train_task_cfg.get('mode', 'pc'),
        use_text=bool(train_task_cfg.get('use_text', False)),
        multi_task=bool(train_task_cfg.get('multi_task', False)),
        balanced=bool(train_task_cfg.get('balanced', False)),
        resume=train_paths_cfg.get('resume', None),
        model_path=model_path,
        deepcad=bool(train_task_cfg.get('deepcad', False)),
        two_stage=bool(train_task_cfg.get('two_stage', False)),
        pc_encoder_type=common_model_cfg.get('pc_encoder_type', 'pointbert'),
        n_points=common_pc_cfg.get('n_points', 8192),
        n_point_tokens=common_pc_cfg.get('n_point_tokens', 512),
        part_points_per_bbox=common_pc_cfg.get('part_points_per_bbox', 64),
        utonia_scale=common_utonia_cfg.get('scale', 2.0),
        utonia_normalize_coord=bool(common_utonia_cfg.get('normalize_coord', True)),
        utonia_use_normal=bool(common_utonia_cfg.get('use_normal', True)),
        trainer_overrides=train_trainer_cfg,
        dataset_config=train_dataset_cfg,
        dataloader_config=train_dataloader_cfg,
        contrastive_config=train_contrastive_cfg,
        bbox_grounding_config=train_bbox_grounding_cfg,
        resolved_setup=resolved_setup)

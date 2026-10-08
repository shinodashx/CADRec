import os
import json
import pickle
import re
import open3d
import trimesh
import skimage
import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, Sampler
from pytorch3d.ops import sample_farthest_points


VARIANT_SUFFIX_PATTERN = re.compile(r'^(?P<base>.+)_(?P<index>\d+)$')


def load_cadrecode_v15our_multitask_annotations(root_dir, split):
    split_key = 'val' if split in ('val', 'validation') else split
    split_dir = os.path.join(root_dir, split_key)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f'Split directory not found: {split_dir}')

    annotations = []
    split_entries = [entry for entry in os.scandir(split_dir) if entry.is_dir()]
    has_batch_level = any(entry.name.startswith('batch_') for entry in split_entries)
    sample_dirs = []
    if has_batch_level:
        for batch_entry in split_entries:
            if not batch_entry.name.startswith('batch_'):
                continue
            for sample_entry in os.scandir(batch_entry.path):
                if sample_entry.is_dir():
                    sample_dirs.append((batch_entry.name, sample_entry.name))
    else:
        for sample_entry in split_entries:
            sample_dirs.append(('', sample_entry.name))

    for batch_name, sample_name in sample_dirs:
        is_variant_dir = VARIANT_SUFFIX_PATTERN.match(sample_name) is not None
        point_variants = [('full', f'{sample_name}.npy')]
        if is_variant_dir:
            point_variants.append(('crop', f'{sample_name}_crop.npy'))

        if batch_name:
            source_dir = os.path.join(batch_name, sample_name)
        else:
            source_dir = sample_name
        source_key = source_dir.replace(os.sep, '_')
        folder_type = 2 if is_variant_dir else 1
        rel_mesh = os.path.join(source_dir, f'{sample_name}.stl')
        rel_code = os.path.join(source_dir, f'{sample_name}.py')
        rel_bbox = os.path.join(source_dir, 'bbox.json')
        for point_variant, npy_name in point_variants:
            rel_npy = os.path.join(source_dir, npy_name)
            point_prefix = f'{source_key}_{point_variant}'
            annotations.append({
                'mesh_path': rel_mesh,
                'ply_path': rel_npy,
                'py_path': rel_code,
                'answer_path': rel_code,
                'description': 'Generate cadquery code',
                'file_name': f'{point_prefix}_code',
                'task_type': 'code',
                'folder_type': folder_type,
                'point_variant': point_variant,
                'source_dir': source_dir,
            })
            annotations.append({
                'mesh_path': rel_mesh,
                'ply_path': rel_npy,
                'py_path': rel_bbox,
                'answer_path': rel_bbox,
                'description': 'Generate cad parts bounding box json',
                'file_name': f'{point_prefix}_bbox',
                'task_type': 'bbox',
                'folder_type': folder_type,
                'point_variant': point_variant,
                'source_dir': source_dir,
            })
    return annotations


def mesh_to_point_cloud(mesh, n_points, n_pre_points=8192, return_normals=False):
    vertices, face_idx = trimesh.sample.sample_surface(mesh, n_pre_points)
    normals = None
    if return_normals:
        if getattr(mesh, 'face_normals', None) is not None and len(mesh.face_normals) > 0:
            normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
        else:
            normals = np.zeros_like(vertices, dtype=np.float32)
    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    vertices = np.asarray(vertices[ids], dtype=np.float32)
    if not return_normals:
        return vertices
    normals = np.asarray(normals[ids], dtype=np.float32)
    return vertices, normals


def mesh_to_image(mesh, camera_distance=-1.8, front=[1, 1, 1], width=500, height=500, img_size=128):
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)

    lookat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    front_array = np.array(front, dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)

    eye = lookat + front_array * camera_distance
    right = np.cross(up, front_array)
    right /= np.linalg.norm(right)
    true_up = np.cross(front_array, right)
    rotation_matrix = np.column_stack((right, true_up, front_array)).T
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = -rotation_matrix @ eye

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(camera_params)

    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)
    image = skimage.transform.resize(
        image,
        output_shape=(img_size, img_size),
        order=2,
        anti_aliasing=True,
        preserve_range=True).astype(np.uint8)

    return Image.fromarray(image)


def load_bbox_normalized(bbox_path, normalize_std=100.0):
    with open(bbox_path, 'r') as f:
        data = json.load(f)
    model_box = data['model_box']

    def norm_box(box):
        return {
            'min': np.array([box['min']['x'], box['min']['y'], box['min']['z']], dtype=np.float32) / normalize_std,
            'max': np.array([box['max']['x'], box['max']['y'], box['max']['z']], dtype=np.float32) / normalize_std,
        }

    parts = {}
    for part_name, part_box in model_box['parts'].items():
        parts[part_name] = norm_box(part_box)
    return parts


def _extract_points_and_normals_from_ply_geometry(geometry, return_normals=False):
    if isinstance(geometry, trimesh.Scene):
        point_chunks = []
        normal_chunks = []
        all_have_normals = True
        for item in geometry.geometry.values():
            points, normals = _extract_points_and_normals_from_ply_geometry(
                item,
                return_normals=return_normals)
            if points.shape[0] == 0:
                continue
            point_chunks.append(points)
            if return_normals:
                if normals is None:
                    all_have_normals = False
                else:
                    normal_chunks.append(normals)

        if not point_chunks:
            raise ValueError('PLY scene has no valid point vertices.')

        points = np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
        normals = None
        if return_normals and all_have_normals and len(normal_chunks) == len(point_chunks):
            normals = np.concatenate(normal_chunks, axis=0).astype(np.float32, copy=False)
        return points, normals

    if not hasattr(geometry, 'vertices'):
        raise ValueError(f'Unsupported PLY geometry type: {type(geometry)}')

    points = np.asarray(geometry.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        raise ValueError('PLY vertex array must have shape [N, >=3].')
    points = points[:, :3]

    normals = None
    if return_normals and hasattr(geometry, 'vertex_normals'):
        raw_normals = np.asarray(geometry.vertex_normals, dtype=np.float32)
        if raw_normals.ndim == 2 and raw_normals.shape[0] == points.shape[0] and raw_normals.shape[1] >= 3:
            normals = raw_normals[:, :3]

    return points, normals


def load_point_cloud_from_ply(ply_path, n_points=None, return_normals=False):
    geometry = trimesh.load(ply_path, process=False)
    points, normals = _extract_points_and_normals_from_ply_geometry(
        geometry,
        return_normals=return_normals)
    if return_normals and normals is None:
        normals = np.zeros_like(points, dtype=np.float32)

    return _resize_point_cloud_sample_count(
        points,
        normals=normals,
        n_points=n_points,
        return_normals=return_normals)


def _resize_point_cloud_sample_count(points, normals=None, n_points=None, return_normals=False):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        raise ValueError('Point cloud array must have shape [N, >=3].')
    points = points[:, :3]

    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.ndim != 2 or normals.shape[0] != points.shape[0] or normals.shape[1] < 3:
            raise ValueError('Point normal array must have shape [N, >=3].')
        normals = normals[:, :3]
    elif return_normals:
        normals = np.zeros_like(points, dtype=np.float32)

    if n_points is not None and int(n_points) > 0 and points.shape[0] != int(n_points):
        target = int(n_points)
        if points.shape[0] > target:
            _, ids = sample_farthest_points(
                torch.tensor(points, dtype=torch.float32).unsqueeze(0),
                K=target)
            ids = ids[0].numpy()
            points = points[ids]
            if normals is not None:
                normals = normals[ids]
        else:
            repeats = (target + points.shape[0] - 1) // points.shape[0]
            tiled_ids = np.tile(np.arange(points.shape[0], dtype=np.int64), repeats)[:target]
            points = points[tiled_ids]
            if normals is not None:
                normals = normals[tiled_ids]

    if not return_normals:
        return points
    return points, normals


def load_point_cloud_from_npy(npy_path, n_points=None, return_normals=False):
    point_array = np.load(npy_path)
    point_array = np.asarray(point_array, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[0] == 0 or point_array.shape[1] < 3:
        raise ValueError(f'NPY point cloud must have shape [N, >=3]: {npy_path}')

    points = point_array[:, :3]
    normals = None
    if point_array.shape[1] >= 6:
        normals = point_array[:, 3:6]
    elif return_normals:
        raise ValueError(f'NPY point cloud must have xyz+normal features [N, >=6]: {npy_path}')

    return _resize_point_cloud_sample_count(
        points,
        normals=normals,
        n_points=n_points,
        return_normals=return_normals)


class CadRecodeDataset(Dataset):
    def __init__(self, root_dir, split, n_points, normalize_std_pc, noise_scale_pc, img_size,
                normalize_std_img, noise_scale_img, num_imgs, mode, n_samples=None, ext='stl',
                annotations=None, use_point_normals=False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.n_samples = n_samples
        self.n_points = n_points
        self.normalize_std_pc = normalize_std_pc
        self.noise_scale_pc = noise_scale_pc
        self.normalize_std_img = normalize_std_img
        self.noise_scale_img = noise_scale_img
        self.num_imgs = num_imgs
        self.mode = mode
        self.use_point_normals = bool(use_point_normals)
        if annotations is not None:
            self.annotations = annotations
        elif self.split in ['train', 'val']:
            pkl_path = os.path.join(self.root_dir, f'{self.split}.pkl')
            with open(pkl_path, 'rb') as f:
                self.annotations = pickle.load(f)
        else:
            paths = os.listdir(os.path.join(self.root_dir, self.split))
            self.annotations = [
                {'mesh_path': os.path.join(self.split, f)}
                for f in paths if f.endswith('.stl')
            ]

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]

        if self.mode == 'pc':
            input_item = self.get_point_cloud(item)
        elif self.mode == 'img':
            input_item = self.get_img(item)
        elif self.mode == 'pc_img':
            if np.random.rand() < 0.5:
                input_item = self.get_point_cloud(item)
            else:
                input_item = self.get_img(item)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        input_item['description'] = item.get(
            'description',
            input_item.get('description', 'Generate cadquery code'))
        file_name = item.get('file_name')
        if file_name is None:
            file_name = os.path.basename(item['mesh_path'])[:-4]
        input_item['file_name'] = file_name

        if self.split in ['train', 'val']:
            answer_path = item.get('answer_path', item['py_path'])
            answer_path = os.path.join(self.root_dir, answer_path)
            with open(answer_path, 'r') as f:
                answer = f.read()
            input_item['answer'] = answer

        return input_item

    def get_img(self, item):
        mesh = trimesh.load(os.path.join(self.root_dir, item['mesh_path']))
        if self.split in ['train', 'val']:
            mesh.apply_transform(trimesh.transformations.scale_matrix(1 / self.normalize_std_img))
            mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(vertices)
        mesh.triangles = open3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
        mesh.compute_vertex_normals()

        fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
        images = []
        for front in fronts:
            image = mesh_to_image(
                mesh, camera_distance=-0.9, front=front, img_size=self.img_size)
            images.append(image)

        images = [ImageOps.expand(image, border=3, fill='black') for image in images]
        if self.num_imgs == 1:
            images = [images[0]]
        elif self.num_imgs == 2:
            images = [Image.fromarray(np.hstack((
                np.array(images[0]), np.array(images[1])
            )))]
        elif self.num_imgs == 4:
            images = [Image.fromarray(np.vstack((
                np.hstack((np.array(images[0]), np.array(images[1]))),
                np.hstack((np.array(images[2]), np.array(images[3])))
            )))]
        else:
            raise ValueError(f'Invalid number of images: {self.num_imgs}')

        input_item = {
            'video': images,
            'description': 'Generate cadquery code'
        }
        return input_item

    def get_point_cloud(self, item):
        mesh = trimesh.load(os.path.join(self.root_dir, item['mesh_path']))
        mesh = self._augment_pc(mesh)
        sampled = mesh_to_point_cloud(
            mesh,
            self.n_points,
            return_normals=self.use_point_normals)
        if self.use_point_normals:
            point_cloud, point_normal = sampled
        else:
            point_cloud = sampled

        if self.split in ['train', 'val']:
            point_cloud = point_cloud / self.normalize_std_pc
        else:
            point_cloud = (point_cloud - 0.5) * 2

        input_item = {
            'point_cloud': point_cloud,
            'description': 'Generate cadquery code',
        }
        if self.use_point_normals:
            input_item['point_normal'] = point_normal
        return input_item

    def _augment_pc(self, mesh):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(loc=0, scale=self.noise_scale_pc, size=mesh.vertices.shape)
        return mesh


class Text2CADDataset(Dataset):
    def __init__(self, root_dir, split, code_dir='cadquery', n_samples=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.n_samples = n_samples
        self.code_dir = code_dir
        pkl_path = os.path.join(self.root_dir, f'{self.split}.pkl')
        with open(pkl_path, 'rb') as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]

        input_item = {
            'description': item['description'],
            'file_name': item['uid']
        }

        if self.split in ['train', 'val']:
            py_path = f'{item["uid"]}.py'
            py_path = os.path.join(self.root_dir, self.code_dir, py_path)
            with open(py_path, 'r') as f:
                answer = f.read()
            input_item['answer'] = answer
        return input_item


class DeepCADDataset(Dataset):
    """DeepCAD dataset using train_val_test_split.json for data partitioning."""
    def __init__(self, root_dir, split_json_path, split, n_points, normalize_std_pc,
                 noise_scale_pc, img_size, normalize_std_img, noise_scale_img, num_imgs,
                 mode, n_samples=None, annotations=None, use_point_normals=False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.n_points = n_points
        self.normalize_std_pc = normalize_std_pc
        self.noise_scale_pc = noise_scale_pc
        self.img_size = img_size
        self.normalize_std_img = normalize_std_img
        self.noise_scale_img = noise_scale_img
        self.num_imgs = num_imgs
        self.mode = mode
        self.n_samples = n_samples
        self.use_point_normals = bool(use_point_normals)

        if annotations is not None:
            self.annotations = annotations
        else:
            pkl_path = os.path.join(self.root_dir, f'{split}.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    self.annotations = pickle.load(f)
            else:
                self.annotations = self._load_annotations_from_split(
                    root_dir=self.root_dir,
                    split_json_path=split_json_path,
                    split=split)
                with open(pkl_path, 'wb') as f:
                    pickle.dump(self.annotations, f)

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    @staticmethod
    def _infer_pc_type_id(item):
        # crop point cloud: 1, full point cloud: 0
        for key in ('ply_path', 'point_cloud_path', 'point_path'):
            value = item.get(key, None)
            if not isinstance(value, str):
                continue
            name = os.path.basename(value).lower()
            if name.endswith(('.npy', '.npz', '.ply')) and 'crop' in name:
                return 1
        return 0

    @staticmethod
    def _infer_source_key(item):
        source_key = item.get('source_key', None)
        if isinstance(source_key, str) and source_key:
            return source_key
        source_dir = item.get('source_dir', None)
        if isinstance(source_dir, str) and source_dir:
            return source_dir.replace(os.sep, '_')

        file_name = item.get('file_name', '')
        if isinstance(file_name, str) and file_name:
            source_key = file_name
            for suffix in ('_code', '_bbox', '_step_pc', '_step', '_full', '_crop'):
                if source_key.endswith(suffix):
                    source_key = source_key[:-len(suffix)]
            if source_key:
                return source_key

        for key in ('ply_path', 'mesh_path', 'point_cloud_path', 'point_path', 'py_path'):
            value = item.get(key, None)
            if isinstance(value, str) and value:
                stem = os.path.splitext(os.path.basename(value))[0]
                if stem:
                    return stem
        return ''

    @staticmethod
    def _infer_task_type(item):
        task_type = item.get('task_type', None)
        if isinstance(task_type, str) and task_type:
            return task_type

        file_name = item.get('file_name', '')
        if isinstance(file_name, str):
            lower_name = file_name.lower()
            if lower_name.endswith('_bbox'):
                return 'bbox'
            if lower_name.endswith('_code'):
                return 'code'
            if lower_name.endswith('_step_pc'):
                return 'step_pc'
        return 'code'

    def __getitem__(self, index):
        item = self.annotations[index]

        if self.mode == 'pc':
            input_item = self.get_point_cloud(item)
        elif self.mode == 'img':
            input_item = self.get_img(item)
        elif self.mode == 'pc_img':
            if np.random.rand() < 0.5:
                input_item = self.get_point_cloud(item)
            else:
                input_item = self.get_img(item)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        input_item['description'] = item.get('description', 'Generate cadquery code')
        input_item['file_name'] = item.get('file_name', '')
        input_item['source_key'] = self._infer_source_key(item)
        input_item['task_type'] = self._infer_task_type(item)
        if 'point_cloud' in input_item:
            input_item['pc_type_id'] = self._infer_pc_type_id(item)

        if self.split in ['train', 'val']:
            answer_path = item.get('answer_path', item['py_path'])
            answer_path = self._resolve_path(answer_path)
            with open(answer_path, 'r') as f:
                answer = f.read()
            input_item['answer'] = answer

        return input_item

    def get_img(self, item):
        mesh = trimesh.load(self._resolve_path(item['mesh_path']))
        mesh.apply_transform(trimesh.transformations.scale_matrix(1 / self.normalize_std_img))
        mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        o3d_mesh = open3d.geometry.TriangleMesh()
        o3d_mesh.vertices = open3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = open3d.utility.Vector3iVector(faces)
        o3d_mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
        o3d_mesh.compute_vertex_normals()

        fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
        images = []
        for front in fronts:
            image = mesh_to_image(
                o3d_mesh, camera_distance=-0.9, front=front, img_size=self.img_size)
            images.append(image)

        images = [ImageOps.expand(image, border=3, fill='black') for image in images]
        if self.num_imgs == 1:
            images = [images[0]]
        elif self.num_imgs == 2:
            images = [Image.fromarray(np.hstack((
                np.array(images[0]), np.array(images[1])
            )))]
        elif self.num_imgs == 4:
            images = [Image.fromarray(np.vstack((
                np.hstack((np.array(images[0]), np.array(images[1]))),
                np.hstack((np.array(images[2]), np.array(images[3])))
            )))]
        else:
            raise ValueError(f'Invalid number of images: {self.num_imgs}')

        return {'video': images, 'description': 'Generate cadquery code'}

    def get_point_cloud(self, item):
        mesh = trimesh.load(self._resolve_path(item['mesh_path']))
        mesh = self._augment_pc(mesh)
        sampled = mesh_to_point_cloud(
            mesh,
            self.n_points,
            return_normals=self.use_point_normals)
        if self.use_point_normals:
            point_cloud, point_normal = sampled
        else:
            point_cloud = sampled
        point_cloud = point_cloud / self.normalize_std_pc
        output = {'point_cloud': point_cloud, 'description': 'Generate cadquery code'}
        if self.use_point_normals:
            output['point_normal'] = point_normal
        return output

    def _augment_pc(self, mesh):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(loc=0, scale=self.noise_scale_pc, size=mesh.vertices.shape)
        return mesh

    def _resolve_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.root_dir, path)

    @staticmethod
    def _load_annotations_from_split(root_dir, split_json_path, split):
        import json
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)

        split_key = 'validation' if split == 'val' else split
        entries = split_data.get(split_key, [])

        annotations = []
        for entry in entries:
            entry_norm = entry.replace('/', os.sep)
            file_id = os.path.basename(entry_norm)
            mesh_path = os.path.join(root_dir, entry_norm, f'{file_id}.stl')
            py_path = os.path.join(root_dir, entry_norm, f'{file_id}.py')

            if os.path.exists(mesh_path) and os.path.exists(py_path):
                annotations.append({
                    'mesh_path': os.path.relpath(mesh_path, root_dir),
                    'py_path': os.path.relpath(py_path, root_dir),
                    'answer_path': os.path.relpath(py_path, root_dir),
                    'file_name': file_id,
                    'description': 'Generate cadquery code'
                })
        return annotations


class DeepCADMultiTaskDataset(DeepCADDataset):
    """DeepCAD multitask dataset for both code and bbox generation."""
    def __init__(self, root_dir, split_json_path, split, n_points, normalize_std_pc,
                 noise_scale_pc, img_size, normalize_std_img, noise_scale_img, num_imgs,
                 mode, n_samples=None, use_point_normals=False):
        self.task_descriptions = {
            'code': 'Generate cadquery code',
            'bbox': 'Generate cad parts bounding box json'
        }
        multi_pkl = os.path.join(root_dir, f'{split}_multi.pkl')
        if os.path.exists(multi_pkl):
            with open(multi_pkl, 'rb') as f:
                annotations = pickle.load(f)
        else:
            annotations = self._load_multitask_annotations(
                root_dir=root_dir,
                split_json_path=split_json_path,
                split=split)
            with open(multi_pkl, 'wb') as f:
                pickle.dump(annotations, f)
        super().__init__(
            root_dir=root_dir,
            split_json_path=split_json_path,
            split=split,
            n_points=n_points,
            normalize_std_pc=normalize_std_pc,
            noise_scale_pc=noise_scale_pc,
            img_size=img_size,
            normalize_std_img=normalize_std_img,
            noise_scale_img=noise_scale_img,
            num_imgs=num_imgs,
            mode=mode,
            n_samples=n_samples,
            use_point_normals=use_point_normals,
            annotations=annotations)

    @staticmethod
    def _load_multitask_annotations(root_dir, split_json_path, split):
        import json
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)

        split_key = 'validation' if split == 'val' else split
        entries = split_data.get(split_key, [])

        annotations = []
        for entry in entries:
            entry_norm = entry.replace('/', os.sep)
            file_id = os.path.basename(entry_norm)
            stl_path = os.path.join(root_dir, entry_norm, f'{file_id}.stl')
            py_path = os.path.join(root_dir, entry_norm, f'{file_id}.py')
            json_path = os.path.join(root_dir, entry_norm, 'bbox.json')

            if not (os.path.exists(stl_path) and os.path.exists(py_path) and os.path.exists(json_path)):
                continue

            sample_id = entry_norm.replace(os.sep, '_')
            rel_mesh = os.path.relpath(stl_path, root_dir)
            rel_py = os.path.relpath(py_path, root_dir)
            rel_json = os.path.relpath(json_path, root_dir)

            annotations.append({
                'mesh_path': rel_mesh,
                'py_path': rel_py,
                'answer_path': rel_py,
                'description': 'Generate cadquery code',
                'file_name': f'{sample_id}_code',
                'source_key': sample_id,
                'task_type': 'code'
            })
            annotations.append({
                'mesh_path': rel_mesh,
                'py_path': rel_json,
                'answer_path': rel_json,
                'description': 'Generate cad parts bounding box json',
                'file_name': f'{sample_id}_bbox',
                'source_key': sample_id,
                'task_type': 'bbox'
            })
        return annotations


class DeepCADV2MultiTaskDataset(DeepCADDataset):
    """DeepCAD V2 multitask dataset using preprocessed ply point clouds."""
    annotation_cache_tag = 'v2multi'
    point_cloud_ext = '.ply'
    point_cloud_loader = staticmethod(load_point_cloud_from_ply)

    def __init__(self, root_dir, split_json_path, split, n_points, normalize_std_pc,
                 noise_scale_pc, img_size, normalize_std_img, noise_scale_img, num_imgs,
                 mode, n_samples=None, use_point_normals=False):
        self.task_descriptions = {
            'code': 'Generate cadquery code',
            'bbox': 'Generate cad parts bounding box json'
        }
        cache_path = os.path.join(root_dir, f'{split}_{self.annotation_cache_tag}.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                annotations = pickle.load(f)
        else:
            annotations = self._load_multitask_annotations(
                root_dir=root_dir,
                split_json_path=split_json_path,
                split=split)
            with open(cache_path, 'wb') as f:
                pickle.dump(annotations, f)

        super().__init__(
            root_dir=root_dir,
            split_json_path=split_json_path,
            split=split,
            n_points=n_points,
            normalize_std_pc=normalize_std_pc,
            noise_scale_pc=noise_scale_pc,
            img_size=img_size,
            normalize_std_img=normalize_std_img,
            noise_scale_img=noise_scale_img,
            num_imgs=num_imgs,
            mode=mode,
            n_samples=n_samples,
            use_point_normals=use_point_normals,
            annotations=annotations)

    def get_point_cloud(self, item):
        sampled = self.point_cloud_loader(
            self._resolve_path(item['ply_path']),
            n_points=self.n_points,
            return_normals=self.use_point_normals)
        if self.use_point_normals:
            point_cloud, point_normal = sampled
        else:
            point_cloud = sampled

        point_cloud = point_cloud.astype(np.float32)
        point_cloud = self._augment_point_cloud(point_cloud)
        point_cloud = point_cloud / self.normalize_std_pc

        output = {
            'point_cloud': point_cloud,
            'description': item.get('description', 'Generate cadquery code')
        }
        if self.use_point_normals:
            output['point_normal'] = point_normal.astype(np.float32)
        return output

    def _augment_point_cloud(self, point_cloud):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            noise = np.random.normal(
                loc=0.0,
                scale=self.noise_scale_pc,
                size=point_cloud.shape)
            point_cloud = point_cloud + noise.astype(np.float32)
        return point_cloud

    @classmethod
    def _load_multitask_annotations(cls, root_dir, split_json_path, split):
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)

        split_key = 'validation' if split == 'val' else split
        entries = split_data.get(split_key, [])

        annotations = []
        for entry in entries:
            entry_norm = entry.replace('/', os.sep)
            file_id = os.path.basename(entry_norm)
            sample_dir = os.path.join(root_dir, entry_norm)
            stl_path = os.path.join(sample_dir, f'{file_id}.stl')
            py_path = os.path.join(sample_dir, f'{file_id}.py')
            ply_path = os.path.join(sample_dir, f'{file_id}{cls.point_cloud_ext}')
            crop_ply_path = os.path.join(sample_dir, f'{file_id}_crop{cls.point_cloud_ext}')
            json_path = os.path.join(sample_dir, 'bbox.json')

            required_paths = [stl_path, py_path, ply_path, json_path]
            if not all(os.path.exists(path) for path in required_paths):
                continue

            is_variant_dir = VARIANT_SUFFIX_PATTERN.match(file_id) is not None
            if is_variant_dir:
                if not os.path.exists(crop_ply_path):
                    continue
                folder_type = 2
            else:
                folder_type = 1
            sample_id = entry_norm.replace(os.sep, '_')
            rel_stl = os.path.relpath(stl_path, root_dir)
            rel_py = os.path.relpath(py_path, root_dir)
            rel_ply = os.path.relpath(ply_path, root_dir)
            rel_json = os.path.relpath(json_path, root_dir)

            point_variants = [
                ('full', rel_ply, f'{sample_id}_full')
            ]
            if folder_type == 2:
                point_variants.append((
                    'crop',
                    os.path.relpath(crop_ply_path, root_dir),
                    f'{sample_id}_crop'))

            for point_variant, rel_point_path, point_prefix in point_variants:
                annotations.append({
                    'mesh_path': rel_stl,
                    'ply_path': rel_point_path,
                    'py_path': rel_py,
                    'answer_path': rel_py,
                    'description': 'Generate cadquery code',
                    'file_name': f'{point_prefix}_code',
                    'source_key': sample_id,
                    'task_type': 'code',
                    'folder_type': folder_type,
                    'point_variant': point_variant,
                })
                annotations.append({
                    'mesh_path': rel_stl,
                    'ply_path': rel_point_path,
                    'py_path': rel_json,
                    'answer_path': rel_json,
                    'description': 'Generate cad parts bounding box json',
                    'file_name': f'{point_prefix}_bbox',
                    'source_key': sample_id,
                    'task_type': 'bbox',
                    'folder_type': folder_type,
                    'point_variant': point_variant,
                })
        return annotations


class DeepCADV22MultiTaskDataset(DeepCADV2MultiTaskDataset):
    """DeepCAD V2.2 multitask dataset using preprocessed npy point clouds."""
    annotation_cache_tag = 'v22multi'
    point_cloud_ext = '.npy'
    point_cloud_loader = staticmethod(load_point_cloud_from_npy)


class CadRecodeV15OURDataset(DeepCADV22MultiTaskDataset):
    """CadRecode V1.5 OUR multitask dataset using preprocessed npy point clouds."""
    annotation_cache_tag = 'cadrecode_v15our_multi'
    point_cloud_ext = '.npy'
    point_cloud_loader = staticmethod(load_point_cloud_from_npy)

    def __init__(self, root_dir, split_json_path, split, n_points, normalize_std_pc,
                 noise_scale_pc, img_size, normalize_std_img, noise_scale_img, num_imgs,
                 mode, n_samples=None, use_point_normals=False):
        self.task_descriptions = {
            'code': 'Generate cadquery code',
            'bbox': 'Generate cad parts bounding box json'
        }
        split_key = 'val' if split in ('val', 'validation') else split
        split_root_dir = os.path.join(root_dir, split_key)
        split_pkl_path = os.path.join(root_dir, f'{split}.pkl')
        cache_pkl_path = os.path.join(root_dir, f'{split}_{self.annotation_cache_tag}.pkl')
        if os.path.exists(split_pkl_path):
            with open(split_pkl_path, 'rb') as f:
                annotations = pickle.load(f)
        elif os.path.exists(cache_pkl_path):
            with open(cache_pkl_path, 'rb') as f:
                annotations = pickle.load(f)
        else:
            annotations = self._load_multitask_annotations(
                root_dir=root_dir,
                split_json_path=split_json_path,
                split=split)
            with open(split_pkl_path, 'wb') as f:
                pickle.dump(annotations, f)

        DeepCADDataset.__init__(
            self,
            root_dir=split_root_dir,
            split_json_path=split_json_path,
            split=split,
            n_points=n_points,
            normalize_std_pc=normalize_std_pc,
            noise_scale_pc=noise_scale_pc,
            img_size=img_size,
            normalize_std_img=normalize_std_img,
            noise_scale_img=noise_scale_img,
            num_imgs=num_imgs,
            mode=mode,
            n_samples=n_samples,
            use_point_normals=use_point_normals,
            annotations=annotations)

    @classmethod
    def _load_multitask_annotations(cls, root_dir, split_json_path, split):
        return load_cadrecode_v15our_multitask_annotations(root_dir=root_dir, split=split)


class DeepCADTwoStageDataset(DeepCADDataset):
    """DeepCAD two-stage dataset: stage-1 bbox + stage-2 step code with bbox part indices."""
    def __init__(self, root_dir, split_json_path, split, n_points, normalize_std_pc,
                 noise_scale_pc, img_size, normalize_std_img, noise_scale_img, num_imgs,
                 mode, n_samples=None, n_points_per_part=64, use_point_normals=False):
        self.task_descriptions = {
            'bbox': 'Generate cad parts bounding box json',
            'step_pc': 'generate cadquery code by step point cloud',
        }
        self.n_points_per_part = int(n_points_per_part)
        if self.n_points_per_part <= 0:
            raise ValueError('n_points_per_part must be a positive integer.')
        self.crop_eps = 0.01

        twostage_pkl = os.path.join(root_dir, f'{split}_twostage.pkl')
        if os.path.exists(twostage_pkl):
            with open(twostage_pkl, 'rb') as f:
                annotations = pickle.load(f)
        else:
            annotations = self._load_twostage_annotations(
                root_dir=root_dir,
                split_json_path=split_json_path,
                split=split)
            with open(twostage_pkl, 'wb') as f:
                pickle.dump(annotations, f)

        super().__init__(
            root_dir=root_dir,
            split_json_path=split_json_path,
            split=split,
            n_points=n_points,
            normalize_std_pc=normalize_std_pc,
            noise_scale_pc=noise_scale_pc,
            img_size=img_size,
            normalize_std_img=normalize_std_img,
            noise_scale_img=noise_scale_img,
            num_imgs=num_imgs,
            mode=mode,
            n_samples=n_samples,
            use_point_normals=use_point_normals,
            annotations=annotations)

    def _sample_part_indices(self, points, candidate_indices, target_count, center):
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
            K=target_count)
        local_ids = local_ids[0].numpy()
        return candidate_indices[local_ids].astype(np.int64)

    def get_part_point_indices(self, point_cloud, item):
        parts = load_bbox_normalized(
            self._resolve_path(item['bbox_path']),
            normalize_std=self.normalize_std_pc)

        point_cloud = point_cloud.astype(np.float32)
        part_indices = []
        for part_name in parts.keys():
            bbox_min = parts[part_name]['min']
            bbox_max = parts[part_name]['max']
            mask = np.all(
                (point_cloud >= bbox_min - self.crop_eps) &
                (point_cloud <= bbox_max + self.crop_eps),
                axis=1)
            candidate_indices = np.nonzero(mask)[0]
            center = (bbox_min + bbox_max) / 2.0
            sampled_indices = self._sample_part_indices(
                point_cloud,
                candidate_indices,
                self.n_points_per_part,
                center)
            part_indices.append(sampled_indices)

        if len(part_indices) == 0:
            return np.zeros((0, self.n_points_per_part), dtype=np.int64)
        return np.stack(part_indices, axis=0)

    def __getitem__(self, index):
        item = self.annotations[index]

        mesh = trimesh.load(self._resolve_path(item['mesh_path']))
        mesh = self._augment_pc(mesh)
        sampled = mesh_to_point_cloud(
            mesh,
            self.n_points,
            return_normals=self.use_point_normals)
        if self.use_point_normals:
            point_cloud, point_normal = sampled
        else:
            point_cloud = sampled
        point_cloud = point_cloud.astype(np.float32)
        point_cloud = point_cloud / self.normalize_std_pc

        part_point_indices = self.get_part_point_indices(point_cloud, item)
        input_item = {
            'point_cloud': point_cloud,
            'part_point_indices': part_point_indices,
            'n_parts': int(part_point_indices.shape[0]),
            'bbox_description': self.task_descriptions['bbox'],
            'step_description': self.task_descriptions['step_pc'],
            'file_name': item.get('file_name', ''),
            'source_key': self._infer_source_key(item),
            'pc_type_id': self._infer_pc_type_id(item),
        }
        if self.use_point_normals:
            input_item['point_normal'] = point_normal.astype(np.float32)

        if self.split in ['train', 'val']:
            bbox_answer_path = item.get('bbox_answer_path', item['bbox_path'])
            step_answer_path = item.get('step_answer_path', item['py_path'])
            with open(self._resolve_path(bbox_answer_path), 'r') as f:
                input_item['bbox_answer'] = f.read()
            with open(self._resolve_path(step_answer_path), 'r') as f:
                input_item['step_answer'] = f.read()

        return input_item

    @staticmethod
    def _load_twostage_annotations(root_dir, split_json_path, split):
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)

        split_key = 'validation' if split == 'val' else split
        entries = split_data.get(split_key, [])

        annotations = []
        for entry in entries:
            entry_norm = entry.replace('/', os.sep)
            file_id = os.path.basename(entry_norm)
            stl_path = os.path.join(root_dir, entry_norm, f'{file_id}.stl')
            py_path = os.path.join(root_dir, entry_norm, f'{file_id}.py')
            bbox_path = os.path.join(root_dir, entry_norm, 'bbox.json')

            if not (os.path.exists(stl_path) and os.path.exists(py_path) and os.path.exists(bbox_path)):
                continue

            sample_id = entry_norm.replace(os.sep, '_')
            annotations.append({
                'mesh_path': os.path.relpath(stl_path, root_dir),
                'py_path': os.path.relpath(py_path, root_dir),
                'bbox_path': os.path.relpath(bbox_path, root_dir),
                'bbox_answer_path': os.path.relpath(bbox_path, root_dir),
                'step_answer_path': os.path.relpath(py_path, root_dir),
                'file_name': sample_id,
                'source_key': sample_id,
            })
        return annotations


class BalancedMultiTaskSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=None):
        if batch_size % 2 != 0:
            raise ValueError('Balanced sampling requires an even batch size.')
        self.dataset = dataset
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.seed = seed
        self._iteration = 0

        annotations = getattr(dataset, 'annotations', None)
        if annotations is None:
            raise ValueError('Dataset missing annotations for balanced sampling.')
        self.code_indices = [
            idx for idx, ann in enumerate(annotations)
            if ann.get('task_type') == 'code'
        ]
        self.json_indices = [
            idx for idx, ann in enumerate(annotations)
            if ann.get('task_type') == 'bbox'
        ]
        if len(self.code_indices) == 0 or len(self.json_indices) == 0:
            raise ValueError('Balanced sampling requires both code and json samples.')

        batches_per_epoch = min(
            len(self.code_indices) // self.half_batch,
            len(self.json_indices) // self.half_batch)
        self.num_samples = batches_per_epoch * self.batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        if self.num_samples == 0:
            return iter([])
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(int(self.seed) + self._iteration)
        else:
            generator.seed()
        self._iteration += 1

        code_perm = torch.randperm(len(self.code_indices), generator=generator).tolist()
        json_perm = torch.randperm(len(self.json_indices), generator=generator).tolist()
        code_perm = code_perm[: self.num_samples // 2]
        json_perm = json_perm[: self.num_samples // 2]

        def iterator():
            for batch_idx in range(self.num_samples // self.batch_size):
                start = batch_idx * self.half_batch
                code_batch = [
                    self.code_indices[i] for i in code_perm[start:start + self.half_batch]
                ]
                json_batch = [
                    self.json_indices[i] for i in json_perm[start:start + self.half_batch]
                ]
                combined = code_batch + json_batch
                order = torch.randperm(len(combined), generator=generator).tolist()
                for idx in order:
                    yield combined[idx]

        return iterator()


def _sampler_infer_task_type(annotation):
    task_type = annotation.get('task_type', None)
    if isinstance(task_type, str) and task_type:
        return task_type

    file_name = annotation.get('file_name', '')
    if isinstance(file_name, str):
        lower_name = file_name.lower()
        if lower_name.endswith('_bbox'):
            return 'bbox'
        if lower_name.endswith('_code'):
            return 'code'
        if lower_name.endswith('_step_pc'):
            return 'step_pc'
    return ''


def _sampler_infer_source_key(annotation):
    source_key = annotation.get('source_key', None)
    if isinstance(source_key, str) and source_key:
        return source_key

    source_dir = annotation.get('source_dir', None)
    if isinstance(source_dir, str) and source_dir:
        return source_dir.replace(os.sep, '_')

    file_name = annotation.get('file_name', '')
    if isinstance(file_name, str) and file_name:
        source_key = file_name
        for suffix in ('_code', '_bbox', '_step_pc', '_step', '_full', '_crop'):
            if source_key.endswith(suffix):
                source_key = source_key[:-len(suffix)]
        if source_key:
            return source_key

    for key in ('ply_path', 'mesh_path', 'point_cloud_path', 'point_path', 'py_path'):
        value = annotation.get(key, None)
        if isinstance(value, str) and value:
            stem = os.path.splitext(os.path.basename(value))[0]
            if stem:
                return stem
    return ''


def _sampler_infer_pc_type_id(annotation):
    raw_type_id = annotation.get('pc_type_id', None)
    if raw_type_id is not None:
        try:
            return 1 if int(raw_type_id) == 1 else 0
        except (TypeError, ValueError):
            pass

    point_variant = annotation.get('point_variant', None)
    if isinstance(point_variant, str):
        normalized = point_variant.strip().lower()
        if normalized == 'crop':
            return 1
        if normalized == 'full':
            return 0

    for key in ('ply_path', 'point_cloud_path', 'point_path'):
        value = annotation.get(key, None)
        if not isinstance(value, str):
            continue
        name = os.path.basename(value).lower()
        if name.endswith(('.npy', '.npz', '.ply')) and 'crop' in name:
            return 1
    return 0


class HybridBalancedContrastiveSampler(Sampler):
    """
    Mixed sampler for contrastive training:
    - Paired batches: force full/crop positive pairs within each task half.
    - Regular batches: keep standard balanced code/json sampling.
    - The two batch types are randomly interleaved each epoch.
    """

    def __init__(self, dataset, batch_size, seed=None):
        if batch_size < 4:
            raise ValueError('Hybrid sampler requires batch_size >= 4.')
        if batch_size % 2 != 0:
            raise ValueError('Hybrid sampler requires an even batch size.')
        if batch_size % 4 != 0:
            raise ValueError(
                'Hybrid sampler requires batch_size divisible by 4 '
                '(half-batch per task must be even to form full/crop pairs).')

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.half_batch = self.batch_size // 2
        self.pairs_per_task_per_batch = self.half_batch // 2
        self.seed = seed
        self._iteration = 0

        annotations = getattr(dataset, 'annotations', None)
        if annotations is None:
            raise ValueError('Dataset missing annotations for hybrid contrastive sampling.')

        self.code_indices = []
        self.json_indices = []
        self._task_source_to_indices = {
            'code': {},
            'bbox': {},
        }
        for idx, ann in enumerate(annotations):
            task_type = _sampler_infer_task_type(ann)
            if task_type == 'code':
                self.code_indices.append(idx)
            elif task_type == 'bbox':
                self.json_indices.append(idx)
            else:
                continue

            source_key = _sampler_infer_source_key(ann)
            if not source_key:
                continue
            pc_type_id = _sampler_infer_pc_type_id(ann)
            source_bucket = self._task_source_to_indices[task_type].setdefault(
                source_key, {0: [], 1: []})
            source_bucket[pc_type_id].append(idx)

        if len(self.code_indices) == 0 or len(self.json_indices) == 0:
            raise ValueError('Hybrid sampler requires both code and json samples.')

        self._pair_counts = {
            'code': self._count_available_pairs(self._task_source_to_indices['code']),
            'bbox': self._count_available_pairs(self._task_source_to_indices['bbox']),
        }
        self.paired_batches = min(
            self._pair_counts['code'] // self.pairs_per_task_per_batch,
            self._pair_counts['bbox'] // self.pairs_per_task_per_batch)

        used_code_for_pairs = self.paired_batches * self.pairs_per_task_per_batch * 2
        used_bbox_for_pairs = self.paired_batches * self.pairs_per_task_per_batch * 2
        remaining_code = max(len(self.code_indices) - used_code_for_pairs, 0)
        remaining_bbox = max(len(self.json_indices) - used_bbox_for_pairs, 0)
        self.regular_batches = min(
            remaining_code // self.half_batch,
            remaining_bbox // self.half_batch)

        self.num_batches = self.paired_batches + self.regular_batches
        self.num_samples = self.num_batches * self.batch_size
        if self.num_samples == 0:
            raise ValueError(
                'Hybrid sampler produced zero samples. '
                f'code={len(self.code_indices)} bbox={len(self.json_indices)} '
                f'pair_counts={self._pair_counts}.')

    @staticmethod
    def _count_available_pairs(source_to_type_indices):
        return sum(
            min(len(type_indices[0]), len(type_indices[1]))
            for type_indices in source_to_type_indices.values())

    @staticmethod
    def _build_pairs_for_task(source_to_type_indices, generator):
        pairs = []
        for type_indices in source_to_type_indices.values():
            full_indices = type_indices[0]
            crop_indices = type_indices[1]
            n_pairs = min(len(full_indices), len(crop_indices))
            if n_pairs == 0:
                continue

            full_order = torch.randperm(len(full_indices), generator=generator).tolist()
            crop_order = torch.randperm(len(crop_indices), generator=generator).tolist()
            for i in range(n_pairs):
                pairs.append((full_indices[full_order[i]], crop_indices[crop_order[i]]))

        if len(pairs) > 0:
            pair_order = torch.randperm(len(pairs), generator=generator).tolist()
            pairs = [pairs[i] for i in pair_order]
        return pairs

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        if self.num_samples == 0:
            return iter([])

        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(int(self.seed) + self._iteration)
        else:
            generator.seed()
        self._iteration += 1

        code_pairs = self._build_pairs_for_task(self._task_source_to_indices['code'], generator)
        bbox_pairs = self._build_pairs_for_task(self._task_source_to_indices['bbox'], generator)
        paired_batches = min(
            len(code_pairs) // self.pairs_per_task_per_batch,
            len(bbox_pairs) // self.pairs_per_task_per_batch,
            self.paired_batches)
        used_code_pair_count = paired_batches * self.pairs_per_task_per_batch
        used_bbox_pair_count = paired_batches * self.pairs_per_task_per_batch

        used_code = set()
        for full_idx, crop_idx in code_pairs[:used_code_pair_count]:
            used_code.add(full_idx)
            used_code.add(crop_idx)
        used_bbox = set()
        for full_idx, crop_idx in bbox_pairs[:used_bbox_pair_count]:
            used_bbox.add(full_idx)
            used_bbox.add(crop_idx)

        remaining_code = [idx for idx in self.code_indices if idx not in used_code]
        remaining_bbox = [idx for idx in self.json_indices if idx not in used_bbox]
        if len(remaining_code) > 0:
            perm = torch.randperm(len(remaining_code), generator=generator).tolist()
            remaining_code = [remaining_code[i] for i in perm]
        if len(remaining_bbox) > 0:
            perm = torch.randperm(len(remaining_bbox), generator=generator).tolist()
            remaining_bbox = [remaining_bbox[i] for i in perm]

        regular_batches = min(
            len(remaining_code) // self.half_batch,
            len(remaining_bbox) // self.half_batch,
            self.regular_batches)

        flags = [1] * paired_batches + [0] * regular_batches
        if len(flags) > 0:
            order = torch.randperm(len(flags), generator=generator).tolist()
            flags = [flags[i] for i in order]

        def iterator():
            pair_cursor = 0
            regular_cursor = 0
            for is_paired in flags:
                if is_paired:
                    start = pair_cursor * self.pairs_per_task_per_batch
                    code_chunk = code_pairs[start:start + self.pairs_per_task_per_batch]
                    bbox_chunk = bbox_pairs[start:start + self.pairs_per_task_per_batch]
                    pair_cursor += 1

                    combined = []
                    for full_idx, crop_idx in code_chunk:
                        combined.extend([full_idx, crop_idx])
                    for full_idx, crop_idx in bbox_chunk:
                        combined.extend([full_idx, crop_idx])
                else:
                    start = regular_cursor * self.half_batch
                    code_chunk = remaining_code[start:start + self.half_batch]
                    bbox_chunk = remaining_bbox[start:start + self.half_batch]
                    regular_cursor += 1
                    combined = code_chunk + bbox_chunk

                inside_order = torch.randperm(len(combined), generator=generator).tolist()
                for order_idx in inside_order:
                    yield combined[order_idx]

        return iterator()

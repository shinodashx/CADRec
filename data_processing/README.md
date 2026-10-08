# Data preprocessing

This directory contains the final preprocessing pipeline used by `data.sh`:

```text
data.sh -> batch_json2cadquery.py -> json2cadquery.py -> cadlib/
```

It converts DeepCAD-format JSON models into CadQuery programs, STL meshes,
point clouds with normals, and `SPLIT` / `STOP` bounding-box annotations for
CADRec. Alternative converters, raw datasets, generated models, and logs are
not included.

## Environment

Use a separate preprocessing environment. The pipeline was checked with
Python 3.11, PyTorch 2.5.1+cu124, and PyTorch3D 0.7.8; the remaining package
versions are recorded in `requirements.txt`.

Install compatible [PyTorch](https://pytorch.org/get-started/locally/) and
[PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
builds first. PyTorch3D provides the compiled farthest-point sampling operator
and is not installed by the requirements file. Then, from the repository root:

```bash
python -m pip install -r data_processing/requirements.txt
python -c "import cadquery, trimesh, torch; from pytorch3d.ops import sample_farthest_points"
```

The current sampler runs on CPU; no GPU index, model checkpoint, or Hugging Face
endpoint is required. Headless servers can set `MPLBACKEND=Agg`; the bundled
`cadlib` does not force a graphical Matplotlib backend.

## Input layout

Supply the source JSON dataset separately. The batch runner reads JSON files
directly inside four-digit subdirectories:

```text
input_root/
  0000/
    model_a.json
    model_b.json
  0001/
    model_c.json
```

JSON files must contain DeepCAD `entities`, `sequence`, and bounding-box
metadata, not the `bbox.json` annotations produced by this pipeline.

## Batch processing

Run a small subset in the foreground first:

```bash
python data_processing/batch_json2cadquery.py \
  --input-root /path/to/source_json \
  --output-root /path/to/processed_data \
  --folders 0000 \
  --workers 1 \
  --timeout 300
```

For a background run with a log and PID file:

```bash
bash data_processing/data.sh \
  --input-root /path/to/source_json \
  --output-root /path/to/processed_data \
  --workers 24
```

The launcher can be called from any working directory and forwards all arguments
to the batch runner. It prints the process ID and log path. Logs default to
`data_processing/datalogs/`; set `LOG_DIR` to override this directory and `PYTHON`
to select a Python executable. Relative input/output arguments are resolved from
the caller's working directory. Without explicit roots, the defaults are
`data_processing/cadmllm/` and `data_processing/cadmllmnew/`, relative to the
installed scripts rather than the current working directory.

`--folders 0000,0001` limits the input folders. Defaults are normalization to
the symmetric `[-100, 100]` range, scale `1.0`, 24 workers, one runtime thread per
worker, and a 300-second timeout per model. Use `--no-normalize` to keep source
coordinates, `--scale` to change the scale, and `--start-method spawn` when a
spawn-based multiprocessing context is needed. Run the Python entrypoint with
`--help` for all options.

**Use a dedicated output directory, separate from the input dataset.** Re-running
a model clears its existing output directory and matching sibling part directories
before regenerating them. Per-model failures do not stop the batch: inspect the
`TOTAL` summary in the log and the failure details in `<output-root>/failures.json`
(written when the run has failures). An existing failure log from an earlier run
is not cleared automatically, and the process exit code alone does not indicate
that every model succeeded.

## Single-model conversion

Use an output directory whose basename matches the input JSON stem:

```bash
python data_processing/json2cadquery.py \
  --json /path/to/source_json/0000/model_a.json \
  --output-dir /path/to/processed_data/0000/model_a
```

To generate only the CadQuery program, replace `--output-dir` with
`--output /path/to/model_a.py`.

## Outputs

Each model directory contains `<name>.py`, `<name>.stl`, `<name>.ply`,
`<name>.npy`, and `bbox.json`. NPY point clouds have shape `(8192, 6)`, dtype
`float32`, and columns `x, y, z, nx, ny, nz`.

- A single-part model has a `STOP` annotation with `model_bbox` and `steps`,
  plus `<name>_crop.ply` and `<name>_crop.npy` in the same directory.
- A multipart model has a `SPLIT` annotation with `model_bbox` and `parts`.
  Its part directories are siblings, such as `model_a_1/` and `model_a_2/`,
  not nested under `model_a/`. Each part has its own program, mesh, point cloud,
  cropped point cloud, and `STOP` annotation. With normalization enabled, each
  part is normalized independently; parent `parts` boxes remain in the parent
  model's coordinate frame.
- Bounding boxes use `[xmin, ymin, zmin, xmax, ymax, zmax]` order. Saved programs
  use rounded coordinates, while mesh generation uses full-precision geometry.

Keep custom dataset and output directories outside the repository. The default
input, output, and log directories are ignored by Git.

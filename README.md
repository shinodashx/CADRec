# CADRec

### Reconstructing a CAD Sequence Recursively with Localized Geometric Contexts

**ACM TOG (SIGGRAPH Asia 2026)**

Haoxuan Song, Bingchen Yang, Jun Xiao, Haiyong Jiang

**[Project Page](https://shinodashx.github.io/CADRec/)** · **[Paper (PDF)](https://shinodashx.github.io/CADRec/files/CADRec.pdf)** · **[Video](https://shinodashx.github.io/CADRec/#overview)**

CADRec reconstructs editable CAD sequences from point clouds through recursive part decomposition, localized geometric contexts, and local CADQuery program synthesis.

[![CADRec reconstruction showcase](docs/assets/teaser.webp)](https://shinodashx.github.io/CADRec/)

Core training and layered recursive testing code for the CADRec grounding + contrastive learning experiment.

## Main files

- `cadrec_config.yaml`: training configuration with contrastive learning and bbox grounding enabled.
- `train.py`, `train.sh`, `train_cadrec.py`: training entrypoints.
- `cadrec.py`: model, collate, contrastive loss, and bbox grounding logic.
- `cadrec_dataset.py`: CAD dataset and sampling utilities.
- `test_layer.py`, `recursive_bbox_infer.py`, `inference.py`: layered recursive inference and testing utilities.
- `yaml_config.py`: YAML configuration loading and snapshot helpers.
- [`data_processing/`](data_processing/): final JSON-to-CadQuery preprocessing pipeline, dependency versions, and usage instructions.

## Configure your own paths

The repository does not include datasets, pretrained weights, or encoder dependency
repositories. Copy the configuration and edit your private copy; `*.local.yaml` and
`*.local.yml` files are ignored by Git:

```bash
cp cadrec_config.yaml cadrec_config.local.yaml
```

Set these fields in `cadrec_config.local.yaml`:

| Setting | Meaning |
| --- | --- |
| `train.paths.data_path` | Your dataset root (example default: `./datasets`). |
| `train.paths.log_path` | Your training output/checkpoint directory (default: `./work_dirs/cadrec`). |
| `train.paths.model_path` | Your initialization checkpoint (example default: `./checkpoints/cadrec`). Set this to an existing directory. |
| `train.paths.resume` | A Trainer checkpoint to resume, including optimizer/training state; otherwise `null`. |
| `train.dataset.deepcadv2_root_subdir` | The processed dataset location beneath `data_path`, or an absolute path. |
| `train.dataset.cadrecode_v15our_root_subdir` | The CadRecode processed dataset beneath `data_path`, or an absolute path, when that dataset is selected. |
| `train.dataset.deepcad_split_json` | Your split JSON file. An absolute path is recommended; relative values are resolved from the CADRec repository directory. |

The example keeps initialization from an existing checkpoint explicit. Only set
`train.paths.model_path: null` if you intentionally want to initialize from
`common.model.base_model_path` (the public Qwen model by default), rather than a
previously trained CADRec checkpoint. Changing initialization weights is not the
same as resuming a Trainer run.

Other relative data/model/output paths are resolved from the working directory
where you launch the command, **not** from the YAML file's directory. YAML path
values are literal; use absolute paths rather than `~` or shell variables. Existing
dataset annotation caches may also contain absolute paths; regenerate those caches
when moving datasets between machines.

## Configure the point encoder

Install the dependencies for the selected point encoder in your environment. These
environment variables apply to training, single-model inference, recursive inference,
and layered evaluation; no source-code edits are needed.

**Utonia:** install its Python package, or add your Utonia repository to `PYTHONPATH`
so it is available to both the model and data-loader workers. A checkout named
`Utonia-main/` under CADRec is also recognized. The pretrained default remains
`utonia` from `Pointcept/Utonia`. To use your own local weights/cache, optionally set:

```bash
export UTONIA_CHECKPOINT="/path/to/your/utonia.pth"
export UTONIA_DOWNLOAD_ROOT="/path/to/your/model_cache"
```

Unset `UTONIA_CHECKPOINT` to use the public pretrained model. An explicitly supplied
checkpoint must exist; it is not silently replaced by downloaded weights.

**Point-BERT:** set `POINTBERT_ROOT` to your dependency checkout containing its
`utils`, `tools`, and `models` packages, and install its dependencies:

```bash
export POINTBERT_ROOT="/path/to/your/Point-BERT"
```

The fallback root is `Point-BERT/` beneath CADRec. By default, configuration and
weights are read from that root; override individual locations if needed:

| Environment variable | Default beneath `POINTBERT_ROOT` |
| --- | --- |
| `POINTBERT_CONFIG` | `cfgs/Mixup_models/Point-BERT.yaml` |
| `POINTBERT_CHECKPOINT` | `ckpt/Point-BERT.pth` |
| `POINTBERT_DVAE_CHECKPOINT` | `ckpt/dVAE.pth` |

The encoder reads the YAML's `model` section only. Point-BERT's pretraining dataset
`_base_` files are not needed, and you do not have to launch from its checkout.

Explicit encoder path arguments take precedence over environment variables.
Environment/constructor paths support `~`; relative overrides use the caller's
working directory. The selected paths are reused when the backbone is reloaded.
Point-BERT imports are deferred until that encoder is selected, so Utonia runs do
not require Point-BERT's packages. Set these variables before starting Python.

## Run training and inference

```bash
python train.py --config cadrec_config.local.yaml
bash train.sh --config cadrec_config.local.yaml
python inference.py --help
python recursive_bbox_infer.py --help
python test_layer.py --help
```

`train.sh` forwards arguments and can be called from any directory. It uses the
active `python` by default; override `PYTHON` and `LOG_DIR` to choose the executable
and launcher logs. GPU visibility and model-download endpoints are inherited from
your environment rather than forced by the launcher.

Inference already accepts your own `--checkpoint-path`, input paths, `--output-dir`,
and `--processor-path`. For example, after configuring the Utonia dependency:

```bash
python recursive_bbox_infer.py --config cadrec_config.local.yaml \
  --checkpoint-path /path/to/your/cadrec_checkpoint \
  --npy-path /path/to/your/sample.npy \
  --output-dir /path/to/your/results
```

Local configuration files, the default data/checkpoint/output directories, and
training snapshots are ignored by Git. Snapshots, dataset caches, and inference
results can record your paths; do not publish them without reviewing their contents.

Path-override regression tests run without downloading weights:
`python -m unittest discover -s tests -v` (using the model's Python dependencies).

## Project website

The project page source lives in [`docs/`](docs/) and uses plain HTML, CSS, and JavaScript, with no build step or third-party runtime dependencies. The live copy is published from [`shinodashx.github.io/dist/CADRec/`](https://github.com/shinodashx/shinodashx.github.io/tree/main/dist/CADRec) through that repository's existing GitHub Pages workflow. This keeps the URL at `https://shinodashx.github.io/CADRec/` without requiring a separate Pages site for the code repository.

To preview locally, run `python3 -m http.server 8080 --directory docs` and open the local server in your browser. Edit `docs/index.html` for paper content, `docs/style.css` for styling, and `docs/script.js` for the accessible result tabs and citation-copy interaction. Video and optimized figures are in `docs/assets/`; the paper and full-resolution figures are in `docs/files/`.

To publish an update, copy the updated contents of `docs/` into `dist/CADRec/` in the `shinodashx.github.io` repository and push its `main` branch. Do not replace the personal homepage's other files.

The paper PDF is compiled from the provided camera-ready LaTeX sources using the current `acmart` class, with duplicated post-preamble publication declarations removed for compilation. Research content is unchanged. Figure previews come from the supplied PDFs; the pipeline uses the paper's final figure. The two supplied video files are identical, so the site includes one copy with English captions.

## Citation

```bibtex
@article{song2026cadrec,
  title = {CADRec: Reconstructing a CAD Sequence Recursively with Localized Geometric Contexts},
  author = {Song, Haoxuan and Yang, Bingchen and Xiao, Jun and Jiang, Haiyong},
  journal = {ACM Transactions on Graphics},
  volume = {45},
  number = {6},
  articleno = {212},
  year = {2026},
  month = dec,
  doi = {10.1145/3842561}
}
```

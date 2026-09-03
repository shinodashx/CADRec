# CADRec

Core training and layered recursive testing code for the CADRec grounding + contrastive learning experiment.

## Main files

- `cadgen_config.yaml`: training configuration with contrastive learning and bbox grounding enabled.
- `train.py`, `train.sh`, `traincadgen0311.py`: training entrypoints.
- `cadgen0311.py`: model, collate, contrastive loss, and bbox grounding logic.
- `cadgendataset0311.py`: CAD dataset and sampling utilities.
- `test_layer.py`, `recursive_bbox_infer0311.py`, `example1.py`: layered recursive inference and testing utilities.
- `yaml_config0311.py`: YAML configuration loading and snapshot helpers.

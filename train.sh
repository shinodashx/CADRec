#!/bin/bash

LOG_DIR="/data/songhx24/Project/utonia_cadgen/utonia_grounding/train_logs"
mkdir -p $LOG_DIR

TIME_STR=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/run_${TIME_STR}.log"

export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

python train.py --config cadrec_config.yaml > "$LOG_FILE" 2>&1 &
echo $! > "${LOG_FILE%.log}.pid"

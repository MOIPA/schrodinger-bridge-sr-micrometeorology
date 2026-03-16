#!/bin/bash

HOME_DIR="/workspace"
SCRIPT_PATH="${HOME_DIR}/scripts/train_diffusion_model.py"
CONFIG_PATH="${HOME_DIR}/configs/diffusion_model.yml"

python3 ${SCRIPT_PATH} --config_path ${CONFIG_PATH} --device "cuda:0"

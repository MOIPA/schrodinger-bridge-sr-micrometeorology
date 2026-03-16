#!/bin/bash

HOME_DIR="/workspace"
SCRIPT_PATH="${HOME_DIR}/scripts/train_schrodinger_bridge_model.py"
CONFIG_PATH="${HOME_DIR}/configs/schrodinger_bridge_model.yml"

python3 ${SCRIPT_PATH} --config_path ${CONFIG_PATH} --device "cpu"

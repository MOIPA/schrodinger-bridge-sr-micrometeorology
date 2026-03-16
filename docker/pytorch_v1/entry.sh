#!/bin/bash
set -e
echo "Starting JupyterLab..."
xvfb-run -a jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='yasuda'

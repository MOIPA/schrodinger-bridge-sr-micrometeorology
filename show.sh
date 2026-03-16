#!/bin/bash
rm visualizations/*
#python visualize_samples_hongkong.py --config_path configs/schrodinger_bridge_model.yml  --model_weight_path data/DL_result/ExperimentSchrodingerBridgeModel/schrodinger_bridge_model/model_weight_0100.pth  --npz_dir ./prepare_npz/ --output_dir ./visualizations --num_samples 5
python visualize_samples_hongkong.py --config_path configs/schrodinger_bridge_model.yml  --model_weight_path data/DL_result/ExperimentSchrodingerBridgeModel/schrodinger_bridge_model/checkpoint.pth  --npz_dir ./prepare_npz/ --output_dir ./visualizations --num_samples 5

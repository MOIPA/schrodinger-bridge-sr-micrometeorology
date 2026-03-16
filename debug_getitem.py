import sys
import pathlib
import argparse
import traceback

# Add project root to the Python path
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

import torch
import torch.nn.functional as F
from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import _get_file_paths, _split_paths_into_train_valid_test
from src.dl_data.dataset_2d_tm2m import Dataset2dTemperature2m
from src.dl_data.dataset_2d_residual_tm2m import Dataset2dResidualTemperature2m

# Imports needed for model and SI initialization
from src.dl_model.model_maker import make_model
from src.dl_model.si_follmer.si_follmer_framework import StochasticInterpolantFollmer

def inspect_forward_pass():
    """
    Performs a single forward pass of the model to inspect intermediate
    tensors (xt, vt, pred_vt) and diagnose zero-loss issues.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model config file.")
    args = parser.parse_args()

    print("Loading configuration...")
    config = load_config("ExperimentSchrodingerBridgeModel", args.config_path)

    # --- Step 1: Load a single data batch ---
    print("\n--- Step 1: Loading one data batch ---")
    root_dir = str(pathlib.Path(__file__).parent.resolve())
    data_dir_path = f"{root_dir}/data/DL_data/{config.loader.dl_data_ver}"
    all_file_paths = _get_file_paths(data_dir_path, "npz")
    dict_file_paths = _split_paths_into_train_valid_test(
        all_file_paths, config.loader.train_valid_test_ratios
    )
    
    dataset_class = Dataset2dTemperature2m
    train_dataset = dataset_class(file_paths=dict_file_paths["train"], config=config.data)
    
    try:
        batch = train_dataset[0]
        # Unsqueeze to add a batch dimension of 1
        for k, v in batch.items():
            batch[k] = v.unsqueeze(0)
        print("[SUCCESS] Successfully retrieved and prepared the first item as a batch.")
    except Exception:
        print(f"\n---!!! FAILED to get item from Dataset !!!---")
        print("A precise error was caught. Full traceback below:\n")
        traceback.print_exc()
        return

    # --- Step 2: Initialize Model and Stochastic Interpolant ---
    print("\n--- Step 2: Initializing Model and SI Framework ---")
    device = torch.device("cpu")
    net = make_model(config.model).to(device)
    si = StochasticInterpolantFollmer(config=config.si, neural_net=net).to(device)
    print("[SUCCESS] Model and SI framework initialized.")

    # --- Step 3: Perform Manual Forward Pass and Calculate Loss ---
    print("\n--- Step 3: Performing manual forward pass and calculating loss ---")
    y0, y1, y_cond = batch["y0"], batch["y"], batch["x"]
    
    # The forward method of StochasticInterpolantFollmer itself calculates and returns the loss.
    calculated_loss = si(y0, y1, y_cond) # This calls si.forward()

    print("[SUCCESS] Forward pass and loss calculation complete.")

    # --- Step 4: Inspect Final Loss ---
    print("\n" + "="*50)
    print("---           FINAL LOSS INSPECTION           ---")
    print("="*50)

    print(f"Calculated MSE Loss: {calculated_loss.item():.10f}")
    if calculated_loss.item() < 1e-9:
        print("--> RESULT: Loss is ZERO (or very close to it).")
    else:
        print("--> RESULT: Loss is NON-ZERO. The core calculation is working.")
        
    print("\n" + "="*50)
    print("---           INSPECTION COMPLETE             ---")
    print("="*50)


if __name__ == '__main__':
    inspect_forward_pass()

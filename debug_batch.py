import sys
import pathlib
import argparse

# Add project root to the Python path
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

import torch
from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers

def inspect_batch():
    """
    Loads one batch of data from the dataloader and prints statistics
    to debug potential data pipeline issues.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model config file.")
    args = parser.parse_args()

    print("Loading configuration...")
    config = load_config("ExperimentSchrodingerBridgeModel", args.config_path)

    print("Creating DataLoader...")
    # We don't need distributed training for this debug script
    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=str(pathlib.Path(__file__).parent.resolve()),
        loader_config=config.loader,
        dataset_config=config.data,
        world_size=None,
        rank=None,
        train_valid_test_kinds=["train"],
    )
    train_loader = dict_loaders["train"]
    print("DataLoader created.")

    try:
        # Get one batch of data
        print("\nFetching one batch of data...")
        batch = next(iter(train_loader))
        print("Batch fetched successfully.")
    except Exception as e:
        print(f"\n---!!! ERROR !!!---")
        print(f"Failed to fetch a batch from the DataLoader: {e}")
        print("This indicates a problem during data loading or preprocessing in the Dataset class.")
        return

    # Inspect the tensors in the batch
    x = batch.get('x')
    y = batch.get('y')
    y0 = batch.get('y0')

    if x is None or y is None or y0 is None:
        print("\n---!!! ERROR !!!---")
        print(f"Batch is missing one or more required keys ('x', 'y', 'y0'). Available keys: {list(batch.keys())}")
        return

    print("\n" + "="*50)
    print("---              TENSOR INSPECTION              ---")
    print("="*50)

    print(f"\n[Shapes and Types]")
    print(f"x (conditional input) shape: {x.shape}, dtype: {x.dtype}")
    print(f"y (HR target)         shape: {y.shape}, dtype: {y.dtype}")
    print(f"y0 (LR input)          shape: {y0.shape}, dtype: {y0.dtype}")

    print(f"\n[Tensor Statistics]")
    print(f"x (conditional):  min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"y (HR target):    min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")
    print(f"y0 (LR input):    min={y0.min():.4f}, max={y0.max():.4f}, mean={y0.mean():.4f}, std={y0.std():.4f}")

    # Check for meaningful difference between y and y0
    diff = torch.abs(y - y0).mean()
    print(f"\n[Sanity Check]")
    print(f"Mean absolute difference between y (HR) and y0 (LR): {diff:.6f}")
    
    if diff < 1e-6:
        print("--> WARNING: y and y0 are nearly identical! The model has nothing to learn.")
    else:
        print("--> OK: y and y0 are different, which is correct.")
        
    print("\n" + "="*50)
    print("---           INSPECTION COMPLETE             ---")
    print("="*50)


if __name__ == '__main__':
    inspect_batch()

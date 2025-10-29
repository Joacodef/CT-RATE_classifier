# scripts/check_feature_dataloader.py
"""
Quick smoke script to validate feature-based datasets and dataloaders.
Runs a single batch through the FeatureDataset and the MLP creation to ensure
loading, batching and forward pass work (useful to debug num_workers / DataLoader issues).

Usage:
    python .\scripts\check_feature_dataloader.py --config .\configs\config.yaml

This script forces num_workers=0 to avoid Windows worker spawn issues and prints
shapes/types for inspection.
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import torch
from src.config import load_config
from src.data.dataset import FeatureDataset
from torch.utils.data import DataLoader
from src.training.trainer import create_mlp_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', choices=['train','valid'], default='train')
    parser.add_argument('--fold', type=int, default=None, help='Optional fold number to select train_fold_{N}.csv and valid_fold_{N}.csv')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Force feature-based workflow
    if not hasattr(cfg, 'workflow'):
        from types import SimpleNamespace
        cfg.workflow = SimpleNamespace()
    cfg.workflow.mode = 'feature-based'
    # Ensure feature config is present
    if not hasattr(cfg.workflow, 'feature_config'):
        from types import SimpleNamespace
        cfg.workflow.feature_config = SimpleNamespace()
    cfg.workflow.feature_config.feature_dir = cfg.paths.feature_dir

    # Create dataset
    feat_root = Path(cfg.workflow.feature_config.feature_dir)
    if (feat_root / args.split).exists():
        feat_dir = feat_root / args.split
    else:
        feat_dir = feat_root

    print(f"Using feature dir: {feat_dir}")

    # Load CSVs: trainer expects cfg.paths.data_subsets.train to be a Path under data_dir
    train_df_path = cfg.paths.data_subsets.train
    valid_df_path = cfg.paths.data_subsets.valid

    # If a fold is provided, override the subset paths to use train_fold_{fold}.csv etc.
    if args.fold is not None:
        try:
            base = Path(cfg.paths.data_subsets.train).parent
            train_df_path = base / f"train_fold_{args.fold}.csv"
            valid_df_path = base / f"valid_fold_{args.fold}.csv"
        except Exception:
            # fall back to original paths if something unexpected
            pass

    print(f"Train CSV: {train_df_path}")
    print(f"Valid CSV: {valid_df_path}")

    df_path = train_df_path if args.split == 'train' else valid_df_path
    # Create dataset by merging split volumes with the unified labels file (same logic as trainer)
    import pandas as pd
    split_df = pd.read_csv(Path(cfg.paths.data_dir) / df_path)
    all_labels_df = pd.read_csv(Path(cfg.paths.data_dir) / cfg.paths.labels.all)
    merged = pd.merge(split_df[['VolumeName']], all_labels_df, on='VolumeName', how='inner')
    if merged.empty:
        raise RuntimeError(f"Merged DataFrame is empty after joining split {df_path} with labels {cfg.paths.labels.all}")
    # Use small subset for quick smoke test
    use_df = merged.head(32).reset_index(drop=True)

    dataset = FeatureDataset(use_df, feat_dir, cfg.pathologies.columns, preload_to_ram=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print(f"Dataset length: {len(dataset)}")
    print(f"Trying to fetch one batch from DataLoader (num_workers=0)...")

    try:
        batch = next(iter(loader))
    except Exception as e:
        print("Error fetching batch:", e)
        raise

    print("Batch keys:", list(batch.keys()))
    img = batch['image']
    lbl = batch['label']
    print(type(img), getattr(img,'shape',None))
    print(type(lbl), getattr(lbl,'shape',None))

    # Create MLP and run a forward pass
    model = create_mlp_classifier(cfg)
    model.eval()
    with torch.no_grad():
        # flatten features if needed
        if isinstance(img, torch.Tensor):
            inp = img.float()
        else:
            # try to convert
            inp = torch.stack([x.float() if isinstance(x, torch.Tensor) else torch.tensor(x).float() for x in img])

        # If input is (B, *) good. Else try to flatten per sample
        if inp.dim() > 2:
            inp = inp.view(inp.size(0), -1)

        out = model(inp)
        print("Forward pass output shape:", out.shape)

    print("Smoke test completed successfully.")


if __name__ == '__main__':
    main()

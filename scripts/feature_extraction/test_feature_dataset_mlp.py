# scripts/feature_extraction/test_feature_dataset_mlp.py

import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


from src.data.dataset import FeatureDataset
from src.training.trainer import create_mlp_classifier

def main():
    # --- Device selection ---
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # --- Load config ---
    from src.config import load_config
    import argparse


    parser = argparse.ArgumentParser(description="Test MLP classifier on extracted features.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    parser.add_argument("--feature-dir", type=str, required=True, help="Directory with .pt feature files.")
    parser.add_argument("--label-csv", type=str, required=False, help="Path to the label CSV file (with labels). If not provided, will use config.paths.labels.all.")
    parser.add_argument("--master-list", type=str, required=False, help="Path to the master list CSV file (with VolumeNames to use). If not provided, will use config.paths.full_dataset_csv (joined with config.paths.data_dir if relative).")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of epochs to train (overrides config).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config).")
    parser.add_argument("--valid-size", type=float, default=0.2, help="Fraction of data to use for validation (default: 0.2)")
    args = parser.parse_args()

    config = load_config(args.config)

    feature_dir = args.feature_dir
    pathology_columns = config.pathologies.columns
    batch_size = args.batch_size if args.batch_size is not None else config.training.batch_size
    num_epochs = args.num_epochs if args.num_epochs is not None else config.training.num_epochs
    learning_rate = config.training.learning_rate if hasattr(config.training, 'learning_rate') else 1e-3
    # Determine label_csv and master_list from args or config
    label_csv = args.label_csv
    if not label_csv:
        label_csv = config.paths.labels.all
    master_list = args.master_list
    if not master_list and hasattr(config.paths, 'full_dataset_csv'):
        # Join with data_dir if not absolute
        import os
        master_list_candidate = config.paths.full_dataset_csv
        if not os.path.isabs(master_list_candidate):
            master_list = os.path.join(config.paths.data_dir, master_list_candidate)
        else:
            master_list = master_list_candidate
    valid_size = args.valid_size

    if not os.path.exists(label_csv):
        print(f"Label CSV not found: {label_csv}")
        return

    df = pd.read_csv(label_csv)
    # If master list is provided, filter label DataFrame to only those VolumeNames
    if master_list:
        master_df = pd.read_csv(master_list)
        # Clean volume names in master list to match label file
        def clean_name(name):
            return str(name).replace('.nii.gz', '').replace('.nii', '')
        master_volume_names = set(master_df['VolumeName'].apply(clean_name))
        df['VolumeName_clean'] = df['VolumeName'].apply(clean_name)
        df = df[df['VolumeName_clean'].isin(master_volume_names)].reset_index(drop=True)
        print(f"After filtering with master list: {len(df)} samples remain.")

    # List all feature files and extract volume names
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.pt')]
    feature_volume_names = set(os.path.splitext(f)[0] for f in feature_files)

    # Clean volume names in DataFrame to match feature file naming
    def clean_name(name):
        return str(name).replace('.nii.gz', '').replace('.nii', '')
    df['VolumeName_clean'] = df['VolumeName'].apply(clean_name)

    # Filter DataFrame to only those with a corresponding feature file
    df_filtered = df[df['VolumeName_clean'].isin(feature_volume_names)].reset_index(drop=True)
    if len(df_filtered) == 0:
        print("No matching features found for any volumes in the master label CSV.")
        return
    print(f"Found {len(df_filtered)} samples with matching features.")



    import numpy as np
    from copy import deepcopy
    n_runs = 5
    aucs = []
    seeds = [42, 123, 999, 2025, 31415]
    for run_idx, seed in enumerate(seeds):
        print(f"\n========== Run {run_idx+1}/{n_runs} (seed={seed}) ==========")
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Shuffle df_filtered before stratified split to get different splits
        df_shuffled = df_filtered.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Multi-label stratified split using IterativeStratification
        from skmultilearn.model_selection import IterativeStratification
        disease_cols = pathology_columns
        X = df_shuffled.index.values.reshape(-1, 1)
        y = df_shuffled[disease_cols].to_numpy()
        stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1-valid_size, valid_size])
        idx1, idx2 = next(stratifier.split(X, y))
        # Ensure train set is the larger split
        if len(idx1) >= len(idx2):
            train_idx, valid_idx = idx1, idx2
        else:
            train_idx, valid_idx = idx2, idx1
        train_df = df_shuffled.iloc[train_idx].reset_index(drop=True)
        valid_df = df_shuffled.iloc[valid_idx].reset_index(drop=True)
        print(f"Using {len(train_df)} samples for train split, {len(valid_df)} for valid split (multi-label stratified).")

        train_dataset = FeatureDataset(train_df, Path(feature_dir), pathology_columns)
        valid_dataset = FeatureDataset(valid_df, Path(feature_dir), pathology_columns)

        # --- MONAI CacheDataset integration ---
        from monai.data import CacheDataset
        use_cache = getattr(config, 'cache', None) and getattr(config.cache, 'use_cache', False)
        memory_rate = getattr(config.cache, 'memory_rate', 0.0) if getattr(config, 'cache', None) else 0.0
        if use_cache:
            print(f"[INFO] Wrapping datasets with MONAI CacheDataset (memory_rate={memory_rate})")
            train_dataset = CacheDataset(data=train_dataset, transform=None, cache_rate=memory_rate, num_workers=0)
            valid_dataset = CacheDataset(data=valid_dataset, transform=None, cache_rate=memory_rate, num_workers=0)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # --- Create Enhanced MLP ---
        sample_feature = train_dataset[0]["image"]
        in_features = sample_feature.numel() if hasattr(sample_feature, 'numel') else len(sample_feature)
        out_features = len(pathology_columns)
        import torch.nn as nn
        class EnhancedMLP(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(128, out_features)
                )
            def forward(self, x):
                return self.model(x)

        model = EnhancedMLP(in_features, out_features).to(device)
        model.train()

        # --- Training setup ---
        loss_type = getattr(config.loss_function, 'type', 'BCEWithLogitsLoss') if hasattr(config, 'loss_function') else 'BCEWithLogitsLoss'
        if loss_type == 'FocalLoss':
            from monai.losses import FocalLoss
            focal_params = getattr(config.loss_function, 'focal_loss', {}) if hasattr(config.loss_function, 'focal_loss') else {}
            # Support both dict and SimpleNamespace
            if isinstance(focal_params, dict):
                alpha = focal_params.get('alpha', 0.25)
                gamma = focal_params.get('gamma', 1.0)
            else:
                alpha = getattr(focal_params, 'alpha', 0.25)
                gamma = getattr(focal_params, 'gamma', 1.0)
            criterion = FocalLoss(to_onehot_y=False, alpha=alpha, gamma=gamma, reduction='mean')
            print(f"[INFO] Using FocalLoss (alpha={alpha}, gamma={gamma})")
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
            print("[INFO] Using BCEWithLogitsLoss")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.training.weight_decay if hasattr(config.training, 'weight_decay') else 0.0)

        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=1e-6)

        from tqdm import tqdm
        from monai.metrics import ROCAUCMetric, FBetaScore
        from monai.metrics.f_beta_score import compute_f_beta_score
        from src.training.utils import EarlyStopping
        patience = getattr(config.training, 'early_stopping_patience', 5)
        early_stopping = EarlyStopping(patience=patience, mode='max', min_delta=0.0001)
        best_valid_auc = float('-inf')
        try:
            for epoch in range(num_epochs):
                model.train()
                total_train_loss = 0.0
                train_auc_metric = ROCAUCMetric()
                train_f1_metric = FBetaScore(beta=1.0)
                train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
                for batch in train_bar:
                    features = batch["image"].float().to(device)
                    labels = batch["label"].float().to(device)
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item() * features.size(0)
                    train_bar.set_postfix(loss=loss.item())
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).int()
                    labels_int = labels.cpu().int()
                    # Ensure shape match
                    if preds.shape != labels_int.shape:
                        preds = preds.view_as(labels_int)
                    train_auc_metric(probs.cpu(), labels_int)
                    train_f1_metric(preds.cpu(), labels_int)
                avg_train_loss = total_train_loss / len(train_dataset)
                train_auc_macro = train_auc_metric.aggregate(average="macro")
                train_auc_micro = train_auc_metric.aggregate(average="micro")
                cm_buffer = train_f1_metric.get_buffer()
                f1_macro_per_class = compute_f_beta_score(cm_buffer.nanmean(dim=0), beta=1.0)
                train_f1_macro = f1_macro_per_class.nanmean().item()
                total_cm = cm_buffer.sum(dim=(0, 1))
                train_f1_micro = compute_f_beta_score(total_cm, beta=1.0).item()
                train_auc_metric.reset()
                train_f1_metric.reset()

                model.eval()
                total_valid_loss = 0.0
                valid_auc_metric = ROCAUCMetric()
                valid_f1_metric = FBetaScore(beta=1.0)
                valid_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", leave=False)
                with torch.no_grad():
                    for batch in valid_bar:
                        features = batch["image"].float().to(device)
                        labels = batch["label"].float().to(device)
                        outputs = model(features)
                        loss = criterion(outputs, labels)
                        total_valid_loss += loss.item() * features.size(0)
                        valid_bar.set_postfix(loss=loss.item())
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).int()
                        labels_int = labels.cpu().int()
                        if preds.shape != labels_int.shape:
                            preds = preds.view_as(labels_int)
                        valid_auc_metric(probs.cpu(), labels_int)
                        valid_f1_metric(preds.cpu(), labels_int)
                avg_valid_loss = total_valid_loss / len(valid_dataset)
                valid_auc_macro = valid_auc_metric.aggregate(average="macro")
                valid_auc_micro = valid_auc_metric.aggregate(average="micro")
                cm_buffer = valid_f1_metric.get_buffer()
                f1_macro_per_class = compute_f_beta_score(cm_buffer.nanmean(dim=0), beta=1.0)
                valid_f1_macro = f1_macro_per_class.nanmean().item()
                total_cm = cm_buffer.sum(dim=(0, 1))
                valid_f1_micro = compute_f_beta_score(total_cm, beta=1.0).item()
                valid_auc_metric.reset()
                valid_f1_metric.reset()

                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Valid Loss: {avg_valid_loss:.4f} | Train AUC: {train_auc_macro:.4f} F1: {train_f1_macro:.4f} | Valid AUC: {valid_auc_macro:.4f} F1: {valid_f1_macro:.4f}")

                scheduler.step(valid_auc_macro)

                if valid_auc_macro > best_valid_auc:
                    best_valid_auc = valid_auc_macro
                if early_stopping(valid_auc_macro):
                    print(f"[INFO] Early stopping triggered at epoch {epoch+1}. Best valid macro AUC: {best_valid_auc:.4f}")
                    break
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user (Ctrl+C). Exiting gracefully...")
            return

        print(f"[RESULT] Run {run_idx+1}: Best valid macro AUC = {best_valid_auc:.4f}")
        aucs.append(best_valid_auc)

    aucs = np.array(aucs)
    print(f"\n========== Summary over {n_runs} runs ==========")
    print(f"Mean best valid macro AUC: {aucs.mean():.4f} ± {aucs.std():.4f}")
    print("MLP feature test completed successfully.")

if __name__ == "__main__":
    main()
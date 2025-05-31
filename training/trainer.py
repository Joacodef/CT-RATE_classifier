# Standard library imports
import time
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any # Added Dict, Any
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import wandb # Direct import of wandb

# Internal imports - config
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config

# Internal imports - models
from models.resnet3d import resnet18_3d, resnet34_3d
from models.densenet3d import densenet121_3d, densenet169_3d
from models.losses import FocalLoss

# Internal imports - data
from data.dataset import CTDataset3D

# Internal imports - training utilities
from .metrics import compute_metrics
from .utils import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint
)

# Internal imports - evaluation
from evaluation.reporting import generate_final_report

# Internal imports - utils
from utils.torch_utils import setup_torch_optimizations

# Get logger
logger = logging.getLogger(__name__)


def create_model(config: Config) -> nn.Module:
    """Create and return the 3D model"""
    # [Function body from your provided code]
    if config.MODEL_TYPE == "resnet3d":
        model = resnet18_3d(
            num_classes=config.NUM_PATHOLOGIES,
            use_checkpointing=config.GRADIENT_CHECKPOINTING
        )
        logger.info("Created ResNet3D-18 model with gradient checkpointing" 
                   if config.GRADIENT_CHECKPOINTING else "Created ResNet3D-18 model")
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")
    return model


def load_and_prepare_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare training and validation data with validation"""
    # [Function body from your provided code]
    logger.info("Loading DataFrames...")
    try:
        train_volumes = pd.read_csv(config.SELECTED_TRAIN_VOLUMES_CSV)[['VolumeName']]
        valid_volumes = pd.read_csv(config.SELECTED_VALID_VOLUMES_CSV)[['VolumeName']]
        train_labels = pd.read_csv(config.TRAIN_LABELS_CSV)
        valid_labels = pd.read_csv(config.VALID_LABELS_CSV)
        train_df = pd.merge(train_volumes, train_labels, on='VolumeName', how='inner')
        valid_df = pd.merge(valid_volumes, valid_labels, on='VolumeName', how='inner')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required CSV file not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
    if train_df.empty or valid_df.empty:
        raise ValueError("Training or validation dataframe is empty")
    for df, name in [(train_df, "training"), (valid_df, "validation")]:
        missing_cols = [col for col in config.PATHOLOGY_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing pathology columns in {name} data: {missing_cols}")
    train_df[config.PATHOLOGY_COLUMNS] = train_df[config.PATHOLOGY_COLUMNS].fillna(0)
    valid_df[config.PATHOLOGY_COLUMNS] = valid_df[config.PATHOLOGY_COLUMNS].fillna(0)
    train_df[config.PATHOLOGY_COLUMNS] = train_df[config.PATHOLOGY_COLUMNS].astype(int)
    valid_df[config.PATHOLOGY_COLUMNS] = valid_df[config.PATHOLOGY_COLUMNS].astype(int)
    logger.info(f"Data loaded: {len(train_df)} training, {len(valid_df)} validation samples")
    for pathology in config.PATHOLOGY_COLUMNS:
        train_pos = train_df[pathology].sum()
        valid_pos = valid_df[pathology].sum()
        logger.info(f"{pathology}: {train_pos}/{len(train_df)} train positive, "
                   f"{valid_pos}/{len(valid_df)} valid positive")
    return train_df, valid_df


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch,
                total_epochs, gradient_accumulation_steps=1, use_amp=False):
    """Train for one epoch with gradient accumulation and mixed precision support"""
    # [Function body from your provided code]
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if hasattr(torch.cuda.amp, 'bfloat16') else torch.float16):
                outputs = model(pixel_values)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
        else:
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * gradient_accumulation_steps
        if batch_idx % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{total_epochs}] "
                       f"Batch [{batch_idx}/{num_batches}] "
                       f"Loss: {loss.item() * gradient_accumulation_steps:.4f}")
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    # [Function body from your provided code]
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        outputs = model(pixel_values)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_predictions.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return avg_loss, all_predictions, all_labels


def train_model(config: Config):
    """Main training function with checkpoint resuming capability and wandb integration"""
    setup_torch_optimizations()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize wandb_run to None. It will be populated if wandb.init() is successful.
    wandb_run = None
    try:
        # Prepare configuration dictionary for wandb.init
        wandb_config_payload = {
            "learning_rate": config.LEARNING_RATE,
            "architecture": config.MODEL_TYPE,
            "target_shape_dhw": config.TARGET_SHAPE_DHW,
            "target_spacing_xyz": config.TARGET_SPACING.tolist(),
            "clip_hu_min": config.CLIP_HU_MIN,
            "clip_hu_max": config.CLIP_HU_MAX,
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
            "weight_decay": config.WEIGHT_DECAY,
            "num_workers": config.NUM_WORKERS,
            "pin_memory": config.PIN_MEMORY,
            "gradient_checkpointing": config.GRADIENT_CHECKPOINTING,
            "mixed_precision": config.MIXED_PRECISION,
            "use_bf16": config.USE_BF16,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "use_cache": config.USE_CACHE,
            "num_pathologies": config.NUM_PATHOLOGIES,
            "pathology_columns": config.PATHOLOGY_COLUMNS,
            "output_dir": str(config.OUTPUT_DIR),
            "resume_from_checkpoint": str(config.RESUME_FROM_CHECKPOINT) if config.RESUME_FROM_CHECKPOINT else None,
        }
        # Attempt to initialize Weights & Biases.
        # If wandb (the module) was not imported successfully (e.g., not installed),
        # this line will raise a NameError for 'wandb'.
        wandb_run = wandb.init(
            project="ct_classifier",
            config=wandb_config_payload
        )
        logger.info(f"Weights & Biases initialized successfully. Run name: {wandb_run.name}")

    except Exception as e:
        # Catches errors specifically from wandb.init() (e.g., network, API key)
        # or if 'wandb' module itself was not imported.
        logger.error(f"Failed to initialize Weights & Biases: {e}. Training will continue without wandb logging.")
        # wandb_run remains None if initialization failed.

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, valid_df = load_and_prepare_data(config)
    
    train_dataset = CTDataset3D(
        train_df, config.TRAIN_IMG_DIR, config.PATHOLOGY_COLUMNS,
        config.TARGET_SPACING, config.TARGET_SHAPE_DHW,
        config.CLIP_HU_MIN, config.CLIP_HU_MAX,
        use_cache=config.USE_CACHE, cache_dir=config.CACHE_DIR,
        augment=True
    )
    valid_dataset = CTDataset3D(
        valid_df, config.VALID_IMG_DIR, config.PATHOLOGY_COLUMNS,
        config.TARGET_SPACING, config.TARGET_SHAPE_DHW,
        config.CLIP_HU_MIN, config.CLIP_HU_MAX,
        use_cache=config.USE_CACHE, cache_dir=config.CACHE_DIR,
        augment=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        persistent_workers=config.NUM_WORKERS > 0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        persistent_workers=config.NUM_WORKERS > 0
    )
    
    model = create_model(config).to(device)

    # Watch the model with wandb if wandb_run was successfully initialized.
    if wandb_run:
        try:
            wandb.watch(model, log="gradients", log_freq=100)
            logger.info("wandb.watch() initiated for model.")
        except Exception as e:
            logger.error(f"Error during wandb.watch(): {e}")

    pos_weights = []
    for col in config.PATHOLOGY_COLUMNS:
        pos_count = train_df[col].sum()
        neg_count = len(train_df) - pos_count
        weight = neg_count / (pos_count + 1e-6)
        pos_weights.append(min(weight, 10.0))
    pos_weight = torch.tensor(pos_weights, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scaler = None
    if config.MIXED_PRECISION:
        scaler = torch.cuda.amp.GradScaler()
    
    start_epoch = 0
    best_auc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'valid_loss': [], 'metrics': []}
    
    if config.RESUME_FROM_CHECKPOINT and Path(config.RESUME_FROM_CHECKPOINT).exists():
        logger.info(f"Resuming from checkpoint: {config.RESUME_FROM_CHECKPOINT}")
        try:
            checkpoint_epoch, checkpoint_metrics_loaded = load_checkpoint(
                config.RESUME_FROM_CHECKPOINT, model, optimizer, scaler
            )
            start_epoch = checkpoint_epoch + 1
            history_path = config.OUTPUT_DIR / 'training_history.json'
            if history_path.exists():
                with open(history_path, 'r') as f: history = json.load(f)
                if len(history['train_loss']) > checkpoint_epoch + 1:
                    history['train_loss'] = history['train_loss'][:checkpoint_epoch + 1]
                    history['valid_loss'] = history['valid_loss'][:checkpoint_epoch + 1]
                    history['metrics'] = history['metrics'][:checkpoint_epoch + 1]
                if history['metrics']:
                    for i, metrics_item in enumerate(history['metrics']):
                        if metrics_item['roc_auc_macro'] > best_auc:
                            best_auc = metrics_item['roc_auc_macro']
                            best_epoch = i
            logger.info(f"Resumed from epoch {checkpoint_epoch + 1}")
            logger.info(f"Best AUC so far: {best_auc:.4f} at epoch {best_epoch + 1}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting training from scratch")
            start_epoch = 0
            
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS - start_epoch, eta_min=1e-6
    )
    for _ in range(start_epoch): scheduler.step()
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode='max')
    
    logger.info(f"Starting training from epoch {start_epoch + 1}...")
    metrics_for_loop: Dict[str, Any] = {} # Holds metrics from the last completed epoch in the loop

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, config.NUM_EPOCHS, config.GRADIENT_ACCUMULATION_STEPS,
            config.MIXED_PRECISION
        )
        valid_loss, predictions, labels = validate_epoch(
            model, valid_loader, criterion, device
        )
        metrics_for_loop = compute_metrics(predictions, labels, config.PATHOLOGY_COLUMNS)
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        logger.info(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}] Time: {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        logger.info(f"Valid AUC (macro): {metrics_for_loop['roc_auc_macro']:.4f}, F1 (macro): {metrics_for_loop['f1_macro']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['metrics'].append(metrics_for_loop)

        # Log to Weights & Biases if wandb_run is active
        if wandb_run:
            log_payload = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                **metrics_for_loop # Spread all computed metrics
            }
            try:
                wandb_run.log(log_payload)
            except Exception as e:
                logger.error(f"Failed to log metrics to wandb: {e}")
        
        history_path = config.OUTPUT_DIR / 'training_history.json'
        with open(history_path, 'w') as f: json.dump(history, f, indent=2)
        
        if metrics_for_loop['roc_auc_macro'] > best_auc:
            best_auc = metrics_for_loop['roc_auc_macro']
            best_epoch = epoch
            best_model_path = config.OUTPUT_DIR / 'best_model.pth'
            save_checkpoint(model, optimizer, scaler, epoch, metrics_for_loop, best_model_path)
            logger.info(f"New best model saved with AUC: {best_auc:.4f}")
            
        if early_stopping(metrics_for_loop['roc_auc_macro']):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 5 == 0:
            checkpoint_path = config.OUTPUT_DIR / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, scaler, epoch, metrics_for_loop, checkpoint_path)
            
        last_checkpoint_path = config.OUTPUT_DIR / 'last_checkpoint.pth'
        save_checkpoint(model, optimizer, scaler, epoch, metrics_for_loop, last_checkpoint_path)
        logger.info(f"Saved checkpoint at epoch {epoch+1}")

    # Determine metrics for the final model save.
    final_metrics_to_save = metrics_for_loop
    # If resuming and loop didn't run, use metrics from loaded checkpoint if available.
    if not final_metrics_to_save and 'checkpoint_metrics_loaded' in locals() and checkpoint_metrics_loaded:
        final_metrics_to_save = checkpoint_metrics_loaded
    elif not final_metrics_to_save: # Still empty (no training, no resume with metrics)
        final_metrics_to_save = {}

    final_model_path = config.OUTPUT_DIR / 'final_model.pth'
    # 'epoch' might not be defined if the loop didn't run.
    last_trained_epoch = epoch if 'epoch' in locals() else start_epoch -1
    save_checkpoint(model, optimizer, scaler, last_trained_epoch, final_metrics_to_save, final_model_path)
    
    history_path = config.OUTPUT_DIR / 'training_history.json'
    with open(history_path, 'w') as f: json.dump(history, f, indent=2)
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best model: Epoch {best_epoch+1} with AUC {best_auc:.4f}")
    generate_final_report(history, config)
    
    # Finish the wandb run if it was initialized
    if wandb_run:
        try:
            wandb_run.finish()
            logger.info("Weights & Biases run finished.")
        except Exception as e:
            logger.error(f"Error finishing wandb run: {e}")
            
    return model, history
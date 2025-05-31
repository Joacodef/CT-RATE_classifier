# Standard library imports
import time
import json
import logging
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Internal imports - config (desde la raíz)
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config

# Internal imports - models
from models.resnet3d import resnet18_3d, resnet34_3d
from models.densenet3d import densenet121_3d, densenet169_3d
from models.losses import FocalLoss

# Internal imports - data
from data.dataset import CTDataset3D

# Internal imports - training utilities (imports relativos)
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
    
    logger.info("Loading DataFrames...")
    
    try:
        # Load selected volumes
        train_volumes = pd.read_csv(config.SELECTED_TRAIN_VOLUMES_CSV)[['VolumeName']]
        valid_volumes = pd.read_csv(config.SELECTED_VALID_VOLUMES_CSV)[['VolumeName']]
        
        # Load labels
        train_labels = pd.read_csv(config.TRAIN_LABELS_CSV)
        valid_labels = pd.read_csv(config.VALID_LABELS_CSV)
        
        # Merge
        train_df = pd.merge(train_volumes, train_labels, on='VolumeName', how='inner')
        valid_df = pd.merge(valid_volumes, valid_labels, on='VolumeName', how='inner')
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required CSV file not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
    
    # Validate data
    if train_df.empty or valid_df.empty:
        raise ValueError("Training or validation dataframe is empty")
    
    # Check for required columns
    for df, name in [(train_df, "training"), (valid_df, "validation")]:
        missing_cols = [col for col in config.PATHOLOGY_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing pathology columns in {name} data: {missing_cols}")
    
    # Fill NaN values with 0 (assuming missing labels are negative)
    train_df[config.PATHOLOGY_COLUMNS] = train_df[config.PATHOLOGY_COLUMNS].fillna(0)
    valid_df[config.PATHOLOGY_COLUMNS] = valid_df[config.PATHOLOGY_COLUMNS].fillna(0)
    
    # Convert to int
    train_df[config.PATHOLOGY_COLUMNS] = train_df[config.PATHOLOGY_COLUMNS].astype(int)
    valid_df[config.PATHOLOGY_COLUMNS] = valid_df[config.PATHOLOGY_COLUMNS].astype(int)
    
    logger.info(f"Data loaded: {len(train_df)} training, {len(valid_df)} validation samples")
    
    # Log class distribution
    for pathology in config.PATHOLOGY_COLUMNS:
        train_pos = train_df[pathology].sum()
        valid_pos = valid_df[pathology].sum()
        logger.info(f"{pathology}: {train_pos}/{len(train_df)} train positive, "
                   f"{valid_pos}/{len(valid_df)} valid positive")
    
    return train_df, valid_df


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch,
                total_epochs, gradient_accumulation_steps=1, use_amp=False):
    """Train for one epoch with gradient accumulation and mixed precision support"""
    
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        # Forward pass
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if hasattr(torch.cuda.amp, 'bfloat16') else torch.float16):
                outputs = model(pixel_values)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
        else:
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Progress logging
        if batch_idx % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{total_epochs}] "
                       f"Batch [{batch_idx}/{num_batches}] "
                       f"Loss: {loss.item() * gradient_accumulation_steps:.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
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
    """Main training function with checkpoint resuming capability"""
    # Setup
    setup_torch_optimizations()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_df, valid_df = load_and_prepare_data(config)
    
    # Create datasets
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
    
    # Create dataloaders
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
    
    # Create model
    model = create_model(config).to(device)
    
    # Calculate class weights for loss function
    pos_weights = []
    for col in config.PATHOLOGY_COLUMNS:
        pos_count = train_df[col].sum()
        neg_count = len(train_df) - pos_count
        weight = neg_count / (pos_count + 1e-6)  # Avoid division by zero
        pos_weights.append(min(weight, 10.0))  # Cap weights at 10
    
    pos_weight = torch.tensor(pos_weights, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Mixed precision training
    scaler = None
    if config.MIXED_PRECISION:
        scaler = torch.cuda.amp.GradScaler()
    
    # Initialize training variables
    start_epoch = 0
    best_auc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [],
        'valid_loss': [],
        'metrics': []
    }
    
    # Check if resuming from checkpoint
    if config.RESUME_FROM_CHECKPOINT and Path(config.RESUME_FROM_CHECKPOINT).exists():
        logger.info(f"Resuming from checkpoint: {config.RESUME_FROM_CHECKPOINT}")
        
        try:
            checkpoint_epoch, checkpoint_metrics = load_checkpoint(
                config.RESUME_FROM_CHECKPOINT, model, optimizer, scaler
            )
            start_epoch = checkpoint_epoch + 1
            
            # Load training history if it exists
            history_path = config.OUTPUT_DIR / 'training_history.json'
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                # Truncate history to checkpoint epoch if needed
                if len(history['train_loss']) > checkpoint_epoch + 1:
                    history['train_loss'] = history['train_loss'][:checkpoint_epoch + 1]
                    history['valid_loss'] = history['valid_loss'][:checkpoint_epoch + 1]
                    history['metrics'] = history['metrics'][:checkpoint_epoch + 1]
                
                # Find best AUC from history
                if history['metrics']:
                    for i, metrics in enumerate(history['metrics']):
                        if metrics['roc_auc_macro'] > best_auc:
                            best_auc = metrics['roc_auc_macro']
                            best_epoch = i
            
            logger.info(f"Resumed from epoch {checkpoint_epoch + 1}")
            logger.info(f"Best AUC so far: {best_auc:.4f} at epoch {best_epoch + 1}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting training from scratch")
            start_epoch = 0
    
    # Learning rate scheduler (adjust for resumed training)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS - start_epoch, eta_min=1e-6
    )
    
    # If resuming, adjust scheduler to correct epoch
    for _ in range(start_epoch):
        scheduler.step()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode='max')
    
    # Training loop
    logger.info(f"Starting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, config.NUM_EPOCHS, config.GRADIENT_ACCUMULATION_STEPS,
            config.MIXED_PRECISION
        )
        
        # Validate
        valid_loss, predictions, labels = validate_epoch(
            model, valid_loader, criterion, device
        )
        
        # Compute metrics
        metrics = compute_metrics(predictions, labels, config.PATHOLOGY_COLUMNS)
        
        # Update scheduler
        scheduler.step()
        
        # Log results
        epoch_time = time.time() - epoch_start_time
        logger.info(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}] "
                   f"Time: {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        logger.info(f"Valid AUC (macro): {metrics['roc_auc_macro']:.4f}, "
                   f"F1 (macro): {metrics['f1_macro']:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['metrics'].append(metrics)
        
        # Save training history after each epoch
        history_path = config.OUTPUT_DIR / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save best model
        if metrics['roc_auc_macro'] > best_auc:
            best_auc = metrics['roc_auc_macro']
            best_epoch = epoch
            best_model_path = config.OUTPUT_DIR / 'best_model.pth'
            save_checkpoint(model, optimizer, scaler, epoch, metrics, best_model_path)
            logger.info(f"New best model saved with AUC: {best_auc:.4f}")
        
        # Early stopping
        if early_stopping(metrics['roc_auc_macro']):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = config.OUTPUT_DIR / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, scaler, epoch, metrics, checkpoint_path)
            
        # Also save a "last_checkpoint.pth" after every epoch for easy resuming
        last_checkpoint_path = config.OUTPUT_DIR / 'last_checkpoint.pth'
        save_checkpoint(model, optimizer, scaler, epoch, metrics, last_checkpoint_path)
        logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    final_model_path = config.OUTPUT_DIR / 'final_model.pth'
    save_checkpoint(model, optimizer, scaler, epoch, metrics, final_model_path)
    
    # Final save of training history
    history_path = config.OUTPUT_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best model: Epoch {best_epoch+1} with AUC {best_auc:.4f}")
    
    # Generate final report
    generate_final_report(history, config)
    
    return model, history

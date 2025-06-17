# training/trainer.py

# Standard library imports
import time
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import wandb

# Internal imports - config
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import Config

# Internal imports - models
from models.resnet3d import resnet18_3d, resnet34_3d
from models.densenet3d import densenet121_3d, densenet169_3d, densenet201_3d, densenet161_3d, densenet_small_3d, densenet_tiny_3d
from models.vit3d import vit_tiny_3d, vit_small_3d, vit_base_3d, vit_large_3d
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
    """Create and return the 3D model based on the provided configuration."""

    model_type = config.MODEL_TYPE.lower()
    model_variant = getattr(config, 'MODEL_VARIANT', 'default').lower()

    if model_type == "resnet3d":
        if model_variant == "34":
            model = resnet34_3d(
                num_classes=config.NUM_PATHOLOGIES,
                use_checkpointing=config.GRADIENT_CHECKPOINTING
            )
            logger.info(f"Created ResNet3D-34 model")
        else:  # Default to ResNet-18
            model = resnet18_3d(
                num_classes=config.NUM_PATHOLOGIES,
                use_checkpointing=config.GRADIENT_CHECKPOINTING
            )
            logger.info(f"Created ResNet3D-18 model")

    elif model_type == "densenet3d":
        densenet_models = {
            "121": densenet121_3d,
            "169": densenet169_3d,
            "201": densenet201_3d,
            "161": densenet161_3d,
            "small": densenet_small_3d,
            "tiny": densenet_tiny_3d
        }

        model_fn = densenet_models.get(model_variant, densenet121_3d)
        model = model_fn(
            num_classes=config.NUM_PATHOLOGIES,
            use_checkpointing=config.GRADIENT_CHECKPOINTING
        )
        logger.info(f"Created DenseNet3D-{model_variant or '121'} model")

    elif model_type == "vit3d":
        vit_models = {
            "tiny": vit_tiny_3d,
            "small": vit_small_3d,
            "base": vit_base_3d,
            "large": vit_large_3d
        }

        model_fn = vit_models.get(model_variant, vit_small_3d)

        # Get ViT-specific config options
        patch_size = getattr(config, 'VIT_PATCH_SIZE', (16, 16, 16))

        model = model_fn(
            num_classes=config.NUM_PATHOLOGIES,
            use_checkpointing=config.GRADIENT_CHECKPOINTING,
            volume_size=config.TARGET_SHAPE_DHW,
            patch_size=patch_size
        )
        logger.info(f"Created ViT3D-{model_variant or 'small'} model with patch_size={patch_size}")

    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")

    # Log model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    return model


def load_and_prepare_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare training and validation dataframes.

    This function reads volume and label CSVs, merges them, handles missing
    pathology columns by filling with 0, and converts pathology columns to integers.
    It also logs dataset statistics.

    Args:
        config: Configuration object containing paths to data CSVs and pathology information.

    Returns:
        A tuple containing two pandas DataFrames: (train_df, valid_df).

    Raises:
        FileNotFoundError: If any of the required CSV files are not found.
        RuntimeError: If there is an error during data loading (excluding FileNotFoundError).
        ValueError: If training or validation dataframes are empty after loading,
                    or if essential pathology columns are missing.
    """
    logger.info("Loading DataFrames...")
    try:
        # Load volume lists and label data.
        train_volumes = pd.read_csv(config.SELECTED_TRAIN_VOLUMES_CSV)[['VolumeName']]
        valid_volumes = pd.read_csv(config.SELECTED_VALID_VOLUMES_CSV)[['VolumeName']]
        train_labels = pd.read_csv(config.TRAIN_LABELS_CSV)
        valid_labels = pd.read_csv(config.VALID_LABELS_CSV)
        # Merge volumes with labels.
        train_df = pd.merge(train_volumes, train_labels, on='VolumeName', how='inner')
        valid_df = pd.merge(valid_volumes, valid_labels, on='VolumeName', how='inner')
    except FileNotFoundError as e:
        # Handle missing file errors.
        logger.error(f"Required CSV file not found: {e}")
        raise FileNotFoundError(f"Required CSV file not found: {e}")
    except Exception as e:
        # Handle other data loading errors.
        logger.error(f"Error loading data: {e}")
        raise RuntimeError(f"Error loading data: {e}")

    # Validate that dataframes are not empty.
    if train_df.empty or valid_df.empty:
        logger.error("Training or validation dataframe is empty after loading and merging.")
        raise ValueError("Training or validation dataframe is empty")

    # Check for missing pathology columns and fill NaNs.
    for df, name in [(train_df, "training"), (valid_df, "validation")]:
        missing_cols = [col for col in config.PATHOLOGY_COLUMNS if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing pathology columns in {name} data: {missing_cols}")
            raise ValueError(f"Missing pathology columns in {name} data: {missing_cols}")
        # Fill NaN values with 0 for pathology columns.
        df[config.PATHOLOGY_COLUMNS] = df[config.PATHOLOGY_COLUMNS].fillna(0)
        # Ensure pathology columns are of integer type.
        df[config.PATHOLOGY_COLUMNS] = df[config.PATHOLOGY_COLUMNS].astype(int)

    logger.info(f"Data loaded: {len(train_df)} training, {len(valid_df)} validation samples")
    # Log positive class distribution for each pathology.
    for pathology in config.PATHOLOGY_COLUMNS:
        train_pos = train_df[pathology].sum()
        valid_pos = valid_df[pathology].sum()
        logger.info(f"{pathology}: {train_pos}/{len(train_df)} train positive, "
                   f"{valid_pos}/{len(valid_df)} valid positive")
    return train_df, valid_df


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                device: torch.device, epoch: int, total_epochs: int,
                gradient_accumulation_steps: int = 1, use_amp: bool = False,
                use_bf16: bool = False) -> float:
    """Trains the model for one epoch.

    Handles forward pass, loss calculation, backward pass, optimizer step,
    and logging of training progress. Supports gradient accumulation and
    Automatic Mixed Precision (AMP).

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader for the training data.
        criterion: The loss function.
        optimizer: The optimizer.
        scaler: GradScaler for mixed-precision training (None if not used).
        device: The device to train on (e.g., 'cuda', 'cpu').
        epoch: The current epoch number (0-indexed).
        total_epochs: The total number of epochs for training.
        gradient_accumulation_steps: Number of steps to accumulate gradients before an optimizer step.
        use_amp: Boolean indicating whether to use Automatic Mixed Precision.
        use_bf16: Boolean indicating whether to use bfloat16 for mixed precision.

    Returns:
        The average training loss for the epoch.
    """
    model.train() # Set model to training mode.
    total_loss = 0.0
    num_batches = len(dataloader)
    optimizer.zero_grad() # Initialize gradients to zero.

    for batch_idx, batch in enumerate(dataloader):
        # Move data to the specified device.
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if use_amp and scaler is not None:
            # Determine the dtype for mixed precision based on config and hardware support.
            amp_dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                outputs = model(pixel_values)
                loss = criterion(outputs, labels)
                # Normalize loss for gradient accumulation.
                loss = loss / gradient_accumulation_steps
        else:
            # Standard precision forward pass.
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            # Normalize loss for gradient accumulation.
            loss = loss / gradient_accumulation_steps

        if use_amp and scaler is not None:
            # Scale loss and perform backward pass with AMP.
            scaler.scale(loss).backward()
        else:
            # Standard precision backward pass.
            loss.backward()

        # Perform optimizer step after accumulating gradients.
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            if use_amp and scaler is not None:
                # Unscale gradients and step optimizer with AMP.
                scaler.step(optimizer)
                # Update scaler for next iteration.
                scaler.update()
            else:
                # Standard optimizer step.
                optimizer.step()
            # Reset gradients for the next accumulation cycle.
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps # Accumulate un-normalized loss.
        # Log batch-level training progress.
        if batch_idx % 10 == 0: # Log every 10 batches.
            logger.info(f"Epoch [{epoch+1}/{total_epochs}] "
                       f"Batch [{batch_idx}/{num_batches}] "
                       f"Loss: {loss.item() * gradient_accumulation_steps:.4f}") # Log un-normalized loss.

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad() # Disable gradient calculations for validation.
def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    """Validates the model for one epoch.

    Args:
        model: The PyTorch model to validate.
        dataloader: DataLoader for the validation data.
        criterion: The loss function.
        device: The device to validate on (e.g., 'cuda', 'cpu').

    Returns:
        A tuple containing:
            - avg_loss (float): The average validation loss.
            - all_predictions (np.ndarray): Concatenated model outputs (logits).
            - all_labels (np.ndarray): Concatenated true labels.
    """
    model.eval() # Set model to evaluation mode.
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch in dataloader:
        # Move data to the specified device.
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Forward pass.
        outputs = model(pixel_values)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Collect predictions and labels.
        all_predictions.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    # Concatenate predictions and labels from all batches.
    all_predictions_np = np.concatenate(all_predictions, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    return avg_loss, all_predictions_np, all_labels_np


def train_model(config: Config) -> Tuple[nn.Module, Dict[str, Any]]:
    """Main function to train the CT classification model.

    This function orchestrates the entire training process, including:
    - Setting up PyTorch optimizations and device.
    - Initializing Weights & Biases (wandb) for experiment tracking.
    - Loading and preparing data.
    - Creating datasets and dataloaders.
    - Instantiating the model, loss function, and optimizer.
    - Handling checkpoint resuming.
    - Executing the training and validation loops for a configured number of epochs.
    - Implementing learning rate scheduling and early stopping.
    - Saving model checkpoints (best, last, and periodic).
    - Logging metrics to console and wandb.
    - Generating a final training report.

    Args:
        config: A Config object containing all hyperparameters and settings for training.

    Returns:
        A tuple containing:
            - model (nn.Module): The trained PyTorch model.
            - history (Dict[str, Any]): A dictionary containing training history
              (losses and metrics per epoch).
    """
    setup_torch_optimizations() # Apply PyTorch performance optimizations.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    wandb_run = None # Initialize wandb run object.
    try:
        # Configuration payload for wandb.
        wandb_config_payload = {
            "learning_rate": config.LEARNING_RATE,
            "architecture": config.MODEL_TYPE,
            "loss_function": config.LOSS_FUNCTION,
            "focal_loss_alpha": config.FOCAL_LOSS_ALPHA if config.LOSS_FUNCTION == "FocalLoss" else None,
            "focal_loss_gamma": config.FOCAL_LOSS_GAMMA if config.LOSS_FUNCTION == "FocalLoss" else None,
            "target_shape_dhw": config.TARGET_SHAPE_DHW,
            "target_spacing_xyz": config.TARGET_SPACING.tolist(),
            "clip_hu_min": config.CLIP_HU_MIN,
            "clip_hu_max": config.CLIP_HU_MAX,
            "orientation_axcodes": config.ORIENTATION_AXCODES, # Added for wandb logging
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
        # Initialize Weights & Biases run.
        wandb_run = wandb.init(
            project="ct_classifier", # Project name in wandb.
            config=wandb_config_payload, # Pass configuration to wandb.
            dir=str(config.OUTPUT_DIR) # Specify directory for wandb files.
        )
        logger.info(f"Weights & Biases initialized successfully. Run name: {wandb_run.name}")

    except Exception as e:
        # Log error if wandb initialization fails.
        logger.error(f"Failed to initialize Weights & Biases: {e}. Training will continue without wandb logging.")

    # Ensure output directory exists.
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Load and prepare data.
    train_df, valid_df = load_and_prepare_data(config)

    # Create training and validation datasets.
    train_dataset = CTDataset3D(
        dataframe=train_df,
        img_dir=config.TRAIN_IMG_DIR,
        pathology_columns=config.PATHOLOGY_COLUMNS,
        target_spacing_xyz=config.TARGET_SPACING,
        target_shape_dhw=config.TARGET_SHAPE_DHW,
        clip_hu_min=config.CLIP_HU_MIN,
        clip_hu_max=config.CLIP_HU_MAX,
        use_cache=config.USE_CACHE,
        cache_dir=config.CACHE_DIR,
        augment=True, # Enable augmentation for training dataset.
        orientation_axcodes=config.ORIENTATION_AXCODES # Pass orientation config
    )
    valid_dataset = CTDataset3D(
        dataframe=valid_df,
        img_dir=config.VALID_IMG_DIR,
        pathology_columns=config.PATHOLOGY_COLUMNS,
        target_spacing_xyz=config.TARGET_SPACING,
        target_shape_dhw=config.TARGET_SHAPE_DHW,
        clip_hu_min=config.CLIP_HU_MIN,
        clip_hu_max=config.CLIP_HU_MAX,
        use_cache=config.USE_CACHE,
        cache_dir=config.CACHE_DIR,
        augment=False, # Disable augmentation for validation dataset.
        orientation_axcodes=config.ORIENTATION_AXCODES # Pass orientation config
    )
    # Create DataLoaders.
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        persistent_workers=config.NUM_WORKERS > 0 # Keep workers alive if num_workers > 0.
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        persistent_workers=config.NUM_WORKERS > 0 # Keep workers alive if num_workers > 0.
    )

    # Create model and move to device.
    model = create_model(config).to(device)

    # Watch model with wandb if initialized.
    if wandb_run:
        try:
            wandb.watch(model, log="gradients", log_freq=100) # Log gradients every 100 batches.
            logger.info("wandb.watch() initiated for model.")
        except Exception as e:
            logger.error(f"Error during wandb.watch(): {e}")

    # Initialize loss criterion based on configuration.
    if config.LOSS_FUNCTION == "FocalLoss":
        criterion = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA)
        logger.info(f"Using FocalLoss with alpha={config.FOCAL_LOSS_ALPHA}, gamma={config.FOCAL_LOSS_GAMMA}")
    elif config.LOSS_FUNCTION == "BCEWithLogitsLoss":
        # Calculate positive class weights for BCEWithLogitsLoss.
        pos_weights = []
        for col in config.PATHOLOGY_COLUMNS:
            pos_count = train_df[col].sum()
            neg_count = len(train_df) - pos_count
            weight = neg_count / (pos_count + 1e-6) # Epsilon to prevent division by zero.
            pos_weights.append(min(weight, 10.0)) # Cap weights.
        pos_weight_tensor = torch.tensor(pos_weights, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        logger.info("Using BCEWithLogitsLoss with calculated pos_weight.")
    else:
        # Raise error for unsupported loss functions.
        raise ValueError(f"Unsupported loss function: {config.LOSS_FUNCTION}")

    # Initialize optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    # Initialize GradScaler for mixed precision if enabled.
    scaler = None
    if config.MIXED_PRECISION:
        scaler = torch.cuda.amp.GradScaler()

    # Initialize training state variables.
    start_epoch = 0
    best_auc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'valid_loss': [], 'metrics': []}

    # Resume from checkpoint if specified and checkpoint exists.
    if config.RESUME_FROM_CHECKPOINT and Path(config.RESUME_FROM_CHECKPOINT).exists():
        logger.info(f"Resuming from checkpoint: {config.RESUME_FROM_CHECKPOINT}")
        try:
            # Load checkpoint.
            checkpoint_epoch, checkpoint_metrics_loaded = load_checkpoint(
                config.RESUME_FROM_CHECKPOINT, model, optimizer, scaler
            )
            start_epoch = checkpoint_epoch + 1 # Set start epoch for training loop.
            # Load training history if available.
            history_path = config.OUTPUT_DIR / 'training_history.json'
            if history_path.exists():
                with open(history_path, 'r') as f: history = json.load(f)
                # Truncate history to the loaded checkpoint's epoch.
                if len(history['train_loss']) > checkpoint_epoch + 1:
                    history['train_loss'] = history['train_loss'][:checkpoint_epoch + 1]
                    history['valid_loss'] = history['valid_loss'][:checkpoint_epoch + 1]
                    history['metrics'] = history['metrics'][:checkpoint_epoch + 1]
                # Restore best AUC and epoch from history.
                if history['metrics']:
                    for i, metrics_item in enumerate(history['metrics']):
                        if 'roc_auc_macro' in metrics_item and metrics_item['roc_auc_macro'] > best_auc:
                            best_auc = metrics_item['roc_auc_macro']
                            best_epoch = i
            logger.info(f"Resumed from epoch {checkpoint_epoch + 1}")
            logger.info(f"Best AUC so far: {best_auc:.4f} at epoch {best_epoch + 1}")
        except Exception as e:
            # Handle errors during checkpoint loading.
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting training from scratch")
            start_epoch = 0
            best_auc = 0.0
            best_epoch = 0

    # Initialize learning rate scheduler.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS - start_epoch, eta_min=1e-6
    )
    # Advance scheduler state to the starting epoch when resuming.
    for _ in range(start_epoch):
         if scheduler: scheduler.step()

    # Initialize early stopping mechanism.
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode='max', min_delta=0.0001)
    # Set best_value for early stopping if resuming.
    if start_epoch > 0 and best_auc > 0:
        early_stopping.best_value = best_auc

    logger.info(f"Starting training from epoch {start_epoch + 1}...")
    # Dictionary to store metrics from the latest completed epoch.
    metrics_for_loop: Dict[str, Any] = {}

    # Main training loop.
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()
        # Train for one epoch.
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, config.NUM_EPOCHS, config.GRADIENT_ACCUMULATION_STEPS,
            config.MIXED_PRECISION, config.USE_BF16
        )
        # Validate for one epoch.
        valid_loss, predictions, labels = validate_epoch(
            model, valid_loader, criterion, device
        )
        # Compute validation metrics.
        metrics_for_loop = compute_metrics(predictions, labels, config.PATHOLOGY_COLUMNS)

        # Step the learning rate scheduler.
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start_time
        logger.info(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}] Time: {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        logger.info(f"Valid AUC (macro): {metrics_for_loop['roc_auc_macro']:.4f}, F1 (macro): {metrics_for_loop['f1_macro']:.4f}")

        # Record history.
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['metrics'].append(metrics_for_loop)

        # Log metrics to Weights & Biases if active.
        if wandb_run:
            log_payload = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "val_roc_auc_macro": metrics_for_loop.get('roc_auc_macro', 0.0), # Log macro AUC
            }
            try:
                wandb_run.log(log_payload)
            except Exception as e:
                logger.error(f"Failed to log metrics to wandb: {e}")

        # Save training history to JSON.
        history_path = config.OUTPUT_DIR / 'training_history.json'
        with open(history_path, 'w') as f: json.dump(history, f, indent=2)

        # Check for best model based on validation ROC AUC macro.
        current_auc = metrics_for_loop.get('roc_auc_macro', 0.0)
        if current_auc > best_auc:
            best_auc = current_auc
            best_epoch = epoch
            best_model_path = config.OUTPUT_DIR / 'best_model.pth'
            save_checkpoint(model, optimizer, scaler, epoch, metrics_for_loop, best_model_path)
            logger.info(f"New best model saved with AUC: {best_auc:.4f}")
            early_stopping.counter = 0 # Reset early stopping counter on improvement.
        else:
            early_stopping.counter +=1 # Increment counter if no improvement.

        # Check for early stopping.
        if early_stopping(current_auc):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break # Exit training loop.

        # Save periodic checkpoint.
        if (epoch + 1) % 5 == 0: # Save every 5 epochs.
            checkpoint_path = config.OUTPUT_DIR / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, scaler, epoch, metrics_for_loop, checkpoint_path)

        # Save last checkpoint.
        last_checkpoint_path = config.OUTPUT_DIR / 'last_checkpoint.pth'
        save_checkpoint(model, optimizer, scaler, epoch, metrics_for_loop, last_checkpoint_path)
        logger.info(f"Saved checkpoint at epoch {epoch+1}")

    # Determine metrics for saving the final model.
    final_metrics_to_save = metrics_for_loop
    # If resuming and training loop did not run, use metrics from loaded checkpoint if available.
    if not final_metrics_to_save and 'checkpoint_metrics_loaded' in locals() and checkpoint_metrics_loaded:
        final_metrics_to_save = checkpoint_metrics_loaded
    # Initialize to an empty dictionary if no metrics are available.
    elif not final_metrics_to_save:
        final_metrics_to_save = {}

    # Save the final model state.
    final_model_path = config.OUTPUT_DIR / 'final_model.pth'
    # Determine the epoch number for the final save.
    last_trained_epoch = epoch if 'epoch' in locals() else start_epoch -1
    save_checkpoint(model, optimizer, scaler, last_trained_epoch, final_metrics_to_save, final_model_path)

    # Save final training history.
    history_path = config.OUTPUT_DIR / 'training_history.json'
    with open(history_path, 'w') as f: json.dump(history, f, indent=2)

    logger.info(f"\nTraining completed!")
    logger.info(f"Best model: Epoch {best_epoch+1} with AUC {best_auc:.4f}")
    # Generate final report if training history is not empty.
    if history['train_loss']:
        generate_final_report(history, config)
    else:
        logger.warning("Training history is empty. Skipping final report generation.")

    # Finish Weights & Biases run if initialized.
    if wandb_run:
        try:
            wandb_run.finish()
            logger.info("Weights & Biases run finished.")
        except Exception as e:
            logger.error(f"Error finishing wandb run: {e}")

    return model, history
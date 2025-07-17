# training/trainer.py

# Standard library imports
import time
import json
import hashlib
import logging
from tqdm.auto import tqdm
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Callable
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Third-party imports
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import wandb
import functools

# MONAI imports
from monai.data import PersistentDataset, DataLoader, Dataset, CacheDataset 
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    Resized,
    RandFlipd,
    RandGaussianNoised,
    RandShiftIntensityd,
    EnsureTyped,
    RandAffined
)
from monai.losses import FocalLoss


# Internal imports - models
from src.models.resnet3d import resnet18_3d, resnet34_3d
from src.models.densenet3d import densenet121_3d, densenet169_3d, densenet201_3d, densenet161_3d
from src.models.vit3d import vit_tiny_3d, vit_small_3d, vit_base_3d, vit_large_3d

# Internal imports - data
from src.data.dataset import CTMetadataDataset, ApplyTransforms

# Internal imports - training utilities
from src.training.metrics import compute_metrics
from src.training.utils import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint
)

# Internal imports - evaluation
from src.evaluation.reporting import generate_final_report

# Internal imports - utils
from src.utils.torch_utils import setup_torch_optimizations

# Get logger
logger = logging.getLogger(__name__)

from src.data.cache_utils import (
    get_or_create_cache_subdirectory,
    deterministic_json_hash,
    worker_init_fn
)


def create_model(config: SimpleNamespace) -> nn.Module:
    """Create and return the 3D model based on the provided configuration."""

    model_type = config.model.type.lower()
    model_variant = str(config.model.variant).lower()
    num_classes = len(config.pathologies.columns)
    use_checkpointing = config.optimization.gradient_checkpointing

    if model_type == "resnet3d":
        if model_variant == "34":
            model = resnet34_3d(num_classes=num_classes, use_checkpointing=use_checkpointing)
            logger.info("Created ResNet3D-34 model")
        else:  # Default to ResNet-18
            model = resnet18_3d(num_classes=num_classes, use_checkpointing=use_checkpointing)
            logger.info("Created ResNet3D-18 model")

    elif model_type == "densenet3d":
        densenet_models = {
            "121": densenet121_3d, "169": densenet169_3d, "201": densenet201_3d, "161": densenet161_3d
        }
        model_fn = densenet_models.get(model_variant, densenet121_3d) # Default to 121
        model = model_fn(num_classes=num_classes, use_checkpointing=use_checkpointing)
        logger.info(f"Created DenseNet3D-{model_variant or '121'} model")

    elif model_type == "vit3d":
        vit_models = {
            "tiny": vit_tiny_3d, "small": vit_small_3d, "base": vit_base_3d, "large": vit_large_3d
        }
        model_fn = vit_models.get(model_variant, vit_small_3d) # Default to small
        model = model_fn(
            num_classes=num_classes,
            use_checkpointing=use_checkpointing,
            volume_size=config.image_processing.target_shape_dhw,
            patch_size=config.model.vit_specific.patch_size
        )
        logger.info(f"Created ViT3D-{model_variant or 'small'} model")

    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    return model


def load_and_prepare_data(config: SimpleNamespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare training and validation dataframes from unified sources.

    This function reads volume lists for a specific data split (e.g., a fold)
    and merges them with a unified, master labels file. It handles missing
    pathology columns by filling with 0 and converts them to integers.

    Args:
        config: Configuration object containing paths and pathology information.

    Returns:
        A tuple containing two pandas DataFrames: (train_df, valid_df).

    Raises:
        FileNotFoundError: If any of the required CSV files are not found.
        RuntimeError: If there is an error during data loading.
        ValueError: If dataframes are empty or essential columns are missing.
    """
    logger.info("Loading DataFrames for the current data split...")
    try:
        data_dir = Path(config.paths.data_dir).resolve()
        # Load the volume lists for the current training and validation split
        train_volumes = pd.read_csv(Path(data_dir) / config.paths.data_subsets.train)[['VolumeName']]
        valid_volumes = pd.read_csv(Path(data_dir) / config.paths.data_subsets.valid)[['VolumeName']]
        
        # Load the single, unified labels file for all volumes
        all_labels = pd.read_csv(Path(data_dir) / config.paths.labels.all)
        logger.info(f"Loaded {len(all_labels)} total labels from the unified labels file at {config.paths.labels.all}")
        
        # Merge the split volumes with the unified labels file
        train_df = pd.merge(train_volumes, all_labels, on='VolumeName', how='inner')
        valid_df = pd.merge(valid_volumes, all_labels, on='VolumeName', how='inner')
        
    except FileNotFoundError as e:
        logger.error(f"Required CSV file not found: {e}")
        raise FileNotFoundError(f"Required CSV file not found: {e}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise RuntimeError(f"Error loading data: {e}")

    # Validate that dataframes are not empty
    if train_df.empty or valid_df.empty:
        logger.error("Training or validation dataframe is empty after loading and merging.")
        raise ValueError("Training or validation dataframe is empty")

    # Check for missing pathology columns and fill NaNs
    for df, name in [(train_df, "training"), (valid_df, "validation")]:
        missing_cols = [col for col in config.pathologies.columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing pathology columns in {name} data: {missing_cols}")
            raise ValueError(f"Missing pathology columns in {name} data: {missing_cols}")
        
        df[config.pathologies.columns] = df[config.pathologies.columns].fillna(0)
        df[config.pathologies.columns] = df[config.pathologies.columns].astype(int)

    logger.info(f"Data loaded: {len(train_df)} training, {len(valid_df)} validation samples")
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

    # Wrap the dataloader with tqdm for a progress bar
    progress_bar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch + 1}/{total_epochs} [Training]", 
        leave=False, 
        unit="batch"
    )

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to the specified device.
        pixel_values = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

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

        # We use the un-normalized loss for logging and tracking
        unnormalized_loss = loss.item() * gradient_accumulation_steps
        total_loss += unnormalized_loss
        
        # Update the progress bar description with the current batch loss
        progress_bar.set_postfix(loss=unnormalized_loss)

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

    progress_bar = tqdm(
        dataloader, 
        desc=f"[Validation]", 
        leave=False, 
        unit="batch"
    )

    for batch in progress_bar:
        # Move data to the specified device.
        pixel_values = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

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


def train_model(
    config: SimpleNamespace,
    device: Optional[torch.device] = None,
    optuna_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
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
        config: A Config object containing all hyperparameters and settings.
        device (Optional[torch.device]): The device to train on. If None, it
            will be automatically detected (CUDA or CPU).
        optuna_callback (Optional[Callable]): An optional callback for Optuna
            pruning, which takes the epoch and validation metrics as input.
    Returns:
        A tuple containing:
            - model (nn.Module): The trained PyTorch model.
            - history (Dict[str, Any]): A dictionary containing training history
              (losses and metrics per epoch).
    """
    setup_torch_optimizations()  # Apply PyTorch performance optimizations.
    
    # Use the provided device or detect automatically.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    wandb_run = None  # Initialize wandb run object.
    # Check for the presence of wandb config and if it's enabled.
    if hasattr(config, "wandb") and config.wandb.enabled:
        try:
            # Configuration payload for wandb.
            wandb_config_payload = {
                "learning_rate": config.training.learning_rate,
                "architecture": config.model.type,
                "model_variant": config.model.variant,
                "loss_function": config.loss_function.type,
                "focal_loss_alpha": config.loss_function.focal_loss.alpha,
                "focal_loss_gamma": config.loss_function.focal_loss.gamma,
                "target_shape_dhw": config.image_processing.target_shape_dhw,
                "target_spacing": config.image_processing.target_spacing,
                "clip_hu_min": config.image_processing.clip_hu_min,
                "clip_hu_max": config.image_processing.clip_hu_max,
                "orientation_axcodes": config.image_processing.orientation_axcodes,
                "epochs": config.training.num_epochs,
                "batch_size": config.training.batch_size,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                "weight_decay": config.training.weight_decay,
                "num_workers": config.training.num_workers,
                "pin_memory": config.training.pin_memory,
                "gradient_checkpointing": config.optimization.gradient_checkpointing,
                "mixed_precision": config.optimization.mixed_precision,
                "use_bf16": config.optimization.use_bf16,
                "early_stopping_patience": config.training.early_stopping_patience,
                "use_cache": config.cache.use_cache,
                "num_pathologies": len(config.pathologies.columns),
                "pathology_columns": config.pathologies.columns,
                "output_dir": str(config.paths.output_dir),
                "resume_from_checkpoint": str(config.training.resume_from_checkpoint) if config.training.resume_from_checkpoint else None,
            }
            # Initialize Weights & Biases run.
            wandb_run = wandb.init(
                project=config.wandb.project,
                config=wandb_config_payload,
                dir=str(config.paths.output_dir),
                name=getattr(config.wandb, 'run_name', None),
                group=getattr(config.wandb, 'group', None),
                resume=getattr(config.wandb, 'resume', None),
                reinit=True,  # Allow re-initialization for multiple trials in one script
                settings=wandb.Settings(_disable_stats=True) # Reduce overhead
            )
            logger.info(f"Weights & Biases initialized successfully. Run name: {wandb_run.name}, Group: {wandb_run.group}")
        except Exception as e:
            # Log error if wandb initialization fails.
            logger.error(f"Failed to initialize Weights & Biases: {e}. Training will continue without wandb logging.")

    # Ensure output directory exists.
    output_dir = Path(config.paths.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data.
    train_df, valid_df = load_and_prepare_data(config)

    # Define MONAI Transform Pipelines
    # These keys must match the dictionary keys returned by CTMetadataDataset
    keys = ["image", "label"]

    preprocess_transforms = Compose([
        LoadImaged(keys="image", image_only=True, ensure_channel_first=True,
                   reader="NibabelReader"),
        Orientationd(keys="image", axcodes=config.image_processing.orientation_axcodes),
        Spacingd(keys="image", pixdim=config.image_processing.target_spacing, mode="bilinear"),
        ScaleIntensityRanged(keys="image",
                               a_min=config.image_processing.clip_hu_min,
                               a_max=config.image_processing.clip_hu_max,
                               b_min=0.0, b_max=1.0, clip=True),
        Resized(keys="image", spatial_size=config.image_processing.target_shape_dhw, mode="area"),
        EnsureTyped(keys=keys, dtype=torch.float32)
    ])

    # Create the base metadata datasets
    base_train_ds = CTMetadataDataset(
        dataframe=train_df,
        img_dir=config.paths.img_dir,
        pathology_columns=config.pathologies.columns,
        path_mode=config.paths.dir_structure
    )
    # Both datasets point to the same unified image directory.
    base_valid_ds = CTMetadataDataset(
        dataframe=valid_df,
        img_dir=config.paths.img_dir,
        pathology_columns=config.pathologies.columns,
        path_mode=config.paths.dir_structure
    )

    augment_transforms = Compose([
        # Spatial augmentations (Potentially GPU-based)
        RandAffined(
            keys="image",
            prob=0.1, 
            rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12), # Rotate by +/- 15 degrees
            scale_range=((0.85, 1.15), (0.85, 1.15), (0.85, 1.15)), # Zoom in/out by 15%
            translate_range=((-20, 20), (-20, 20), (0, 0)), # Translate on H and W axes
            mode="bilinear",
            padding_mode="zeros",
            device="cpu" 
        ),
        # CPU-based intensity augmentations
        RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.01),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        EnsureTyped(keys=keys, dtype=torch.float32)
    ])

    # Compose the final datasets with caching and augmentations
    if config.cache.use_cache:
        logger.info(f"Hybrid caching enabled. Disk cache is persistent.")

        # --- Training Dataset Chain ---
        
        # 1. Determine the configuration-specific cache directory for training data.
        train_cache_dir = get_or_create_cache_subdirectory(
            base_cache_dir=config.paths.cache_dir,
            transforms=preprocess_transforms,
            split="train"
        )
        
        # 2. On-disk cache for the entire dataset using the specific directory.
        train_persistent_ds = PersistentDataset(
            data=base_train_ds,
            transform=preprocess_transforms,
            cache_dir=train_cache_dir,  # Use the new dynamic cache path
            hash_func=deterministic_json_hash,
            hash_transform=deterministic_json_hash
        )

        # 3. In-memory cache for a percentage of the on-disk data.
        logger.info(f"Caching {config.cache.memory_rate * 100:.0f}% of training data in RAM.")
        train_in_memory_ds = CacheDataset(
            data=train_persistent_ds,
            cache_rate=config.cache.memory_rate,
            num_workers=config.training.num_workers
        )

        # 4. Apply on-the-fly augmentations to the cached data.
        if config.training.augment:
            train_dataset = ApplyTransforms(train_in_memory_ds, augment_transforms)
            logger.info("Applying on-the-fly augmentations to hybrid-cached training data.")
        else:
            train_dataset = train_in_memory_ds
            logger.info("Hybrid-cached training data ready.")

        # --- Validation Dataset Chain ---
        
        # 1. Determine the configuration-specific cache directory for validation data.
        valid_cache_dir = get_or_create_cache_subdirectory(
            base_cache_dir=config.paths.cache_dir,
            transforms=preprocess_transforms,
            split="valid"
        )

        # 2. On-disk cache for the validation set.
        valid_persistent_ds = PersistentDataset(
            data=base_valid_ds,
            transform=preprocess_transforms,
            cache_dir=valid_cache_dir,  # Use the new dynamic cache path
            hash_func=deterministic_json_hash,
            hash_transform=deterministic_json_hash
        )
        
        # 3. In-memory cache for the validation set.
        logger.info(f"Caching {config.cache.memory_rate * 100:.0f}% of validation data in RAM.")
        valid_dataset = CacheDataset(
            data=valid_persistent_ds,
            cache_rate=config.cache.memory_rate,
            num_workers=config.training.num_workers
        )
        logger.info("Hybrid-cached validation data ready.")

    else:
        # No caching, apply all transforms on-the-fly
        logger.info("Caching is disabled. Applying all transforms on-the-fly.")
        final_train_transforms = Compose([preprocess_transforms, augment_transforms]) if config.training.augment else preprocess_transforms
        train_dataset = Dataset(data=base_train_ds, transform=final_train_transforms)
        valid_dataset = Dataset(data=base_valid_ds, transform=preprocess_transforms)


    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True,
        num_workers=config.training.num_workers, pin_memory=config.training.pin_memory,
        persistent_workers=config.training.num_workers > 0,
        prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None,
        worker_init_fn=worker_init_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers, pin_memory=config.training.pin_memory,
        persistent_workers=config.training.num_workers > 0,
        prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None,
        worker_init_fn=worker_init_fn
    )

    # Create model and move to device.
    model = create_model(config).to(device)

    # Watch model with wandb if initialized.
    if wandb_run:
        try:
            wandb.watch(model, log="gradients", log_freq=100)  # Log gradients every 100 batches.
            logger.info("wandb.watch() initiated for model.")
        except Exception as e:
            logger.error(f"Error during wandb.watch(): {e}")

    # Initialize loss criterion based on configuration.
    if config.loss_function.type == "FocalLoss":
        criterion = FocalLoss(alpha=config.loss_function.focal_loss.alpha, gamma=config.loss_function.focal_loss.gamma)
        logger.info(f"Using FocalLoss with alpha={config.loss_function.focal_loss.alpha}, gamma={config.loss_function.focal_loss.gamma}")
    elif config.loss_function.type == "BCEWithLogitsLoss":
        # Calculate positive class weights for BCEWithLogitsLoss.
        pos_weights = []
        for col in config.pathologies.columns:
            pos_count = train_df[col].sum()
            neg_count = len(train_df) - pos_count
            weight = neg_count / (pos_count + 1e-6)  # Epsilon to prevent division by zero.
            pos_weights.append(min(weight, 10.0))  # Cap weights.
        pos_weight_tensor = torch.tensor(pos_weights, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        logger.info("Using BCEWithLogitsLoss with calculated pos_weight.")
    else:
        # Raise error for unsupported loss functions.
        raise ValueError(f"Unsupported loss function: {config.loss_function.type}")

    # Initialize optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay
    )
    # Initialize GradScaler for mixed precision if enabled.
    scaler = None
    if config.optimization.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()

    # Initialize training state variables.
    start_epoch = 0
    best_auc = 0.0
    history = {'train_loss': [], 'valid_loss': [], 'metrics': []}
    checkpoint_metrics_loaded = {}

    # Resume from checkpoint if specified and checkpoint exists.
    if config.training.resume_from_checkpoint and Path(config.training.resume_from_checkpoint).exists():
        logger.info(f"Resuming from checkpoint: {config.training.resume_from_checkpoint}")
        try:
            # Load checkpoint.
            checkpoint_epoch, checkpoint_metrics_loaded = load_checkpoint(
                config.training.resume_from_checkpoint, model, optimizer, scaler
            )
            start_epoch = checkpoint_epoch + 1  # Set start epoch for training loop.
            best_auc = checkpoint_metrics_loaded.get('roc_auc_macro', 0.0)
            
            # Load training history if available.
            history_path = Path(config.training.resume_from_checkpoint).parent / 'training_history.json'
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                # Truncate history to the loaded checkpoint's epoch.
                history['train_loss'] = history['train_loss'][:start_epoch]
                history['valid_loss'] = history['valid_loss'][:start_epoch]
                history['metrics'] = history['metrics'][:start_epoch]

            logger.info(f"Resumed from epoch {start_epoch}")
            logger.info(f"Best AUC from loaded checkpoint: {best_auc:.4f}")
        except Exception as e:
            # Handle errors during checkpoint loading.
            logger.error(f"Error loading checkpoint: {e}", exc_info=True)
            logger.info("Starting training from scratch")
            start_epoch = 0
            best_auc = 0.0
            history = {'train_loss': [], 'valid_loss': [], 'metrics': []}


    # Initialize learning rate scheduler.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.num_epochs - start_epoch, eta_min=1e-6
    )
    # Advance scheduler state to the starting epoch when resuming.
    if start_epoch > 0:
        for _ in range(start_epoch):
            if scheduler: scheduler.step()

    # Initialize early stopping mechanism.
    early_stopping = EarlyStopping(patience=config.training.early_stopping_patience, mode='max', min_delta=0.0001)
    # Set best_value for early stopping if resuming.
    if start_epoch > 0 and best_auc > 0:
        early_stopping.best_value = best_auc

    logger.info(f"Starting training from epoch {start_epoch + 1}...")

    # Main training loop.
    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start_time = time.time()
        # Train for one epoch.
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, config.training.num_epochs, config.training.gradient_accumulation_steps,
        )
        # Validate for one epoch.
        valid_loss, predictions, labels = validate_epoch(
            model, valid_loader, criterion, device
        )
        # Compute validation metrics.
        metrics_for_loop = compute_metrics(predictions, labels, config.pathologies.columns)
        # Add validation loss to metrics dictionary
        metrics_for_loop['loss'] = valid_loss

        # Step the learning rate scheduler.
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start_time
        logger.info(f"\nEpoch [{epoch+1}/{config.training.num_epochs}] Time: {epoch_time:.2f}s")
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
                "val_roc_auc_macro": metrics_for_loop.get('roc_auc_macro', 0.0),
                "val_f1_macro": metrics_for_loop.get('f1_macro', 0.0),
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            try:
                wandb_run.log(log_payload)
            except Exception as e:
                logger.error(f"Failed to log metrics to wandb: {e}")

        # Save training history to JSON.
        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f: json.dump(history, f, indent=2)

        # Check for best model based on validation ROC AUC macro.
        current_auc = metrics_for_loop.get('roc_auc_macro', 0.0)
        if current_auc > best_auc:
            best_auc = current_auc
            best_model_path = output_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, scaler, epoch, metrics_for_loop, best_model_path)
            logger.info(f"New best model saved with AUC: {best_auc:.4f}")
        
        # Check for early stopping.
        if early_stopping(current_auc):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break  # Exit training loop.
            
        # Save last checkpoint.
        last_checkpoint_path = output_dir / 'last_checkpoint.pth'
        save_checkpoint(model, optimizer, scaler, epoch, metrics_for_loop, last_checkpoint_path)


        # If an Optuna callback is provided, execute it with the latest
        # validation metrics. The callback itself should handle raising
        # the TrialPruned exception if necessary.
        if optuna_callback:
            optuna_callback(epoch, metrics_for_loop)

    # Determine metrics for saving the final model.
    final_metrics_to_save = history['metrics'][-1] if history['metrics'] else {}

    # Save the final model state.
    final_model_path = output_dir / 'final_model.pth'
    last_trained_epoch = epoch if 'epoch' in locals() else start_epoch -1
    save_checkpoint(model, optimizer, scaler, last_trained_epoch, final_metrics_to_save, final_model_path)

    # Save final training history.
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f: json.dump(history, f, indent=2)

    logger.info(f"\nTraining completed!")
    if history['metrics']:
        best_epoch_idx = np.argmax([m.get('roc_auc_macro', 0.0) for m in history['metrics']])
        best_auc_final = history['metrics'][best_epoch_idx].get('roc_auc_macro', 0.0)
        logger.info(f"Best model: Epoch {history.get('metrics')[best_epoch_idx].get('epoch', best_epoch_idx)+1} with AUC {best_auc_final:.4f}")
        generate_final_report(history, config, best_epoch_idx=best_epoch_idx)
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
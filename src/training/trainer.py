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
    RandFlipd,
    RandGaussianNoised,
    RandShiftIntensityd,
    EnsureTyped,
    RandAffined,
    RandGaussianSmoothd,
    RandAdjustContrastd,
)
from monai.losses import FocalLoss
from monai.metrics import ROCAUCMetric, FBetaScore
from monai.metrics.f_beta_score import compute_f_beta_score


# Internal imports - models
from src.models.resnet3d import resnet18_3d, resnet34_3d
from src.models.densenet3d import densenet121_3d, densenet169_3d, densenet201_3d, densenet161_3d
from src.models.vit3d import vit_tiny_3d, vit_small_3d, vit_base_3d, vit_large_3d
from src.models.mlp import create_mlp_classifier
from src.models.logistic import create_logistic_classifier

# Internal imports - data
from src.data.dataset import CTMetadataDataset, ApplyTransforms, LabelAttacherDataset, FeatureDataset
from src.data.transforms import get_preprocessing_transforms

# Internal imports - training utilities
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
    deterministic_hash,
    worker_init_fn
)


def generate_wandb_run_name(config: SimpleNamespace) -> str:
    """Create a compact W&B run name using model variant, workflow mode, and dataset."""

    def _sanitize_component(component: str, fallback: str) -> str:
        if not component:
            return fallback
        sanitized = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '-' for ch in component)
        sanitized = sanitized.strip('-_')
        return sanitized or fallback

    model_ns = getattr(config, 'model', SimpleNamespace())
    model_type = _sanitize_component(str(getattr(model_ns, 'type', 'model')).upper(), 'MODEL')
    model_variant = getattr(model_ns, 'variant', None)
    if model_variant:
        variant_part = _sanitize_component(str(model_variant).upper(), 'VAR')
        model_part = f"{model_type}-{variant_part}"
    else:
        model_part = model_type

    workflow_ns = getattr(config, 'workflow', SimpleNamespace())
    workflow_mode = _sanitize_component(str(getattr(workflow_ns, 'mode', 'workflow')).lower(), 'workflow')

    dataset_part = 'dataset'
    try:
        data_subsets = getattr(getattr(config, 'paths', SimpleNamespace()), 'data_subsets', SimpleNamespace())
        train_subset = getattr(data_subsets, 'train', None)
        if train_subset:
            if not isinstance(train_subset, Path):
                train_subset = Path(train_subset)
            dataset_part = _sanitize_component(train_subset.stem.lower() or train_subset.name.lower(), 'dataset')
    except Exception:
        dataset_part = 'dataset'

    signature_payload = {
        'lr': getattr(getattr(config, 'training', SimpleNamespace()), 'learning_rate', None),
        'batch': getattr(getattr(config, 'training', SimpleNamespace()), 'batch_size', None),
        'augment': getattr(getattr(config, 'training', SimpleNamespace()), 'augment', None),
        'cache': getattr(getattr(config, 'cache', SimpleNamespace()), 'use_cache', None),
        'mp': getattr(getattr(config, 'optimization', SimpleNamespace()), 'mixed_precision', None),
    }
    try:
        signature_raw = json.dumps(signature_payload, sort_keys=True, default=str).encode('utf-8')
        signature_part = hashlib.sha1(signature_raw).hexdigest()[:4]
    except Exception:
        signature_part = 'custom'

    return f"{model_part}_{workflow_mode}_{dataset_part}_{signature_part}"


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

    def _normalize_volume_name(name: Any) -> Optional[str]:
        if pd.isna(name):
            return None

        name_str = str(name).strip()
        if not name_str:
            return None

        lowered = name_str.lower()
        for ext in ('.nii.gz', '.nii'):
            if lowered.endswith(ext):
                return name_str[: -len(ext)]
        return name_str

    try:
        data_dir = Path(config.paths.data_dir).resolve()
        # Load the volume lists for the current training and validation split
        train_volumes = pd.read_csv(Path(data_dir) / config.paths.data_subsets.train)[['VolumeName']].copy()
        valid_volumes = pd.read_csv(Path(data_dir) / config.paths.data_subsets.valid)[['VolumeName']].copy()

        train_volumes['__volume_key'] = train_volumes['VolumeName'].apply(_normalize_volume_name)
        valid_volumes['__volume_key'] = valid_volumes['VolumeName'].apply(_normalize_volume_name)
        train_volumes['__order'] = np.arange(len(train_volumes))
        valid_volumes['__order'] = np.arange(len(valid_volumes))

        if train_volumes['__volume_key'].isna().any():
            missing_raw = train_volumes.loc[train_volumes['__volume_key'].isna(), 'VolumeName'].tolist()
            raise ValueError(f"Unable to normalize training volume names: {missing_raw}")
        if valid_volumes['__volume_key'].isna().any():
            missing_raw = valid_volumes.loc[valid_volumes['__volume_key'].isna(), 'VolumeName'].tolist()
            raise ValueError(f"Unable to normalize validation volume names: {missing_raw}")

        # Load the single, unified labels file for all volumes
        all_labels = pd.read_csv(Path(data_dir) / config.paths.labels.all).copy()
        logger.info(f"Loaded {len(all_labels)} total labels from the unified labels file at {config.paths.labels.all}")

        all_labels['__volume_key'] = all_labels['VolumeName'].apply(_normalize_volume_name)

        label_keys = set(all_labels['__volume_key'].dropna())

        missing_train = train_volumes.loc[~train_volumes['__volume_key'].isin(label_keys), 'VolumeName']
        if not missing_train.empty:
            raise ValueError(
                "Training volume names not found in labels (after normalizing extensions): "
                f"{sorted(missing_train.tolist())}"
            )

        missing_valid = valid_volumes.loc[~valid_volumes['__volume_key'].isin(label_keys), 'VolumeName']
        if not missing_valid.empty:
            raise ValueError(
                "Validation volume names not found in labels (after normalizing extensions): "
                f"{sorted(missing_valid.tolist())}"
            )

        # Merge the split volumes with the unified labels file using normalized keys
        train_df = pd.merge(
            train_volumes[['__volume_key', '__order']],
            all_labels,
            on='__volume_key',
            how='inner'
        ).sort_values('__order').drop(columns=['__volume_key', '__order']).reset_index(drop=True)

        valid_df = pd.merge(
            valid_volumes[['__volume_key', '__order']],
            all_labels,
            on='__volume_key',
            how='inner'
        ).sort_values('__order').drop(columns=['__volume_key', '__order']).reset_index(drop=True)
        
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
    """
    Trains the model for one epoch.

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
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        use_amp: Boolean indicating whether to use Automatic Mixed Precision.
        use_bf16: Boolean indicating whether to use bfloat16 for mixed precision.

    Returns:
        The average training loss for the epoch.
    """
    model.train()  # Set model to training mode.
    total_loss = 0.0
    num_batches = len(dataloader)
    optimizer.zero_grad()  # Initialize gradients to zero.

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
            # Determine the dtype for mixed precision.
            amp_dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
            # Use the new torch.amp.autocast API (device_type argument) to avoid deprecation warnings.
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
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

        # We use the un-normalized loss for logging and tracking.
        unnormalized_loss = loss.item() * gradient_accumulation_steps
        total_loss += unnormalized_loss
        
        # Update the progress bar description with the current batch loss.
        progress_bar.set_postfix(loss=f"{unnormalized_loss:.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                   device: torch.device, pathology_names: list) -> Tuple[float, dict]:
    """Validates the model for one epoch using MONAI metrics."""
    model.eval()
    total_loss = 0.0

    # Instantiate metric objects before the loop
    auc_metric = ROCAUCMetric()
    # F1-score is calculated using FBetaScore with beta=1.0
    f1_metric = FBetaScore(beta=1.0)

    progress_bar = tqdm(
        dataloader, 
        desc="[Validation]", 
        leave=False, 
        unit="batch"
    )

    for batch in progress_bar:
        pixel_values = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(pixel_values)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Convert logits to probabilities for AUC and binary predictions for F1
        probabilities = torch.sigmoid(outputs)
        binary_predictions = (probabilities > 0.5).long()

        # Ensure labels are in the correct format (long integers)
        y_true = labels.long()

        # Update metric objects with correctly formatted tensors
        auc_metric(probabilities, y_true)
        f1_metric(binary_predictions, y_true)

    avg_loss = total_loss / len(dataloader)

    # --- Metric Aggregation ---
    # Aggregate the final results after the loop using the correct methods.

    # 1. AUC Score Aggregation (this part remains the same)
    auc_macro = auc_metric.aggregate(average="macro")
    auc_micro = auc_metric.aggregate(average="micro")
    per_pathology_auc = auc_metric.aggregate(average="none")

    # 2. F1 Score Aggregation (Manual calculation for precision)
    # Get the raw confusion matrix buffer of shape (N, C, 4) [TP, FP, TN, FN]
    cm_buffer = f1_metric.get_buffer()

    # Calculate Macro F1: Compute F1 per class, then average the scores.
    # We first average the confusion matrix components across the batch dimension.
    f1_macro_per_class = compute_f_beta_score(cm_buffer.nanmean(dim=0), beta=1.0)
    f1_macro = f1_macro_per_class.nanmean().item()

    # Calculate Micro F1: Sum confusion matrix components across batch and classes, then compute F1.
    total_cm = cm_buffer.sum(dim=(0, 1))
    f1_micro = compute_f_beta_score(total_cm, beta=1.0).item()

    # 3. Assemble the final metrics dictionary
    metrics_dict = {
        'roc_auc_macro': auc_macro,
        'roc_auc_micro': auc_micro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
    }

    if per_pathology_auc is not None:
        for i, name in enumerate(pathology_names):
            metrics_dict[f"{name}_auc"] = per_pathology_auc[i].item() if hasattr(per_pathology_auc[i], 'item') else per_pathology_auc[i]

    # Reset metrics for the next epoch's validation
    auc_metric.reset()
    f1_metric.reset()

    return avg_loss, metrics_dict



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
            configured_run_name = getattr(config.wandb, 'run_name', None)
            run_name = configured_run_name or generate_wandb_run_name(config)
            if not configured_run_name:
                logger.info(f"Generated W&B run name: {run_name}")
            wandb_run = wandb.init(
                project=config.wandb.project,
                config=wandb_config_payload,
                dir=str(config.paths.output_dir),
                name=run_name,
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
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data.
    train_df, valid_df = load_and_prepare_data(config)

    # Determine the training workflow
    workflow_ns = getattr(config, 'workflow', SimpleNamespace())
    workflow_mode = str(getattr(workflow_ns, 'mode', 'end-to-end')).lower()
    if not hasattr(workflow_ns, 'feature_config'):
        workflow_ns.feature_config = SimpleNamespace()
        config.workflow = workflow_ns
    model_type = str(getattr(config.model, 'type', '')).lower()

    feature_cfg_ns = getattr(config.workflow, 'feature_config', SimpleNamespace())
    feature_dir_configured = getattr(feature_cfg_ns, 'feature_dir', None)

    logger.info(f"Workflow mode raw: '{workflow_mode}', feature_dir: {feature_dir_configured}")

    feature_model_types = {'mlp', 'logistic', 'logistic_regression'}

    if model_type in feature_model_types and workflow_mode != 'feature-based':
        logger.warning(
            "%s model requires feature-based workflow. Switching workflow.mode to 'feature-based' automatically.",
            model_type.upper()
        )
        workflow_mode = 'feature-based'
        config.workflow.mode = 'feature-based'

    logger.info(f"Starting training in '{workflow_mode}' mode with model '{model_type}'.")

    if workflow_mode == 'feature-based' and not feature_dir_configured:
        raise ValueError(
            "Feature-based workflow requires 'workflow.feature_config.feature_dir' to be configured."
        )

    if workflow_mode == 'feature-based' and model_type not in feature_model_types:
        valid_names = ', '.join(sorted(feature_model_types))
        raise ValueError(
            f"Feature-based workflow requires one of the feature classifiers ({valid_names}). Please update config.model.type accordingly."
        )
    if workflow_mode != 'feature-based' and model_type in feature_model_types:
        raise ValueError(
            f"{config.model.type} model is only supported in feature-based workflow. Please switch workflow.mode to 'feature-based'."
        )

    if workflow_mode == 'feature-based':
        # --- Feature-Based Workflow ---
        logger.info("Setting up feature-based data pipeline.")
        feature_dir = Path(config.workflow.feature_config.feature_dir)

        # Determine whether to preload features into RAM (config option)
        preload_flag = False
        if hasattr(config, 'training') and hasattr(config.training, 'features_preload_to_ram'):
            preload_flag = bool(config.training.features_preload_to_ram)
        if preload_flag:
            logger.info("Feature preloading enabled: feature tensors will be loaded into RAM at dataset init.")

        # Support two layouts for feature storage:
        # A) feature_dir/train/*.pt and feature_dir/valid/*.pt (preferred)
        # B) feature_dir/*.pt (flattened) -> use same directory for both train and valid
        train_sub = feature_dir / 'train'
        valid_sub = feature_dir / 'valid'
        if train_sub.exists() and valid_sub.exists():
            train_feature_dir = train_sub
            valid_feature_dir = valid_sub
            logger.info("Using feature subfolders: 'train' and 'valid'.")
        elif any(feature_dir.glob('*.pt')):
            train_feature_dir = feature_dir
            valid_feature_dir = feature_dir
            logger.info("Using flattened feature directory for both train and valid.")
        else:
            raise FileNotFoundError(
                f"Feature directory layout not recognised. Expected '{feature_dir}/train' and '{feature_dir}/valid' or .pt files at '{feature_dir}'."
            )

        train_dataset = FeatureDataset(
            dataframe=train_df,
            feature_dir=train_feature_dir,
            pathology_columns=config.pathologies.columns,
            preload_to_ram=preload_flag
        )
        valid_dataset = FeatureDataset(
            dataframe=valid_df,
            feature_dir=valid_feature_dir,
            pathology_columns=config.pathologies.columns,
            preload_to_ram=preload_flag
        )

        # In this mode, augmentations are not applicable as we are using static features.
        augment_transforms = None
    else:
        # --- End-to-End (Image-Based) Workflow ---

        # Define the preprocessing and augmentation transform pipelines.
        preprocess_transforms = get_preprocessing_transforms(config)
        
        # The augmentations will be passed to the training loop directly.
        augment_transforms = None
        if config.training.augment:
            logger.info("On-the-fly GPU augmentations are enabled.")
            augment_transforms = Compose([
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2), # Flip along the L-R axis
                RandAffined(
                    keys="image",
                    prob=0.15,
                    rotate_range=(np.pi / 36, np.pi / 36, np.pi / 12), # Reduced rotation
                    scale_range=((0.9, 1.1), (0.9, 1.1), (0.9, 1.1)),
                    translate_range=((-10, 10), (-10, 10), (0, 0)),
                    mode="bilinear",
                    padding_mode="zeros",
                ),
                RandGaussianSmoothd(keys=["image"], prob=0.15, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
                RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.5)),
                RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.01),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                # Final type check after all transforms.
                EnsureTyped(keys=["image", "label"], dtype=config.torch_dtype)
            ])

            

        if config.cache.use_cache:
            logger.info("Persistent caching is enabled.")
            
            # --- Training Data Pipeline ---
            # 1. Label-agnostic dataset to find image paths.
            base_train_ds = CTMetadataDataset(dataframe=train_df, img_dir=config.paths.img_dir, path_mode=config.paths.dir_structure)
            
            # 2. Define cache directory based on the preprocessing transforms.
            train_cache_dir = get_or_create_cache_subdirectory(config.paths.cache_dir, preprocess_transforms, split="train")
            
            # 3. Persistent disk cache for preprocessed images (no labels).
            train_image_source = PersistentDataset(data=base_train_ds, transform=preprocess_transforms, cache_dir=train_cache_dir, hash_func=deterministic_hash)
            
            # 4. In-memory cache for faster access.
            if config.cache.memory_rate > 0:
                logger.info(f"Using {config.cache.memory_rate * 100:.0f}% in-memory cache for training.")
                train_image_source = CacheDataset(data=train_image_source, cache_rate=config.cache.memory_rate, num_workers=config.training.num_workers)

            # 5. Attach labels to the cached images on-the-fly.
            train_dataset = LabelAttacherDataset(image_source=train_image_source, labels_df=train_df, pathology_columns=config.pathologies.columns)
            
            # 6. Apply augmentations after labeling.
            if augment_transforms:
                train_dataset = ApplyTransforms(data=train_dataset, transform=augment_transforms)

            # --- Validation Data Pipeline (same logic, no augmentation) ---
            base_valid_ds = CTMetadataDataset(dataframe=valid_df, img_dir=config.paths.img_dir, path_mode=config.paths.dir_structure)
            valid_cache_dir = get_or_create_cache_subdirectory(config.paths.cache_dir, preprocess_transforms, split="valid")
            valid_image_source = PersistentDataset(data=base_valid_ds, transform=preprocess_transforms, cache_dir=valid_cache_dir, hash_func=deterministic_hash)
            if config.cache.memory_rate > 0:
                logger.info(f"Using {config.cache.memory_rate * 100:.0f}% in-memory cache for validation.")
                valid_image_source = CacheDataset(data=valid_image_source, cache_rate=config.cache.memory_rate, num_workers=config.training.num_workers)
            valid_dataset = LabelAttacherDataset(image_source=valid_image_source, labels_df=valid_df, pathology_columns=config.pathologies.columns)

        else:
            logger.info("Caching is disabled. All transforms will be applied on-the-fly.")
            
            # --- Training Data Pipeline (No Caching) ---
            base_train_ds = CTMetadataDataset(dataframe=train_df, img_dir=config.paths.img_dir, path_mode=config.paths.dir_structure)
            train_preprocessed_ds = Dataset(data=base_train_ds, transform=preprocess_transforms)
            train_dataset = LabelAttacherDataset(image_source=train_preprocessed_ds, labels_df=train_df, pathology_columns=config.pathologies.columns)
            
            if augment_transforms:
                train_dataset = ApplyTransforms(data=train_dataset, transform=augment_transforms)

            # --- Validation Data Pipeline (No Caching) ---
            base_valid_ds = CTMetadataDataset(dataframe=valid_df, img_dir=config.paths.img_dir, path_mode=config.paths.dir_structure)
            valid_preprocessed_ds = Dataset(data=base_valid_ds, transform=preprocess_transforms)
            valid_dataset = LabelAttacherDataset(image_source=valid_preprocessed_ds, labels_df=valid_df, pathology_columns=config.pathologies.columns)

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

    if model_type == 'mlp':
        model = create_mlp_classifier(config).to(device)
    elif model_type in {'logistic', 'logistic_regression'}:
        model = create_logistic_classifier(config).to(device)
    else:
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
        grad_scaler_cls = getattr(torch.amp, "GradScaler", None)
        if grad_scaler_cls is not None:
            try:
                scaler = grad_scaler_cls(device_type=device.type)
            except TypeError:
                scaler = grad_scaler_cls()
        if scaler is None:
            scaler = torch.cuda.amp.GradScaler()

    # Initialize training state variables.
    start_epoch = 0
    best_val_metric = 0.0
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
            best_val_metric = checkpoint_metrics_loaded.get('roc_auc_macro', 0.0)
            
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
            # Reports the best ROC AUC (macro) from the loaded checkpoint.
            logger.info(f"Best ROC AUC (macro) from loaded checkpoint: {best_val_metric:.4f}")
        except Exception as e:
            # Handle errors during checkpoint loading.
            logger.error(f"Error loading checkpoint: {e}", exc_info=True)
            logger.info("Starting training from scratch")
            start_epoch = 0
            best_val_metric = 0.0
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
    if start_epoch > 0 and best_val_metric > 0:
        early_stopping.best_value = best_val_metric

    logger.info(f"Starting training from epoch {start_epoch + 1}...")

    # Main training loop.
    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start_time = time.time()
        # Train for one epoch.
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, config.training.num_epochs,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            use_amp=config.optimization.mixed_precision,
            use_bf16=config.optimization.use_bf16
        )
        # Validate for one epoch.
        valid_loss, metrics_for_loop = validate_epoch(
            model, valid_loader, criterion, device, config.pathologies.columns
        )
        # Add validation loss to metrics dictionary
        metrics_for_loop['loss'] = valid_loss

        # Step the learning rate scheduler.
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start_time
        logger.info(f"\nEpoch [{epoch+1}/{config.training.num_epochs}] Time: {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        logger.info(f"Valid F1 (macro): {metrics_for_loop['f1_macro']:.4f}, AUC (macro): {metrics_for_loop['roc_auc_macro']:.4f}")

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
        current_metric = metrics_for_loop.get('roc_auc_macro', 0.0)
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_model_path = output_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, scaler, epoch, metrics_for_loop, best_model_path)
            logger.info(f"New best model saved with ROC AUC (macro): {best_val_metric:.4f}")
        
        # Check for early stopping.
        if early_stopping(current_metric):
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
        logger.info(
            f"Best model: Epoch {history.get('metrics')[best_epoch_idx].get('epoch', best_epoch_idx)+1} with ROC AUC (macro) {best_auc_final:.4f}"
        )
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




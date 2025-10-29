# scripts/train.py
import sys
import argparse
from pathlib import Path
import logging
from types import SimpleNamespace
import torch
import torch.multiprocessing

# Set the multiprocessing sharing strategy to 'file_system' for Linux systems.
# This is a workaround for the "Too many open files" error that can occur when
# using the default 'file_descriptor' strategy with a large number of workers.
if sys.platform == "linux":
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError as e:
        # This might fail if the strategy is already set, which is fine.
        logging.warning(
            "Failed to set multiprocessing sharing strategy, it might already be set. "
            f"Details: {e}"
        )

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

from src.config import load_config
from src.training.trainer import train_model
from src.utils.logging_config import setup_logging
from src.training.utils import find_latest_checkpoint

# scripts/train.py

def main():
    """
    Main function to start or resume the training process for a specific fold.
    It loads a base configuration and allows for overrides via command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the training pipeline for a specific cross-validation fold.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base YAML configuration file.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="The specific fold number to train (e.g., 0, 1, 2...).\n"
             "If not provided, the script uses the default paths in the config file."
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default=None,
        choices=['resnet3d', 'densenet3d', 'vit3d'],
        help='Override the model architecture from the config file.'
    )
    parser.add_argument(
        '--model-variant',
        type=str,
        default=None,
        help='Override the model variant from the config file.'
    )
    parser.add_argument(
        '--workflow',
        type=str,
        default='end-to-end',
        choices=['end-to-end', 'feature-based'],
        help="Set the training workflow. Use 'feature-based' to train on precomputed features."
    )
    parser.add_argument(
        '--resume',
        nargs='?',
        const=True,
        default=None,
        help=(
            "Resume training. \n"
            "- If used as a flag (`--resume`), it automatically finds the latest checkpoint.\n"
            "- If given a path (`--resume path/to/ckpt.pth`), it resumes from that specific file."
        )
    )
    args = parser.parse_args()

    # 1. Load the base configuration
    config = load_config(args.config)

    # 2. Override config with command-line arguments if provided
    if args.model_type:
        logging.info(f"Overriding model.type from '{config.model.type}' to '{args.model_type}'")
        config.model.type = args.model_type
    
    if args.model_variant:
        logging.info(f"Overriding model.variant from '{config.model.variant}' to '{args.model_variant}'")
        config.model.variant = args.model_variant

    # Workflow-level override (only --workflow is supported from CLI)
    if args.workflow:
        prev_workflow = getattr(config, 'workflow', SimpleNamespace(mode='end-to-end'))
        logging.info(f"Overriding workflow.mode from '{getattr(prev_workflow, 'mode', None)}' to '{args.workflow}'")
        if not hasattr(config, 'workflow'):
            config.workflow = SimpleNamespace()
        config.workflow.mode = args.workflow

    # If feature-based mode is selected, read the feature_dir from config.paths.feature_dir
    if getattr(config, 'workflow', None) and getattr(config.workflow, 'mode', None) == 'feature-based':
        if not hasattr(config.workflow, 'feature_config'):
            config.workflow.feature_config = SimpleNamespace()
        # Use the feature_dir defined under config.paths (populated via env var)
        if hasattr(config.paths, 'feature_dir'):
            logging.info(f"Using feature directory from config.paths.feature_dir: {config.paths.feature_dir}")
            config.workflow.feature_config.feature_dir = config.paths.feature_dir
        else:
            logging.error("Feature-based workflow selected but 'paths.feature_dir' is not set in the config.")
            return

        # Disable augmentations because they are incompatible with precomputed features
        if hasattr(config, 'training') and hasattr(config.training, 'augment'):
            logging.info('Disabling on-the-fly augmentations for feature-based workflow')
            config.training.augment = False
        # On Windows, DataLoader workers commonly hang when loading complex objects.
        # Force a safe single-process data loading configuration to avoid hangs.
        try:
            if sys.platform.startswith('win'):
                if hasattr(config.training, 'num_workers') and config.training.num_workers > 0:
                    logging.warning(
                        "Windows detected: overriding training.num_workers>0 to 0 and disabling pin_memory to avoid DataLoader hangs for feature-based workflow."
                    )
                    config.training.num_workers = 0
                    if hasattr(config.training, 'pin_memory'):
                        config.training.pin_memory = False
        except Exception:
            # don't crash on unexpected config shapes; this is a best-effort mitigation
            pass
        # --- Sanity checks: ensure feature dir and required subfolders exist ---
        try:
            from pathlib import Path as _Path
            feature_root = _Path(config.workflow.feature_config.feature_dir)
            if not feature_root.exists():
                logging.error(f"Feature directory does not exist: {feature_root}")
                return

            train_sub = feature_root / 'train'
            valid_sub = feature_root / 'valid'

            # Accept either layout A: feature_root/{train,valid} OR layout B: feature_root contains all .pt files
            has_subfolders = train_sub.exists() and valid_sub.exists()
            has_pt_files_at_root = any(feature_root.glob('*.pt'))

            if not has_subfolders and not has_pt_files_at_root:
                logging.error(
                    "Feature directory must either contain 'train' and 'valid' subfolders, "
                    "or contain .pt feature files at its top level. "
                    f"Checked: {feature_root}"
                )
                return

            if has_subfolders:
                logging.info("Detected feature layout with 'train' and 'valid' subfolders.")
            else:
                logging.info("Detected flattened feature layout: using the same feature dir for both train and valid.")

        except Exception as e:
            logging.error(f"Error while checking feature directory: {e}")
            return

    # 3. Dynamically set paths based on the specified fold
    if args.fold is not None:
        logger.info(f"Running training for fold: {args.fold}")
        
        # Get the directory where fold splits are stored
        # Assumes the original config path is something like '.../train_fold_0.csv'
        original_train_path = Path(config.paths.data_subsets.train)
        folds_dir = original_train_path.parent

        # Update the train and validation paths for the specific fold
        config.paths.data_subsets.train = str(folds_dir / f"train_fold_{args.fold}.csv")
        config.paths.data_subsets.valid = str(folds_dir / f"valid_fold_{args.fold}.csv")
        logger.info(f"Using training data: {config.paths.data_subsets.train}")
        logger.info(f"Using validation data: {config.paths.data_subsets.valid}")

    # Modify the output directory to be fold-specific only if a fold was provided.
    # This prevents folds from overwriting each other's checkpoints and logs.
    base_output_dir = Path(config.paths.output_dir)
    if args.fold is not None:
        # Keep output_dir as a Path object (other code expects Path operations)
        config.paths.output_dir = base_output_dir / f"fold_{args.fold}"
    else:
        # Ensure output_dir is a Path for downstream usage
        config.paths.output_dir = base_output_dir
    logger.info(f"Output will be saved to: {config.paths.output_dir}")

    # 4. Handle resume logic
    if args.resume is not None:
        checkpoint_path = None
        # The output_dir has already been updated to the fold-specific one
        output_dir = Path(config.paths.output_dir)
        
        if args.resume is True:
            # Automatic resume: find the latest checkpoint in the fold's output directory
            logging.info(f"Attempting to resume from the latest checkpoint in {output_dir}...")
            checkpoint_path = find_latest_checkpoint(output_dir)
            if not checkpoint_path:
                logging.warning(
                    f"Automatic resume failed: no checkpoint found in {output_dir}. "
                    "Starting a new training session for this fold."
                )
        else:
            # Specific resume: use the provided path
            logging.info(f"Attempting to resume from specific checkpoint: {args.resume}")
            checkpoint_path = Path(args.resume)
            if not checkpoint_path.exists():
                logging.error(f"Resume failed: Specified checkpoint not found at {checkpoint_path}")
                return # Exit if specific checkpoint is not found

        if checkpoint_path:
            config.training.resume_from_checkpoint = str(checkpoint_path)
            logging.info(f"Training will resume from: {checkpoint_path}")

    # 5. Setup logging and start training
    log_dir = Path(config.paths.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=log_dir / 'training.log')
    
    logging.info(f"Configuration for fold {args.fold} loaded and processed. Starting training.")
    train_model(config)

if __name__ == "__main__":
    # Set the start method to 'spawn' for CUDA compatibility in multiprocessing.
    # 'spawn' is required for using CUDA in subprocesses.
    # torch.multiprocessing.set_start_method('spawn', force=True)
    main()
# scripts/train.py
import sys
import argparse
from pathlib import Path
import logging
import torch
import torch.multiprocessing

# Set the multiprocessing sharing strategy to 'file_system' for Linux systems.
# This is a workaround for the "Too many open files" error that can occur when
# using the default 'file_descriptor' strategy with a large number of workers.
# if sys.platform == "linux":
#     try:
#         torch.multiprocessing.set_sharing_strategy('file_system')
#     except RuntimeError as e:
#         # This might fail if the strategy is already set, which is fine.
#         logging.warning(
#             "Failed to set multiprocessing sharing strategy, it might already be set. "
#             f"Details: {e}"
#         )

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import load_config
from src.training.trainer import train_model
from src.utils.logging_config import setup_logging
from src.training.utils import find_latest_checkpoint

def main():
    """
    Main function to start or resume the training process.
    It loads a base configuration and allows for overrides via command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the training pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base YAML configuration file.",
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

    # 2. Override config with command-line arguments
    if args.model_type:
        logging.info(f"Overriding model.type from '{config.model.type}' to '{args.model_type}'")
        config.model.type = args.model_type
    
    if args.model_variant:
        logging.info(f"Overriding model.variant from '{config.model.variant}' to '{args.model_variant}'")
        config.model.variant = args.model_variant

    # 3. Handle resume logic
    if args.resume is not None:
        checkpoint_path = None
        if args.resume is True:
            # Automatic resume: find the latest checkpoint
            logging.info("Attempting to resume from the latest checkpoint...")
            checkpoint_path = find_latest_checkpoint(config.paths.output_dir)
            if not checkpoint_path:
                logging.warning(
                    f"Automatic resume failed: no checkpoint found in {config.paths.output_dir}. "
                    "Starting a new training session."
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

    # 4. Setup logging and start training
    log_dir = config.paths.output_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=log_dir / 'ct_3d_training.log')
    
    logging.info("Configuration loaded and processed. Starting training.")
    train_model(config)

if __name__ == "__main__":
    # Set the start method to 'spawn' for CUDA compatibility in multiprocessing.
    # This must be done within the `if __name__ == '__main__':` block.
    # 'spawn' is required for using CUDA in subprocesses.
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import load_config
from src.training.trainer import train_model
from src.utils.logging_config import setup_logging
from src.training.utils import find_latest_checkpoint
import logging


def main():
    """
    Main function to start or resume the training process.
    It loads a base configuration from a YAML file and allows for
    overrides via command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
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
        action='store_true',
        help='Resume training from the latest checkpoint in the output directory.'
    )
    args = parser.parse_args()

    # 1. Load the base configuration from the specified YAML file
    config = load_config(args.config)

    # 2. Override the configuration with command-line arguments if provided
    if args.model_type:
        logging.info(f"Overriding model.type from '{config.model.type}' to '{args.model_type}'")
        config.model.type = args.model_type
    
    if args.model_variant:
        logging.info(f"Overriding model.variant from '{config.model.variant}' to '{args.model_variant}'")
        config.model.variant = args.model_variant

    if args.resume:
        logging.info("Attempting to resume from the latest checkpoint...")
        latest_checkpoint = find_latest_checkpoint(config.paths.output_dir)
        if latest_checkpoint:
            config.training.resume_from_checkpoint = latest_checkpoint
            logging.info(f"Found checkpoint. Resuming from: {latest_checkpoint}")
        else:
            logging.warning(
                f"--resume flag was used, but no checkpoint was found in {config.paths.output_dir}. "
                "Starting a new training session."
            )
    
    # 3. Setup logging and start training
    # The output directory is now defined in the config, so we can create the log file there
    log_file = config.paths.output_dir / "training_run.log"
    setup_logging(log_file_path=log_file)
    
    logging.info("Configuration loaded and processed. Starting training.")
    train_model(config)


if __name__ == "__main__":
    main()
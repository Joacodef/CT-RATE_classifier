#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path to allow imports from `src`
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import load_config
from src.training.trainer import train_model
from src.utils.logging_config import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file for the training run.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the checkpoint file (.pth) to resume from.'
    )
    args = parser.parse_args()
    
    # Load the base configuration from the specified YAML file
    config = load_config(args.config)
    
    # Set the checkpoint path on the loaded config object
    config.training.resume_from_checkpoint = args.checkpoint
    
    # Setup logging to use the output directory from the loaded config
    setup_logging(log_file_path=config.paths.output_dir / "training_resume.log")
    logger = logging.getLogger(__name__)
    
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Resuming training from checkpoint: {args.checkpoint}")
    
    # Start the training process, which will now resume from the checkpoint
    train_model(config)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
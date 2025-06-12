#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add project root and src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from config import Config
from training.trainer import train_model
from utils.logging_config import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    args = parser.parse_args()
    
    logger = setup_logging()
    config = Config()
    config.RESUME_FROM_CHECKPOINT = args.checkpoint
    
    logger.info(f"Resuming training from: {args.checkpoint}")
    model, history = train_model(config)
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
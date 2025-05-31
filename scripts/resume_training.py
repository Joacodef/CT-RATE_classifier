#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
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
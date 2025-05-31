#!/usr/bin/env python3
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from training.trainer import train_model
from training.utils import find_latest_checkpoint
from utils.logging_config import setup_logging

def main():
    # Setup logging
    logger = setup_logging()
    
    # Initialize configuration
    config = Config()
    
    # Find latest checkpoint if exists
    latest_checkpoint = find_latest_checkpoint(config.OUTPUT_DIR)
    if latest_checkpoint:
        config.RESUME_FROM_CHECKPOINT = str(latest_checkpoint)
        logger.info(f"Found checkpoint: {latest_checkpoint}")
    
    # Train model
    model, history = train_model(config)
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
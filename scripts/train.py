#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from training.trainer import train_model
from training.utils import find_latest_checkpoint
from utils.logging_config import setup_logging

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CT 3D Classifier')
    parser.add_argument('--model-type', type=str, default=None,
                       choices=['resnet3d', 'densenet3d', 'vit3d'],
                       help='Model architecture to use')
    parser.add_argument('--model-variant', type=str, default=None,
                       help='Model variant (e.g., 18, 34 for ResNet; tiny, small, base for ViT)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Initialize configuration
    config = Config()
    
    # Override model settings from command line if provided
    if args.model_type:
        config.MODEL_TYPE = args.model_type
        logger.info(f"Model type overridden to: {args.model_type}")
    
    if args.model_variant:
        config.MODEL_VARIANT = args.model_variant
        logger.info(f"Model variant set to: {args.model_variant}")
    
    # Find latest checkpoint if requested
    if args.resume:
        latest_checkpoint = find_latest_checkpoint(config.OUTPUT_DIR)
        if latest_checkpoint:
            config.RESUME_FROM_CHECKPOINT = str(latest_checkpoint)
            logger.info(f"Found checkpoint: {latest_checkpoint}")
    
    # Train model
    model, history = train_model(config)
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Export trained model to different formats (ONNX, TorchScript, etc.)
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.onnx
import logging

# Add the project root to the Python path to allow imports from `src`
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import load_config
from src.training.trainer import create_model
from src.utils.logging_config import setup_logging

def export_to_onnx(model, dummy_input, output_path: str):
    """Export model to ONNX format"""
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

def export_to_torchscript(model, dummy_input, output_path: str):
    """Export model to TorchScript format"""
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)

def main():
    parser = argparse.ArgumentParser(description='Export trained model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file used for model training.'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model checkpoint (.pth file).'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='onnx',
        choices=['onnx', 'torchscript', 'both'],
        help='The format to export the model to.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='exported_models',
        help='Directory to save the exported model(s).'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for model loading and export.'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file_path=output_dir / "model_export.log")
    logger = logging.getLogger(__name__)
    
    # Load configuration from the specified YAML file
    config = load_config(args.config)
    device = torch.device(args.device)
    
    # Create model using the centralized create_model function
    logger.info(f"Creating model architecture: {config.model.type} (variant: {config.model.variant})")
    # Ensure gradient checkpointing is disabled for export
    config.optimization.gradient_checkpointing = False
    model = create_model(config)
    
    # Load model weights
    logger.info(f"Loading model weights from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input based on the shape defined in the config
    dummy_input_shape = (1, 1, *config.image_processing.target_shape_dhw)
    dummy_input = torch.randn(dummy_input_shape, device=device)
    logger.info(f"Created dummy input of shape: {dummy_input_shape}")
    
    # Define model name for exported files
    model_name = Path(args.model).stem
    
    # Export to the requested format(s)
    if args.format in ['onnx', 'both']:
        onnx_path = output_dir / f"{model_name}.onnx"
        logger.info(f"Exporting to ONNX format at: {onnx_path}")
        export_to_onnx(model, dummy_input, str(onnx_path))
        logger.info("ONNX export completed.")
    
    if args.format in ['torchscript', 'both']:
        torchscript_path = output_dir / f"{model_name}.pt"
        logger.info(f"Exporting to TorchScript format at: {torchscript_path}")
        export_to_torchscript(model, dummy_input, str(torchscript_path))
        logger.info("TorchScript export completed.")
    
    logger.info("Model export finished successfully.")

if __name__ == "__main__":
    main()
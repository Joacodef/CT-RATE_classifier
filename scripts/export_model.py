#!/usr/bin/env python3
"""
Export trained model to different formats (ONNX, TorchScript, etc.)
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.onnx

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.resnet3d import resnet18_3d
from utils.logging_config import setup_logging

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
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--format', type=str, default='onnx',
                       choices=['onnx', 'torchscript', 'both'],
                       help='Export format')
    parser.add_argument('--output-dir', type=str, default='exported_models',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    logger = setup_logging('model_export.log')
    config = Config()
    device = torch.device(args.device)
    
    # Load model
    model = resnet18_3d(num_classes=config.NUM_PATHOLOGIES, use_checkpointing=False)
    checkpoint = torch.load(args.model, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, *config.TARGET_SHAPE_DHW, device=device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(args.model).stem
    
    # Export to requested format(s)
    if args.format in ['onnx', 'both']:
        onnx_path = output_dir / f"{model_name}.onnx"
        logger.info(f"Exporting to ONNX: {onnx_path}")
        export_to_onnx(model, dummy_input, str(onnx_path))
        logger.info("ONNX export completed")
    
    if args.format in ['torchscript', 'both']:
        torchscript_path = output_dir / f"{model_name}.pt"
        logger.info(f"Exporting to TorchScript: {torchscript_path}")
        export_to_torchscript(model, dummy_input, str(torchscript_path))
        logger.info("TorchScript export completed")
    
    logger.info("Model export completed successfully")

if __name__ == "__main__":
    main()
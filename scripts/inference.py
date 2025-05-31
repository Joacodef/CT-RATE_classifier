#!/usr/bin/env python3
"""
Run inference on individual CT volumes
"""

import sys
import argparse
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.resnet3d import resnet18_3d
from data.preprocessing import preprocess_ct_volume
from utils.logging_config import setup_logging

class CTInference:
    """Single volume inference class"""
    
    def __init__(self, model_path: str, config: Config, device: str = 'auto'):
        self.config = config
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.logger = setup_logging('inference.log')
    
    def _setup_device(self, device: str) -> torch.device:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        model = resnet18_3d(num_classes=self.config.NUM_PATHOLOGIES, use_checkpointing=False)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        return model
    
    @torch.no_grad()
    def predict_volume(self, volume_path: str, return_probabilities: bool = True) -> dict:
        """Predict pathologies for a single CT volume"""
        
        volume_path = Path(volume_path)
        if not volume_path.exists():
            raise FileNotFoundError(f"Volume not found: {volume_path}")
        
        # Preprocess volume
        tensor = preprocess_ct_volume(
            volume_path,
            self.config.TARGET_SPACING,
            self.config.TARGET_SHAPE_DHW,
            self.config.CLIP_HU_MIN,
            self.config.CLIP_HU_MAX
        )
        
        # Add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        outputs = self.model(tensor)
        
        # Convert to probabilities
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Create results dictionary
        results = {
            'volume_path': str(volume_path),
            'predictions': {}
        }
        
        for i, pathology in enumerate(self.config.PATHOLOGY_COLUMNS):
            prob = float(probabilities[i])
            results['predictions'][pathology] = {
                'probability': prob,
                'prediction': int(prob > 0.5),
                'confidence': max(prob, 1 - prob)
            }
        
        return results
    
    def predict_batch(self, volume_paths: list, output_file: str = None) -> pd.DataFrame:
        """Predict pathologies for multiple volumes"""
        
        results = []
        
        for i, volume_path in enumerate(volume_paths):
            self.logger.info(f"Processing {i+1}/{len(volume_paths)}: {volume_path}")
            
            try:
                result = self.predict_volume(volume_path)
                
                # Flatten results for DataFrame
                row = {'volume_path': result['volume_path']}
                for pathology, pred_data in result['predictions'].items():
                    row[f'{pathology}_probability'] = pred_data['probability']
                    row[f'{pathology}_prediction'] = pred_data['prediction']
                    row[f'{pathology}_confidence'] = pred_data['confidence']
                
                results.append(row)
                
            except Exception as e:
                self.logger.error(f"Error processing {volume_path}: {e}")
                # Add row with NaN values
                row = {'volume_path': str(volume_path)}
                for pathology in self.config.PATHOLOGY_COLUMNS:
                    row[f'{pathology}_probability'] = np.nan
                    row[f'{pathology}_prediction'] = np.nan
                    row[f'{pathology}_confidence'] = np.nan
                results.append(row)
        
        results_df = pd.DataFrame(results)
        
        if output_file:
            results_df.to_csv(output_file, index=False)
            self.logger.info(f"Results saved to: {output_file}")
        
        return results_df

def main():
    parser = argparse.ArgumentParser(description='Run inference on CT volumes')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to CT volume or directory containing volumes')
    parser.add_argument('--output', type=str, default='inference_results.csv',
                       help='Output CSV file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold')
    
    args = parser.parse_args()
    
    config = Config()
    inference = CTInference(args.model, config, args.device)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single volume
        result = inference.predict_volume(str(input_path))
        
        print(f"\nPredictions for {input_path.name}:")
        print("=" * 50)
        
        for pathology, pred_data in result['predictions'].items():
            status = "POSITIVE" if pred_data['prediction'] else "NEGATIVE"
            print(f"{pathology:30s} | {pred_data['probability']:.3f} | {status}")
        
        # Save as JSON
        output_json = Path(args.output).with_suffix('.json')
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)
        
    elif input_path.is_dir():
        # Multiple volumes
        volume_paths = list(input_path.glob("*.nii.gz"))
        
        if not volume_paths:
            print(f"No .nii.gz files found in {input_path}")
            return
        
        results_df = inference.predict_batch([str(p) for p in volume_paths], args.output)
        print(f"\nProcessed {len(results_df)} volumes")
        print(f"Results saved to: {args.output}")
        
    else:
        print(f"Input path not found: {input_path}")

if __name__ == "__main__":
    main()

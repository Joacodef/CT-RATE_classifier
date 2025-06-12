#!/usr/bin/env python3
"""
Run inference on individual CT volumes using a MONAI-based preprocessing pipeline.
"""

import sys
import argparse
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Add project root and src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from config.config import Config
from models.resnet3d import resnet18_3d # Assuming ResNet-18 is still the target for inference
# Updated imports for MONAI preprocessing
from data.preprocessing import (
    create_monai_preprocessing_pipeline,
    preprocess_ct_volume_monai
)
from utils.logging_config import setup_logging

class CTInference:
    """Single volume inference class using MONAI for preprocessing."""

    def __init__(self, model_path: str, config: Config, device: str = 'auto', orientation_axcodes: str = "LPS"):
        """
        Initializes the CTInference instance.

        Args:
            model_path: Path to the trained model checkpoint.
            config: Configuration object.
            device: Device to run inference on ('auto', 'cuda', 'cpu').
            orientation_axcodes: Target orientation for MONAI's Orientationd transform.
        """
        self.config = config
        self.device = self._setup_device(device)
        self.logger = setup_logging('inference.log') # Setup logger instance variable
        self.model = self._load_model(model_path)

        # Initialize the MONAI preprocessing pipeline
        self.monai_pipeline = create_monai_preprocessing_pipeline(
            target_spacing_xyz=self.config.TARGET_SPACING,
            target_shape_dhw=self.config.TARGET_SHAPE_DHW,
            clip_hu_min=self.config.CLIP_HU_MIN,
            clip_hu_max=self.config.CLIP_HU_MAX,
            orientation_axcodes=orientation_axcodes
        )
        self.logger.info(f"CTInference initialized with MONAI pipeline. Orientation: {orientation_axcodes}")


    def _setup_device(self, device_option: str) -> torch.device: # Renamed 'device' to 'device_option' to avoid conflict
        """Sets up the computation device."""
        if device_option == 'auto':
            selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            selected_device = device_option
        self.logger.info(f"Using device: {selected_device}") # Log the selected device
        return torch.device(selected_device)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Loads the trained model from a checkpoint."""
        # Create model architecture.
        # For now, this still specifically loads resnet18_3d.
        # Future improvement: Load model based on config.MODEL_TYPE.
        model = resnet18_3d(num_classes=self.config.NUM_PATHOLOGIES, use_checkpointing=False)
        self.logger.info(f"Loading ResNet-18 3D model for inference.")

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                self.logger.info(f"Loaded model state_dict from epoch {epoch} of {model_path}")
            else:
                # This case assumes the checkpoint is just the state_dict.
                model.load_state_dict(checkpoint)
                self.logger.info(f"Loaded model state_dict directly from {model_path}")

            model = model.to(self.device)
            model.eval() # Set model to evaluation mode.
            self.logger.info(f"Model successfully loaded onto {self.device} and set to eval mode.")
            return model
        except FileNotFoundError:
            self.logger.error(f"Model checkpoint file not found: {model_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            raise


    @torch.no_grad() # Disables gradient calculations during inference.
    def predict_volume(self, volume_path_str: str) -> dict: # Renamed volume_path to volume_path_str
        """
        Predicts pathologies for a single CT volume using the MONAI pipeline.

        Args:
            volume_path_str: String path to the CT volume file.

        Returns:
            A dictionary containing predictions and probabilities for each pathology.
        """
        volume_path = Path(volume_path_str) # Convert string to Path object
        if not volume_path.exists():
            self.logger.error(f"Volume not found: {volume_path}")
            raise FileNotFoundError(f"Volume not found: {volume_path}")

        self.logger.info(f"Preprocessing volume: {volume_path}")
        # Preprocess volume using the MONAI pipeline
        tensor = preprocess_ct_volume_monai(
            nii_path=volume_path,
            preprocessing_pipeline=self.monai_pipeline,
            target_shape_dhw=self.config.TARGET_SHAPE_DHW # For error fallback
        )

        # Add batch dimension if not already [1, C, D, H, W]
        # preprocess_ct_volume_monai should return [1, D, H, W]
        # For MONAI, ensure C is 1, so [1, 1, D, H, W] if model expects 5D with C=1
        # Assuming the model input is [B, C, D, H, W], and preprocess_ct_volume_monai returns [1, D, H, W] (single channel)
        # The current resnet3d.py conv1 expects nn.Conv3d(1, 64, ...), so input should be [B, 1, D, H, W]
        # The `preprocess_ct_volume_monai` already returns it with a channel dim of 1.
        # So tensor shape is [1, D, H, W]. For batch processing, it needs to be [N, 1, D, H, W].
        # Here, N=1 for single volume prediction.

        # No unsqueeze(0) needed here if preprocess_ct_volume_monai already returns [1,D,H,W] and model expects [B,1,D,H,W]
        # The batch dimension is implicitly handled by DataLoader or added here for single inference.
        # For a single prediction, it becomes the batch dimension.
        input_tensor = tensor.to(self.device) # tensor is already [1, D, H, W]

        self.logger.info(f"Running inference on processed tensor with shape: {input_tensor.shape}")
        # Run inference
        outputs = self.model(input_tensor)

        # Convert to probabilities
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0] # Get the first (and only) batch item

        # Create results dictionary
        results = {
            'volume_path': str(volume_path),
            'predictions': {}
        }

        for i, pathology in enumerate(self.config.PATHOLOGY_COLUMNS):
            prob = float(probabilities[i])
            # Prediction based on a 0.5 threshold, could be made configurable
            prediction = int(prob > 0.5)
            confidence = max(prob, 1 - prob) # Simple confidence measure

            results['predictions'][pathology] = {
                'probability': prob,
                'prediction': prediction,
                'confidence': confidence
            }
        self.logger.info(f"Prediction complete for {volume_path}")
        return results

    def predict_batch(self, volume_paths: list, output_file: str = None) -> pd.DataFrame:
        """Predict pathologies for multiple volumes"""

        all_results_data = [] # Store dictionaries for DataFrame creation

        for i, volume_path_str in enumerate(volume_paths):
            self.logger.info(f"Processing {i+1}/{len(volume_paths)}: {volume_path_str}")

            try:
                # Use predict_volume for each file path
                result_dict = self.predict_volume(volume_path_str)

                # Flatten results for DataFrame
                row_data = {'volume_path': result_dict['volume_path']}
                for pathology, pred_data in result_dict['predictions'].items():
                    row_data[f'{pathology}_probability'] = pred_data['probability']
                    row_data[f'{pathology}_prediction'] = pred_data['prediction']
                    row_data[f'{pathology}_confidence'] = pred_data['confidence']
                all_results_data.append(row_data)

            except Exception as e:
                self.logger.error(f"Error processing {volume_path_str}: {e}")
                # Add row with NaN values for errored files
                error_row = {'volume_path': str(volume_path_str)}
                for pathology in self.config.PATHOLOGY_COLUMNS:
                    error_row[f'{pathology}_probability'] = np.nan
                    error_row[f'{pathology}_prediction'] = np.nan
                    error_row[f'{pathology}_confidence'] = np.nan
                all_results_data.append(error_row)

        results_df = pd.DataFrame(all_results_data)

        if output_file:
            try:
                results_df.to_csv(output_file, index=False)
                self.logger.info(f"Batch prediction results saved to: {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save results to {output_file}: {e}")

        return results_df

def main():
    parser = argparse.ArgumentParser(description='Run inference on CT volumes')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to CT volume or directory containing volumes (.nii.gz)')
    parser.add_argument('--output', type=str, default='inference_results.csv',
                       help='Output CSV file for batch processing, or base name for single JSON.')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'])
    # Threshold argument is not directly used in predict_volume's core logic for raw probabilities,
    # but prediction (0/1) is derived using 0.5. This could be passed to CTInference if desired.
    # parser.add_argument('--threshold', type=float, default=0.5,
    #                    help='Prediction threshold')
    parser.add_argument('--orientation', type=str, default="LPS",
                        help="Target orientation for MONAI preprocessing (e.g., LPS, RAS).")


    args = parser.parse_args()
    logger = setup_logging(log_file='inference_main.log') # Setup logger for main function as well

    config = Config()
    # Pass orientation from CLI args to CTInference
    inference = CTInference(args.model, config, args.device, orientation_axcodes=args.orientation)

    input_path = Path(args.input)

    if input_path.is_file():
        if not input_path.name.endswith((".nii", ".nii.gz")): # Basic check for NIfTI
            logger.error(f"Input file {input_path} does not appear to be a NIfTI file (.nii or .nii.gz).")
            return

        logger.info(f"Performing inference on single volume: {input_path}")
        try:
            result = inference.predict_volume(str(input_path))

            print(f"\nPredictions for {input_path.name}:")
            print("=" * 50)

            for pathology, pred_data in result['predictions'].items():
                status = "POSITIVE" if pred_data['prediction'] else "NEGATIVE"
                print(f"{pathology:30s} | Prob: {pred_data['probability']:.3f} | Pred: {status} | Conf: {pred_data['confidence']:.3f}")

            # Save as JSON for single file inference
            output_json_path = Path(args.output).with_suffix('.json')
            output_json_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
            with open(output_json_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Single volume prediction saved to: {output_json_path}")

        except Exception as e:
            logger.error(f"Failed to process single volume {input_path}: {e}")


    elif input_path.is_dir():
        logger.info(f"Performing inference on directory: {input_path}")
        # Glob for .nii and .nii.gz files
        volume_paths = list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.nii"))

        if not volume_paths:
            logger.warning(f"No .nii or .nii.gz files found in {input_path}")
            print(f"No .nii or .nii.gz files found in {input_path}")
            return

        logger.info(f"Found {len(volume_paths)} volumes for batch processing.")
        # Ensure output filename for batch is .csv
        output_csv_path = Path(args.output).with_suffix('.csv')
        output_csv_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists

        results_df = inference.predict_batch([str(p) for p in volume_paths], str(output_csv_path))
        if not results_df.empty:
            print(f"\nProcessed {len(results_df)} volumes.")
            print(f"Batch results saved to: {output_csv_path}")
        else:
            print("No results generated from batch processing.")
            logger.warning("Batch processing resulted in an empty DataFrame.")

    else:
        logger.error(f"Input path not found or is not a file/directory: {input_path}")
        print(f"Input path not found or is not a file/directory: {input_path}")

if __name__ == "__main__":
    main()
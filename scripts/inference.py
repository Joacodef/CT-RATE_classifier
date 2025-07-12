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
import logging
from types import SimpleNamespace
from tqdm import tqdm

# Add the project root to the Python path to allow imports from `src`
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import load_config
from src.training.trainer import create_model
from src.utils.logging_config import setup_logging

# MONAI imports for preprocessing
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    Resized,
    EnsureTyped,
)

class CTInference:
    """Single volume inference class using MONAI for preprocessing."""

    def __init__(self, model_path: str, config: SimpleNamespace, device: str = 'auto'):
        """
        Initializes the CTInference instance.

        Args:
            model_path: Path to the trained model checkpoint.
            config: Configuration object, typically loaded from the training config.
            device: Device to run inference on ('auto', 'cuda', 'cpu').
        """
        self.config = config
        self.device = self._setup_device(device)
        self.logger = logging.getLogger(__name__)
        self.model = self._load_model(model_path)

        # Define the MONAI preprocessing pipeline directly.
        # This ensures consistency with the validation pipeline in the trainer.
        self.monai_pipeline = Compose([
            LoadImaged(keys="image", image_only=True, ensure_channel_first=True, reader="NibabelReader"),
            Orientationd(keys="image", axcodes=self.config.image_processing.orientation_axcodes),
            Spacingd(keys="image", pixdim=self.config.image_processing.target_spacing, mode="bilinear"),
            ScaleIntensityRanged(
                keys="image",
                a_min=self.config.image_processing.clip_hu_min,
                a_max=self.config.image_processing.clip_hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            Resized(keys="image", spatial_size=self.config.image_processing.target_shape_dhw, mode="area"),
            EnsureTyped(keys="image", dtype=torch.float32)
        ])
        self.logger.info("CTInference initialized with a MONAI preprocessing pipeline.")


    def _setup_device(self, device_option: str) -> torch.device:
        """Sets up the computation device."""
        if device_option == 'auto':
            selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            selected_device = device_option
        
        device = torch.device(selected_device)
        self.logger.info(f"Using device: {device}")
        return device

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Loads the trained model from a checkpoint."""
        self.logger.info(
            f"Creating model architecture: {self.config.model.type} "
            f"(variant: {self.config.model.variant}) for inference."
        )
        # Ensure gradient checkpointing is disabled for inference
        self.config.optimization.gradient_checkpointing = False
        model = create_model(self.config)

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle both dictionary-based checkpoints and raw state_dict files
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'N/A')
                self.logger.info(f"Loaded model state_dict from epoch {epoch} of {model_path}")
            else:
                model.load_state_dict(checkpoint)
                self.logger.info(f"Loaded model state_dict directly from {model_path}")

            model = model.to(self.device)
            model.eval()
            self.logger.info(f"Model successfully loaded onto {self.device} and set to eval mode.")
            return model
        except FileNotFoundError:
            self.logger.error(f"Model checkpoint file not found: {model_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            raise


    @torch.no_grad()
    def predict_volume(self, volume_path_str: str) -> dict:
        """
        Predicts pathologies for a single CT volume using the MONAI pipeline.

        Args:
            volume_path_str: String path to the CT volume file.

        Returns:
            A dictionary containing predictions and probabilities for each pathology.
        """
        volume_path = Path(volume_path_str)
        if not volume_path.exists():
            self.logger.error(f"Volume not found: {volume_path}")
            raise FileNotFoundError(f"Volume not found: {volume_path}")

        self.logger.info(f"Preprocessing volume: {volume_path}")
        
        # Create a data dictionary and apply the MONAI pipeline
        data_dict = {'image': volume_path}
        processed_dict = self.monai_pipeline(data_dict)
        tensor = processed_dict['image']

        # Add a batch dimension and send to the correct device
        input_tensor = tensor.unsqueeze(0).to(self.device)

        self.logger.info(f"Running inference on processed tensor with shape: {input_tensor.shape}")
        
        # Run inference
        outputs = self.model(input_tensor)

        # Convert logits to probabilities
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

        # Format results
        results = {
            'volume_path': str(volume_path),
            'predictions': {}
        }

        for i, pathology in enumerate(self.config.pathologies.columns):
            prob = float(probabilities[i])
            prediction = int(prob > 0.5) # Threshold can be adjusted
            
            results['predictions'][pathology] = {
                'probability': prob,
                'prediction': prediction
            }
        self.logger.info(f"Prediction complete for {volume_path}")
        return results

    def predict_batch(self, volume_paths: list, output_file: str = None) -> pd.DataFrame:
        """Predicts pathologies for a list of volume paths."""
        all_results_data = []

        for i, volume_path_str in enumerate(tqdm(volume_paths, desc="Batch Inference")):
            self.logger.info(f"Processing {i+1}/{len(volume_paths)}: {volume_path_str}")
            try:
                result_dict = self.predict_volume(volume_path_str)
                row_data = {'volume_path': result_dict['volume_path']}
                for pathology, pred_data in result_dict['predictions'].items():
                    row_data[f'{pathology}_probability'] = pred_data['probability']
                    row_data[f'{pathology}_prediction'] = pred_data['prediction']
                all_results_data.append(row_data)

            except Exception as e:
                self.logger.error(f"Failed to process {volume_path_str}: {e}")
                error_row = {'volume_path': str(volume_path_str)}
                for pathology in self.config.pathologies.columns:
                    error_row[f'{pathology}_probability'] = np.nan
                    error_row[f'{pathology}_prediction'] = np.nan
                all_results_data.append(error_row)

        results_df = pd.DataFrame(all_results_data)

        if output_file:
            try:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(output_file, index=False)
                self.logger.info(f"Batch prediction results saved to: {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save results to {output_file}: {e}")

        return results_df

def main():
    parser = argparse.ArgumentParser(description='Run inference on CT volumes')
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to the YAML configuration file used during model training.'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained model checkpoint (.pth file).'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to a single CT volume (.nii.gz) or a directory of volumes.'
    )
    parser.add_argument(
        '--output', type=str, default='inference_results',
        help='Base name for the output file(s) (e.g., .csv for batch, .json for single).'
    )
    parser.add_argument(
        '--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu']
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file_path=Path(args.output).parent / 'inference.log')
    logger = logging.getLogger(__name__)

    try:
        config = load_config(args.config)
        
        inference = CTInference(
            model_path=args.model,
            config=config,
            device=args.device
        )

        input_path = Path(args.input)

        if input_path.is_file():
            if not input_path.name.endswith((".nii", ".nii.gz")):
                logger.error(f"Input file {input_path} is not a NIfTI file.")
                return

            logger.info(f"Performing inference on single volume: {input_path}")
            result = inference.predict_volume(str(input_path))

            print(f"\nPredictions for {input_path.name}:")
            print("=" * 50)
            for pathology, pred_data in result['predictions'].items():
                status = "POSITIVE" if pred_data['prediction'] else "NEGATIVE"
                print(f"{pathology:40s} | Prob: {pred_data['probability']:.4f} | Pred: {status}")

            output_json_path = Path(args.output).with_suffix('.json')
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, 'w') as f:
                json.dump(result, f, indent=4)
            logger.info(f"Single volume prediction saved to: {output_json_path}")

        elif input_path.is_dir():
            logger.info(f"Scanning directory for volumes: {input_path}")
            volume_paths = list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.nii"))
            
            if not volume_paths:
                logger.warning(f"No .nii or .nii.gz files found in {input_path}")
                return

            logger.info(f"Found {len(volume_paths)} volumes for batch processing.")
            output_csv_path = Path(args.output).with_suffix('.csv')
            
            results_df = inference.predict_batch([str(p) for p in volume_paths], str(output_csv_path))
            print(f"\nProcessed {len(results_df)} volumes. Results saved to: {output_csv_path}")

        else:
            logger.error(f"Input path not found or is not a file/directory: {input_path}")

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
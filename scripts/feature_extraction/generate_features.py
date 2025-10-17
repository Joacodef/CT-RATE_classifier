# scripts/feature_extraction/generate_features.py
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import load_config
from src.data.dataset import CTMetadataDataset
from src.data.transforms import get_preprocessing_transforms
from src.training.trainer import create_model
from src.utils.logging_config import setup_logging
from monai.data import DataLoader, Dataset


def adapt_model_to_feature_extractor(model: nn.Module, model_type: str) -> nn.Module:
    """
    Adapts a trained model to act as a feature extractor by removing its
    final classification layer.

    Args:
        model (nn.Module): The model to modify.
        model_type (str): The type of the model (e.g., 'resnet3d', 'densenet3d', 'vit3d').

    Returns:
        nn.Module: The modified model.

    Raises:
        ValueError: If the model type is not supported.
    """
    if model_type in ["resnet3d", "densenet3d"]:
        # For ResNet and DenseNet, the classifier is typically named 'fc' or 'classifier'
        if hasattr(model, 'fc'):
            in_features = model.fc[-1].in_features
            model.fc = nn.Identity()
            logging.info(f"Replaced model.fc with Identity. Feature dim: {in_features}")
        elif hasattr(model, 'classifier'):
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Identity()
            logging.info(f"Replaced model.classifier with Identity. Feature dim: {in_features}")
        else:
            raise ValueError("Could not find 'fc' or 'classifier' layer in the model.")
    elif model_type == "vit3d":
        # For Vision Transformer, the classifier is named 'head'
        in_features = model.head[-1].in_features
        model.head = nn.Identity()
        logging.info(f"Replaced model.head with Identity. Feature dim: {in_features}")
    else:
        raise ValueError(f"Unsupported model type for feature extraction: {model_type}")
    
    return model


@torch.no_grad()
def generate_features(config, model_checkpoint: str, output_dir: Path, split: str):
    """
    Generates and saves feature vectors for a given dataset split.

    Args:
        config: The project configuration object.
        model_checkpoint (str): Path to the pre-trained model checkpoint.
        output_dir (Path): Directory to save the feature vectors.
        split (str): The data split to process ('train', 'valid', or 'all').
    """
    # 1. Setup device and logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")

    # 2. Load and adapt the model
    logger.info(f"Loading base model from checkpoint: {model_checkpoint}")
    model = create_model(config)
    
    checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = adapt_model_to_feature_extractor(model, config.model.type).to(device)
    model.eval()

    # 3. Load the data
    logger.info(f"Loading data for '{split}' split...")
    if split == 'all':
        df_volumes = pd.read_csv(Path(config.paths.data_dir) / config.paths.full_dataset_csv)
    elif split == 'train':
        df_volumes = pd.read_csv(config.paths.data_subsets.train)
    elif split == 'valid':
        df_volumes = pd.read_csv(config.paths.data_subsets.valid)
    else:
        raise ValueError(f"Invalid split specified: {split}. Choose 'train', 'valid', or 'all'.")

    logger.info(f"Found {len(df_volumes)} volumes to process for the '{split}' split.")

    # 4. Create the data pipeline
    preprocess_transforms = get_preprocessing_transforms(config)
    base_dataset = CTMetadataDataset(
        dataframe=df_volumes,
        img_dir=config.paths.img_dir,
        path_mode=config.paths.dir_structure,
    )
    processed_dataset = Dataset(data=base_dataset, transform=preprocess_transforms)
    data_loader = DataLoader(
        processed_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
    )

    # 5. Feature extraction loop
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving features to: {split_output_dir}")

    batch_size = config.training.batch_size
    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Generating features for '{split}' split")):
        images = batch["image"].to(device)
        
        # Retrieve volume names from the original dataframe using the batch index,
        # as the unshuffled DataLoader guarantees sequential order.
        start_idx = batch_idx * batch_size
        end_idx = start_idx + len(images)
        volume_names = df_volumes.iloc[start_idx:end_idx]['VolumeName'].tolist()

        features = model(images)
        features = features.cpu()

        for i, volume_name in enumerate(volume_names):
            feature_vector = features[i]
            # Ensure the volume name is clean for the filename
            clean_volume_name = volume_name.replace(".nii.gz", "").replace(".nii", "")
            output_path = split_output_dir / f"{clean_volume_name}.pt"
            torch.save(feature_vector, output_path)

    logger.info("Feature generation complete.")


def main():
    """Main function to run the feature generation script."""
    parser = argparse.ArgumentParser(
        description="Generate feature vectors from a pre-trained 3D model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base YAML configuration file.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to the pre-trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the generated feature vectors will be saved.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "valid", "all"],
        help="The data split to process.",
    )
    args = parser.parse_args()

    # Setup logging
    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=log_dir / f"feature_generation_{args.split}.log")

    config = load_config(args.config)
    
    # Override batch size for efficiency if needed
    config.training.batch_size = config.training.batch_size * 2
    logging.info(f"Using batch size of {config.training.batch_size} for feature extraction.")

    generate_features(
        config,
        args.model_checkpoint,
        Path(args.output_dir),
        args.split,
    )


if __name__ == "__main__":
    main()
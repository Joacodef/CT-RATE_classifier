# config/config.py

from pathlib import Path
import numpy as np
from dotenv import load_dotenv
import os

dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

class Config:
    """
    Configuration class for the CT 3D Classifier project.
    Manages all settings, paths, and hyperparameters for the application.
    """
    TRAIN_IMG_DIR = Path(os.getenv("CT_FULL_TRAIN_DIR"))
    VALID_IMG_DIR = Path(os.getenv("CT_FULL_VALID_DIR"))
    BASE_PROJECT_DIR = Path(os.getenv("BASE_PROJECT_DIR"))
    SELECTED_TRAIN_VOLUMES_CSV = BASE_PROJECT_DIR / "generated_subsets" / "selected_train_volumes.csv"
    SELECTED_VALID_VOLUMES_CSV = BASE_PROJECT_DIR / "generated_subsets" / "selected_valid_volumes.csv"
    TRAIN_LABELS_CSV = BASE_PROJECT_DIR / "dataset" / "multi_abnormality_labels" / "train_predicted_labels.csv"
    VALID_LABELS_CSV = BASE_PROJECT_DIR / "dataset" / "multi_abnormality_labels" / "valid_predicted_labels.csv"

    # Model configuration
    MODEL_TYPE = "vit3d"  # Options: "resnet3d", "densenet3d", "vit3d"
    MODEL_VARIANT = "base"      # Options depend on MODEL_TYPE:
                                # - resnet3d: "18", "34"
                                # - densenet3d: "121", "169", "201", "161", "small", "tiny"
                                # - vit3d: "tiny", "small", "base", "large"

    # Vision Transformer specific settings (only used if MODEL_TYPE == "vit3d")
    VIT_PATCH_SIZE = (16, 16, 16)  # Patch size for ViT models
    VIT_EMBED_DIM = 384  # Embedding dimension (overrides default for variant)
    VIT_DEPTH = 12  # Number of transformer blocks
    VIT_NUM_HEADS = 6  # Number of attention heads

    # Loss Function Configuration
    LOSS_FUNCTION = "FocalLoss"  # Options: "BCEWithLogitsLoss", "FocalLoss"
    FOCAL_LOSS_ALPHA = 1.0 # Alpha parameter for FocalLoss
    FOCAL_LOSS_GAMMA = 2.0 # Gamma parameter for FocalLoss

    # Training parameters
    NUM_EPOCHS = 30
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    NUM_WORKERS = 8
    PIN_MEMORY = True

    # Image processing
    TARGET_SPACING = np.array([1.0, 1.0, 1.0])  # mm
    TARGET_SHAPE_DHW = (96, 224, 224)  # Reduced depth for memory efficiency
    CLIP_HU_MIN = -1000
    CLIP_HU_MAX = 1000
    ORIENTATION_AXCODES = "LPS" # Added: Target orientation for MONAI (e.g., "LPS", "RAS")

    # Training optimization
    GRADIENT_CHECKPOINTING = True
    MIXED_PRECISION = True
    USE_BF16 = True
    EARLY_STOPPING_PATIENCE = 5

    # Cache configuration
    USE_CACHE = True
    CACHE_DIR = Path(os.getenv("CACHE_DIR"))

    # Pathology labels
    PATHOLOGY_COLUMNS = [
        "Medical material", "Arterial wall calcification", "Cardiomegaly",
        "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
        "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
        "Bronchiectasis", "Interlobular septal thickening"
    ]
    NUM_PATHOLOGIES = len(PATHOLOGY_COLUMNS)
    OUTPUT_DIR = BASE_PROJECT_DIR / "ct_3d_classifier_output"

    RESUME_FROM_CHECKPOINT = None  # Path to checkpoint file or None to start fresh
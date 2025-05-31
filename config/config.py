from pathlib import Path
import numpy as np

class Config:
    # Data paths
    TRAIN_IMG_DIR = Path(r"E:\CT-RATE_data_volumes\dataset\train_fixed")
    VALID_IMG_DIR = Path(r"E:\CT-RATE_data_volumes\dataset\valid_fixed")
    BASE_PROJECT_DIR = Path(r"E:\ProyectoRN")
    SELECTED_TRAIN_VOLUMES_CSV = BASE_PROJECT_DIR / "generated_subsets" / "selected_train_volumes.csv"
    SELECTED_VALID_VOLUMES_CSV = BASE_PROJECT_DIR / "generated_subsets" / "selected_valid_volumes.csv"
    TRAIN_LABELS_CSV = BASE_PROJECT_DIR / "dataset" / "multi_abnormality_labels" / "train_predicted_labels.csv"
    VALID_LABELS_CSV = BASE_PROJECT_DIR / "dataset" / "multi_abnormality_labels" / "valid_predicted_labels.csv"
    
    # Model configuration
    MODEL_TYPE = "resnet3d"  # Options: "resnet3d", "densenet3d", "custom3d"
    
    # Training parameters
    NUM_EPOCHS = 30
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    NUM_WORKERS = 0
    PIN_MEMORY = True
    
    # Image processing
    TARGET_SPACING = np.array([1.0, 1.0, 1.0])  # mm
    TARGET_SHAPE_DHW = (96, 224, 224)  # Reduced depth for memory efficiency
    CLIP_HU_MIN = -1000
    CLIP_HU_MAX = 1000
    
    # Training optimization
    GRADIENT_CHECKPOINTING = True
    MIXED_PRECISION = True
    USE_BF16 = True
    EARLY_STOPPING_PATIENCE = 5
    
    # Cache configuration
    USE_CACHE = True
    CACHE_DIR = Path("D:/preprocessed_cache_3d")
    
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
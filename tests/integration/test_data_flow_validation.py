# tests/integration/test_data_flow_validation.py
"""
Data Flow and Tensor Validation Test

This test focuses on inspecting a single batch of data as it emerges from the
full preprocessing pipeline. Its purpose is to verify that the tensors have
the correct shape, data type, and value ranges, and are free from corruption
(e.g., NaN values), ensuring they are ready for model consumption.
"""

import sys
from pathlib import Path
import pytest
import yaml
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from types import SimpleNamespace

# Add the project root to the Python path to allow imports from `src`
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import load_config
from src.training.trainer import create_model
from src.data.dataset import CTMetadataDataset, LabelAttacherDataset
from src.data.transforms import get_preprocessing_transforms
from monai.data import Dataset, DataLoader

# --- Fixtures ---

@pytest.fixture(scope="function")
def setup_test_environment(tmp_path):
    """Sets up a temporary, self-contained environment for a full test run."""
    data_dir = tmp_path / "data"
    img_dir = data_dir / "images"
    splits_dir = data_dir / "splits"
    labels_dir = data_dir / "labels"
    for d in [img_dir, splits_dir, labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pathologies = ["Cardiomegaly", "Atelectasis"]
    train_vols = [f"train_vol_{i}.nii.gz" for i in range(2)]
    
    # Create a dummy NIfTI with a distinct value range for testing normalization
    dummy_nifti_data = np.random.rand(64, 64, 64).astype(np.float32) * 2000 - 1000
    dummy_nifti_img = nib.Nifti1Image(dummy_nifti_data, np.eye(4))
    for vol_name in train_vols:
        # Replicates the nested path: ".../train_vol/train_vol_0/"
        name_parts = Path(vol_name).stem.split('.')[0].split('_')
        top_level_dir = "_".join(name_parts[:-1]) # Creates "train_vol"
        bottom_level_dir = "_".join(name_parts)   # Creates "train_vol_0"
        
        volume_dir = img_dir / top_level_dir / bottom_level_dir
        volume_dir.mkdir(parents=True, exist_ok=True)
        # Save the file inside its nested subdirectory
        nib.save(dummy_nifti_img, volume_dir / vol_name)

    pd.DataFrame({"VolumeName": train_vols}).to_csv(splits_dir / "train_fold_0.csv", index=False)
    labels_df = pd.DataFrame({
        "VolumeName": train_vols,
        "Cardiomegaly": [1, 0], "Atelectasis": [0, 1]
    })
    labels_df.to_csv(labels_dir / "all_predicted_labels.csv", index=False)

    yield {
        "root_dir": tmp_path, "data_dir": data_dir, "img_dir": img_dir,
        "pathologies": pathologies
    }

@pytest.fixture(scope="function")
def generate_test_config(setup_test_environment, monkeypatch):
    """Generates a minimal configuration for the data validation test."""
    env = setup_test_environment
    root_dir = env["root_dir"]
    
    monkeypatch.setenv("BASE_PROJECT_DIR", str(root_dir))
    monkeypatch.setenv("DATA_DIR", str(env["data_dir"]))

    config_data = {
        'paths': {
            'img_dir': str(env["img_dir"]),
            'base_project_dir': '${BASE_PROJECT_DIR}',
            'data_dir': '${DATA_DIR}',
            'dir_structure': 'nested',
            'data_subsets': {'train': 'splits/train_fold_0.csv'},
            'labels': {'all': 'labels/all_predicted_labels.csv'},
            'output_dir': str(root_dir / "output")
        },
        'torch_dtype': 'float32',
        'model': {'type': 'resnet3d', 'variant': '18', 'vit_specific': {}},
        'optimization': {'gradient_checkpointing': False},
        'image_processing': {
            'target_spacing': [1.0, 1.0, 1.0], 'target_shape_dhw': [64, 64, 64],
            'clip_hu_min': -1000, 'clip_hu_max': 1000, 'orientation_axcodes': 'RAS',
            'scale_b_min': 0.0, 'scale_b_max': 1.0
        },
        'pathologies': {'columns': env["pathologies"]}
    }
    config_path = root_dir / "test_config_data_val.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return load_config(config_path)

# --- Test Class ---

class TestDataFlowValidation:
    """Contains the test case for validating a single batch."""
    def test_batch_properties(self, generate_test_config):
        """
        Constructs the data pipeline, fetches one batch, and validates its properties.
        """
        # 1. Setup: Get config and load the initial dataframe
        config = generate_test_config
        train_df = pd.read_csv(config.paths.data_dir / config.paths.data_subsets.train)
        labels_df = pd.read_csv(config.paths.data_dir / config.paths.labels.all)
        train_df = pd.merge(train_df, labels_df, on='VolumeName', how='inner')
        
        # 2. Manually construct the data pipeline
        preprocess_transforms = get_preprocessing_transforms(config)
        
        base_ds = CTMetadataDataset(
            dataframe=train_df,
            img_dir=config.paths.img_dir,
            path_mode=config.paths.dir_structure
        )
        
        # This setup is for a non-cached pipeline
        processed_ds = Dataset(data=base_ds, transform=preprocess_transforms)
        final_ds = LabelAttacherDataset(
            image_source=processed_ds,
            labels_df=train_df,
            pathology_columns=config.pathologies.columns
        )
        
        # 3. Create a DataLoader and fetch one batch
        data_loader = DataLoader(final_ds, batch_size=2, shuffle=False)
        batch = next(iter(data_loader))

        # 4. Perform detailed assertions on the batch tensors
        # -- Image Tensor Validation --
        assert 'image' in batch
        image_tensor = batch['image']
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (2, 1, 64, 64, 64)
        assert image_tensor.dtype == torch.float32
        
        # Check that ScaleIntensityRanged worked correctly
        assert image_tensor.min() >= 0.0
        assert image_tensor.max() <= 1.0
        
        # Check for data corruption
        assert not torch.isnan(image_tensor).any()
        assert not torch.isinf(image_tensor).any()

        # -- Label Tensor Validation --
        assert 'label' in batch
        label_tensor = batch['label']
        assert isinstance(label_tensor, torch.Tensor)
        assert label_tensor.shape == (2, len(config.pathologies.columns))
        assert label_tensor.dtype == torch.float32

        # 5. Validate model compatibility
        model = create_model(config)
        model.eval() # Set to evaluation mode for the test pass
        with torch.no_grad():
            output = model(image_tensor)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, len(config.pathologies.columns))
# tests/integration/feature_extraction/test_generate_features.py
import sys
import yaml
from pathlib import Path
import pytest
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from unittest.mock import patch

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from scripts.feature_extraction.generate_features import main as generate_features_main
from src.models.resnet3d import resnet18_3d

# --- Fixtures ---

@pytest.fixture(scope="function")
def setup_feature_extraction_environment(tmp_path: Path, monkeypatch):
    """
    Sets up a temporary, self-contained environment for the feature extraction
    integration test.
    """
    # 1. Define and create directory structure
    data_dir = tmp_path / "data"
    img_dir = data_dir / "images"
    output_dir = tmp_path / "output_features"
    model_dir = tmp_path / "models"
    
    for d in [img_dir, output_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Set environment variables for config loading
    monkeypatch.setenv("BASE_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    # 2. Create dummy NIfTI files
    volume_names = [f"vol_{i}.nii.gz" for i in range(2)]
    dummy_nifti_data = np.zeros((16, 16, 16), dtype=np.float32)
    dummy_nifti_img = nib.Nifti1Image(dummy_nifti_data, np.eye(4))
    for vol_name in volume_names:
        # The 'flat' dir_structure mode expects a "__transformed" suffix
        file_stem = vol_name.replace(".nii.gz", "")
        transformed_name = f"{file_stem}__transformed.nii.gz"
        nib.save(dummy_nifti_img, img_dir / transformed_name)

    # 3. Create mock model checkpoint
    model = resnet18_3d(num_classes=18)
    model_checkpoint_path = model_dir / "mock_model.pth"
    torch.save(model.state_dict(), model_checkpoint_path)

    # 4. Create dummy dataset CSV
    full_dataset_csv_path = data_dir / "full_dataset.csv"
    pd.DataFrame({"VolumeName": volume_names}).to_csv(full_dataset_csv_path, index=False)

    # 5. Create minimal YAML config
    config_data = {
        'paths': {
            'base_project_dir': '${BASE_PROJECT_DIR}',
            'data_dir': '${DATA_DIR}',
            'img_dir': str(img_dir),
            'dir_structure': 'flat',
            'full_dataset_csv': 'full_dataset.csv',
        },
        'model': {'type': 'resnet3d', 'variant': '18'},
        'optimization': {
            'gradient_checkpointing': False
        },
        'image_processing': {
            'target_spacing': [1.0, 1.0, 1.0],
            'target_shape_dhw': [16, 16, 16],
            'clip_hu_min': -1000, 'clip_hu_max': 1000,
            'orientation_axcodes': 'RAS'
        },
        'training': {'batch_size': 1, 'num_workers': 0},
        'pathologies': {'columns': [f"Pathology_{i}" for i in range(18)]}, # Match the num
        'torch_dtype': 'float32',
        
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    return {
        "config_path": str(config_path),
        "model_checkpoint_path": str(model_checkpoint_path),
        "output_dir": str(output_dir),
        "num_volumes": len(volume_names),
        "expected_feature_dim": 512  # ResNet-18 feature dimension
    }

# --- Test Class ---

class TestGenerateFeaturesScript:
    """
    Integration test for the generate_features.py script.
    """

    @patch('sys.argv', new_callable=list)
    def test_script_runs_and_creates_features(self, mock_argv, setup_feature_extraction_environment):
        """
        Tests that the script runs end-to-end, creating the correct feature
        files in the specified output directory.
        """
        # 1. Arrange: Get paths from the test environment fixture
        env = setup_feature_extraction_environment
        
        # Mock the command-line arguments that the script expects
        mock_argv.extend([
            "generate_features.py",
            "--config", env["config_path"],
            "--model-checkpoint", env["model_checkpoint_path"],
            "--output-dir", env["output_dir"],
            "--split", "all"
        ])

        # 2. Act: Run the main function of the script
        generate_features_main()

        # 3. Assert: Verify the outputs
        output_path = Path(env["output_dir"]) / "all"
        assert output_path.exists(), "The output directory for the split was not created."

        # Check that the correct number of feature files were created
        feature_files = list(output_path.glob("*.pt"))
        assert len(feature_files) == env["num_volumes"]

        # Load one of the feature files to inspect its content
        loaded_feature = torch.load(feature_files[0], weights_only=False)
        
        assert isinstance(loaded_feature, torch.Tensor)
        assert loaded_feature.ndim == 1, "Feature should be a 1D vector."
        assert loaded_feature.shape[0] == env["expected_feature_dim"]
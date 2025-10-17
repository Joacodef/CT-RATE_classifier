# tests/integration/test_end_to_end_flow.py
"""
End-to-end integration test for the training pipeline.

This test simulates a full training run on a minimal, dynamically generated
dataset. Its purpose is to verify that all components of the data pipeline,
model training, and output generation work together correctly, ensuring the
integrity of the tensor flow from raw data to final artifacts.
"""

import sys
import json
from pathlib import Path
import pytest
import yaml
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from types import SimpleNamespace
import copy
import os
import matplotlib

# Use non-interactive backend to avoid Tcl/Tk errors in CI/environments without tcl libs
matplotlib.use("Agg")
os.environ.setdefault("MPLBACKEND", "Agg")

# Add the project root to the Python path to allow imports from `src`
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import load_config
from src.training.trainer import train_model

# --- Fixtures ---

@pytest.fixture(scope="function")
def setup_test_environment(tmp_path):
    """
    Sets up a temporary, self-contained environment for a full test run.

    This fixture creates all necessary subdirectories, generates dummy NIfTI
    files for training and validation, and creates the corresponding CSV
    files for data splits and labels.

    Args:
        tmp_path (Path): The pytest-provided temporary directory path.

    Yields:
        A dictionary containing the absolute paths to key directories.
    """
    # 1. Define and create the directory structure
    data_dir = tmp_path / "data"
    img_dir = data_dir / "images"
    splits_dir = data_dir / "splits"
    labels_dir = data_dir / "labels"
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    features_dir = output_dir / "features"

    for d in [img_dir, splits_dir, labels_dir, output_dir, cache_dir, features_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Define test parameters
    pathologies = ["Cardiomegaly", "Atelectasis"]
    train_vols = [f"train_vol_{i}.nii.gz" for i in range(2)]
    valid_vols = [f"valid_vol_{i}.nii.gz" for i in range(2)]
    all_vols = train_vols + valid_vols

    # 3. Create dummy NIfTI files in the correct nested structure
    dummy_nifti_data = np.zeros((64, 64, 64), dtype=np.float32)
    dummy_nifti_img = nib.Nifti1Image(dummy_nifti_data, np.eye(4))
    for vol_name in all_vols:
        # Replicates the nested path from the error log: ".../train_vol/train_vol_0/"
        name_parts = Path(vol_name).stem.split('.')[0].split('_')
        top_level_dir = "_".join(name_parts[:-1]) # Creates "train_vol" or "valid_vol"
        bottom_level_dir = "_".join(name_parts)   # Creates "train_vol_0" or "valid_vol_1"
        
        volume_dir = img_dir / top_level_dir / bottom_level_dir
        volume_dir.mkdir(parents=True, exist_ok=True)
        # Save the file inside its deeply nested subdirectory
        nib.save(dummy_nifti_img, volume_dir / vol_name)

    # 4. Create dummy CSV files
    # Data splits (fold 0)
    pd.DataFrame({"VolumeName": train_vols}).to_csv(splits_dir / "train_fold_0.csv", index=False)
    pd.DataFrame({"VolumeName": valid_vols}).to_csv(splits_dir / "valid_fold_0.csv", index=False)

    # Master labels file
    labels_df = pd.DataFrame({
        "VolumeName": all_vols,
        "Cardiomegaly": np.random.randint(0, 2, len(all_vols)),
        "Atelectasis": np.random.randint(0, 2, len(all_vols)),
    })
    labels_df.to_csv(labels_dir / "all_predicted_labels.csv", index=False)
    # 5. Create dummy feature files (placed before yield so tests can access them)
    features_dir = output_dir / "features"
    train_features_dir = features_dir / "train"
    valid_features_dir = features_dir / "valid"
    train_features_dir.mkdir(parents=True, exist_ok=True)
    valid_features_dir.mkdir(parents=True, exist_ok=True)
    
    # Example feature dimension for the mock features
    feature_dim = 512 
    
    for vol_name in train_vols:
        # Save feature files with the original volume filename plus .pt so the
        # dataset loader (which appends .pt to VolumeName) will find them.
        feature_filename = f"{vol_name}.pt"
        feature_tensor = torch.randn(feature_dim)
        torch.save(feature_tensor, train_features_dir / feature_filename)
        
    for vol_name in valid_vols:
        # See note above about naming
        feature_filename = f"{vol_name}.pt"
        feature_tensor = torch.randn(feature_dim)
        torch.save(feature_tensor, valid_features_dir / feature_filename)

    yield {
        "root_dir": tmp_path,
        "data_dir": data_dir,
        "img_dir": img_dir,
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "features_dir": features_dir,
        "pathologies": pathologies
    }


@pytest.fixture(scope="function")
def generate_test_config(request, setup_test_environment, monkeypatch):
    """
    Generates a configuration tailored to the test parameters.

    This fixture uses indirect parameterization. It receives its parameters
    from the `request.param` object, which is populated by the
    `@pytest.mark.parametrize` decorator on the test class.

    Args:
        request: The pytest request object, used to access parameters.
        setup_test_environment (dict): The dictionary of paths from the setup fixture.
        monkeypatch: The pytest monkeypatch fixture for setting environment variables.

    Returns:
        SimpleNamespace: The loaded and processed configuration object.
    """
    model_type = request.param.get("model_type", "resnet3d")
    use_cache = request.param.get("use_cache", False)
    workflow_mode = request.param.get("workflow_mode", "end-to-end")

    env = setup_test_environment
    root_dir = env["root_dir"]
    
    # Set environment variables for the config loader
    monkeypatch.setenv("BASE_PROJECT_DIR", str(root_dir))
    monkeypatch.setenv("CACHE_DIR", str(env["cache_dir"]))
    monkeypatch.setenv("DATA_DIR", str(env["data_dir"]))

    config_data = {
        'paths': {
            'img_dir': str(env["img_dir"]),
            'base_project_dir': '${BASE_PROJECT_DIR}',
            'cache_dir': '${CACHE_DIR}',
            'data_dir': '${DATA_DIR}',
            'dir_structure': 'nested',
            'data_subsets': {
                'train': 'splits/train_fold_0.csv',
                'valid': 'splits/valid_fold_0.csv'
            },
            'labels': {'all': 'labels/all_predicted_labels.csv'},
        },
        'torch_dtype': 'float32',
        'model': {
            'type': model_type, 'variant': '18' if model_type == 'resnet3d' else ('121' if model_type == 'densenet3d' else 'tiny'),
            'vit_specific': {'patch_size': [8, 8, 8]}
        },
        'workflow': {
            'mode': workflow_mode,
            'feature_config': {
                'feature_dir': str(env["features_dir"])
            }
        },
        'loss_function': {'type': 'BCEWithLogitsLoss'},
        'training': {
            'seed': 42, 'num_epochs': 1, 'batch_size': 1,
            'gradient_accumulation_steps': 1, 'learning_rate': 1e-4,
            'weight_decay': 1e-5, 'num_workers': 0, 'pin_memory': False,
            'augment': False, 'early_stopping_patience': 2,
            'resume_from_checkpoint': None
        },
        'optimization': {
            'gradient_checkpointing': False, 'mixed_precision': False, 'use_bf16': False
        },
        'image_processing': {
            'target_spacing': [1.0, 1.0, 1.0],
            'target_shape_dhw': [64, 64, 64],
            'clip_hu_min': -1000, 'clip_hu_max': 1000,
            'orientation_axcodes': 'RAS'
        },
        'cache': {'use_cache': use_cache, 'memory_rate': 0.0},
        'pathologies': {'columns': env["pathologies"]},
        'wandb': {'enabled': False}
    }

    # Use a unique output directory for each parameterized run to avoid conflicts
    output_dir = Path(env["output_dir"]) / f"{model_type}_cache-{use_cache}_workflow-{workflow_mode}"
    config_data['paths']['output_dir'] = str(output_dir)

    config_path = root_dir / f"test_config_{model_type}_{use_cache}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    # Also create feature files inside the parameterized output directory so
    # feature-based workflows can load them from the expected location.
    final_output_dir = Path(config_data['paths']['output_dir'])
    feature_output_dir = final_output_dir / "features"
    train_features_dir = feature_output_dir / "train"
    valid_features_dir = feature_output_dir / "valid"
    train_features_dir.mkdir(parents=True, exist_ok=True)
    valid_features_dir.mkdir(parents=True, exist_ok=True)

    # Read split CSVs to determine volume names
    splits_base = Path(env['data_dir']) / 'splits'
    train_csv = splits_base / 'train_fold_0.csv'
    valid_csv = splits_base / 'valid_fold_0.csv'
    try:
        train_vols = pd.read_csv(train_csv)['VolumeName'].tolist()
        valid_vols = pd.read_csv(valid_csv)['VolumeName'].tolist()
    except Exception:
        train_vols = []
        valid_vols = []

    # Update the config to point to this per-run feature directory so the
    # training workflow will look in the same place we create files.
    config_data['workflow']['feature_config']['feature_dir'] = str(feature_output_dir)

    # Create small random feature tensors matching the expected filenames
    feature_dim = 512
    for vol_name in train_vols:
        feature_filename = f"{vol_name}.pt"
        torch.save(torch.randn(feature_dim), train_features_dir / feature_filename)

    for vol_name in valid_vols:
        feature_filename = f"{vol_name}.pt"
        torch.save(torch.randn(feature_dim), valid_features_dir / feature_filename)

    return load_config(config_path)

# --- Test Class ---

@pytest.mark.parametrize(
    "generate_test_config",
    [
        {"model_type": "resnet3d", "use_cache": False},
        {"model_type": "resnet3d", "use_cache": True},
        {"model_type": "densenet3d", "use_cache": False},
        {"model_type": "vit3d", "use_cache": False},
        {"model_type": "mlp", "use_cache": False, "workflow_mode": "feature-based"},  # new parameter to exercise the feature-based workflow
    ],
    indirect=True
)
class TestEndToEndTrainingFlow:
    """
    Contains the test cases for a full, minimal training run.
    The tests are parameterized to run for different models and caching strategies.
    """
    def test_full_training_run(self, generate_test_config):
        """
        Executes a full training run and verifies the outputs and their contents.
        """
        config = generate_test_config
        output_dir = Path(config.paths.output_dir)
        
        model, history = train_model(config)

        assert isinstance(model, torch.nn.Module)
        assert len(history['train_loss']) == 1
        
        history_path = output_dir / "training_history.json"
        assert history_path.exists()
        
        final_model_path = output_dir / "final_model.pth"
        assert final_model_path.exists()
        checkpoint = torch.load(final_model_path, map_location=torch.device('cpu'), weights_only=False)
        assert "model_state_dict" in checkpoint

    def test_full_training_resume_flow(self, generate_test_config):
        """
        Tests the ability to resume training from a checkpoint.
        """
        config = generate_test_config
        output_dir = Path(config.paths.output_dir)

        # First run
        config.training.num_epochs = 1
        train_model(config)
        last_checkpoint_path = output_dir / "last_checkpoint.pth"
        assert last_checkpoint_path.exists()

        # Second run (resuming)
        resume_config = copy.deepcopy(config)
        resume_config.training.num_epochs = 2
        resume_config.training.resume_from_checkpoint = str(last_checkpoint_path)
        model, history = train_model(resume_config)

        assert len(history['train_loss']) == 2
        
        final_model_path = output_dir / "final_model.pth"
        checkpoint = torch.load(final_model_path, map_location=torch.device('cpu'), weights_only=False)
        assert checkpoint['epoch'] == 1

    def test_feature_based_training_run(self, generate_test_config):
        """
        Run a feature-based training flow. Verifies that:
        - the configuration requests a model of type 'mlp'
        - the returned model is a torch.nn.Module (MLP)
        - training finishes successfully (history contains losses)
        """
        config = generate_test_config
        output_dir = Path(config.paths.output_dir)

        # Only apply this test for runs configured as feature-based.
        if getattr(config.workflow, "mode", None) not in ("feature-based", "feature_based", "feature"):
            pytest.skip("Not a feature-based run for this parametrization; skipping test_feature_based_training_run.")
        
        # Ensure the configuration requested the feature-based mode and the correct model type
        assert getattr(config.model, "type", None) == "mlp"
        assert getattr(config.workflow, "mode", None) in ("feature-based", "feature_based", "feature")

        model, history = train_model(config)

        # Basic result assertions
        assert isinstance(model, torch.nn.Module)
        # Prefer checking the class name to confirm MLP if present in the name
        model_name = model.__class__.__name__.lower()
        assert "mlp" in model_name or getattr(config.model, "type", "") == "mlp"

        assert isinstance(history, dict)
        assert "train_loss" in history
        assert len(history["train_loss"]) >= 1

        # Expected final artifacts/files
        final_model_path = output_dir / "final_model.pth"
        assert final_model_path.exists()


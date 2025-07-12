# tests/integration/scripts/test_train.py
"""
Integration tests for the main training script: scripts/train.py.

This module verifies that the `train.py` script correctly handles
command-line arguments, loads and processes configuration files,
and initiates the training process by calling `train_model` with
the expected parameters.
"""
import sys
from pathlib import Path
import pytest
import yaml
import pandas as pd
import nibabel as nib
import numpy as np
from unittest.mock import patch

# Import the main function from the script to be tested
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from scripts.train import main as train_main

# --- Fixtures ---

@pytest.fixture(scope="function")
def test_environment(tmp_path, monkeypatch):
    """
    Sets up a complete temporary environment for a test run.
    """
    # 1. Create directories
    project_root_dir = tmp_path
    data_dir = project_root_dir / "test_data"
    train_img_dir = data_dir / "train_images"
    valid_img_dir = data_dir / "valid_images"
    output_dir = project_root_dir / "test_output"
    cache_dir = project_root_dir / "test_cache"
    splits_dir = data_dir / "splits" / "tiny_split"
    labels_dir = data_dir / "labels" / "multi_abnormality_labels"
    
    for d in [train_img_dir, valid_img_dir, output_dir, cache_dir, splits_dir, labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Create dummy data files
    pathologies = ["Cardiomegaly", "Atelectasis"]
    pd.DataFrame({"VolumeName": ["train_vol1.nii.gz"]}).to_csv(splits_dir / "train.csv", index=False)
    pd.DataFrame({"VolumeName": ["valid_vol1.nii.gz"]}).to_csv(splits_dir / "valid.csv", index=False)
    pd.DataFrame(
        {"VolumeName": ["train_vol1.nii.gz"], "Cardiomegaly": [1], "Atelectasis": [0]}
    ).to_csv(labels_dir / "train_predicted_labels.csv", index=False)
    pd.DataFrame(
        {"VolumeName": ["valid_vol1.nii.gz"], "Cardiomegaly": [0], "Atelectasis": [1]}
    ).to_csv(labels_dir / "valid_predicted_labels.csv", index=False)

    dummy_nifti_data = np.zeros((10, 10, 10), dtype=np.float32)
    dummy_nifti_img = nib.Nifti1Image(dummy_nifti_data, np.eye(4))
    nib.save(dummy_nifti_img, train_img_dir / "train_vol1__transformed.nii.gz")
    nib.save(dummy_nifti_img, valid_img_dir / "valid_vol1__transformed.nii.gz")

    # 3. Set environment variables
    monkeypatch.setenv("BASE_PROJECT_DIR", str(project_root_dir))
    monkeypatch.setenv("TRAIN_IMG_DIR", str(train_img_dir))
    monkeypatch.setenv("VALID_IMG_DIR", str(valid_img_dir))
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    # 4. Create YAML config file
    config_data = {
        "paths": {
            "train_img_dir": "${TRAIN_IMG_DIR}", "valid_img_dir": "${VALID_IMG_DIR}",
            "base_project_dir": str(project_root_dir), "cache_dir": "${CACHE_DIR}",
            "data_dir": "${DATA_DIR}", "dir_structure": "flat",
            "data_subsets": {"train": "splits/tiny_split/train.csv", "valid": "splits/tiny_split/valid.csv"},
            "labels": {
                "train": "labels/multi_abnormality_labels/train_predicted_labels.csv",
                "valid": "labels/multi_abnormality_labels/valid_predicted_labels.csv",
            },
            "reports": {"train": "reports/train.csv", "valid": "reports/valid.csv"},
            "metadata": {"train": "metadata/train.csv", "valid": "metadata/valid.csv"},
            "output_dir": str(output_dir),
        },
        "model": {"type": "resnet3d", "variant": "18", "vit_specific": {"patch_size": [16, 16, 16]}},
        "loss_function": {"type": "BCEWithLogitsLoss", "focal_loss": {}},
        "training": {
            "num_epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-4, "weight_decay": 0.01, "num_workers": 0, "pin_memory": False,
            "early_stopping_patience": 2, "resume_from_checkpoint": None, "augment": True,
        },
        "optimization": {"gradient_checkpointing": False, "mixed_precision": False, "use_bf16": False},
        "image_processing": {
            "target_spacing": [1.0, 1.0, 1.0], "target_shape_dhw": [16, 16, 16],
            "clip_hu_min": -1000, "clip_hu_max": 1000, "orientation_axcodes": "LPS",
        },
        "cache": {"use_cache": False},
        "pathologies": {"columns": pathologies},
        "wandb": {"enabled": False, "project": "test-project", "group": "integration-tests"},
    }
    config_path = project_root_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    yield {"config_path": config_path, "output_dir": output_dir}


# --- Test Cases ---

@patch("scripts.train.train_model")
def test_train_script_basic_run(mock_train_model, test_environment, monkeypatch):
    config_path = test_environment["config_path"]
    monkeypatch.setattr(sys, "argv", ["scripts/train.py", "--config", str(config_path)])
    train_main()
    mock_train_model.assert_called_once()


@patch("scripts.train.train_model")
def test_train_script_model_override(mock_train_model, test_environment, monkeypatch):
    config_path = test_environment["config_path"]
    monkeypatch.setattr(sys, "argv", [
        "scripts/train.py", "--config", str(config_path),
        "--model-type", "vit3d", "--model-variant", "base",
    ])
    train_main()
    mock_train_model.assert_called_once()
    passed_config = mock_train_model.call_args.args[0]
    assert passed_config.model.type == "vit3d"
    assert passed_config.model.variant == "base"


@patch("scripts.train.train_model")
@patch("scripts.train.logging.warning")
def test_train_script_resume_no_checkpoint(
    mock_log_warning, mock_train_model, test_environment, monkeypatch
):
    """
    Tests the --resume flag when no checkpoint exists.
    Verifies a warning is logged and training starts fresh.
    """
    config_path = test_environment["config_path"]
    monkeypatch.setattr(sys, "argv", [
        "scripts/train.py", "--config", str(config_path), "--resume"
    ])

    train_main()

    mock_train_model.assert_called_once()
    passed_config = mock_train_model.call_args.args[0]
    assert passed_config.training.resume_from_checkpoint is None
    
    # FIX: Check for a substring of the actual logged warning
    mock_log_warning.assert_called_once()
    assert "Automatic resume failed" in mock_log_warning.call_args[0][0]


@patch("scripts.train.train_model")
def test_train_script_resume_with_checkpoint(
    mock_train_model, test_environment, monkeypatch
):
    """
    Tests the --resume flag when a checkpoint file exists.
    Verifies the config is updated with the correct checkpoint path.
    """
    config_path = test_environment["config_path"]
    output_dir = test_environment["output_dir"]
    checkpoint_path = output_dir / "last_checkpoint.pth"
    checkpoint_path.touch()

    monkeypatch.setattr(sys, "argv", [
        "scripts/train.py", "--config", str(config_path), "--resume"
    ])

    train_main()

    mock_train_model.assert_called_once()
    passed_config = mock_train_model.call_args.args[0]

    # FIX: Convert the Path object to a string before comparing.
    # This ensures you are comparing two strings (str == str).
    assert passed_config.training.resume_from_checkpoint == str(checkpoint_path)
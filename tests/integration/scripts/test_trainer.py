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

    This fixture creates a temporary directory structure, mock data files (CSVs, NIfTI),
    a YAML configuration file, and sets necessary environment variables to simulate
    a realistic execution context for the training script.

    Args:
        tmp_path (Path): The pytest-provided temporary path.
        monkeypatch: The pytest fixture for modifying classes, methods, etc.

    Yields:
        dict: A dictionary of key paths and directories for use in tests.
    """
    # 1. Create directories
    project_root = tmp_path
    data_dir = project_root / "test_data"
    train_img_dir = data_dir / "train_images"
    valid_img_dir = data_dir / "valid_images"
    output_dir = project_root / "test_output"
    cache_dir = project_root / "test_cache"
    generated_subsets_dir = data_dir / "generated_subsets"
    labels_dir = data_dir / "dataset/multi_abnormality_labels"

    for d in [
        train_img_dir,
        valid_img_dir,
        output_dir,
        cache_dir,
        generated_subsets_dir,
        labels_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Create dummy data files
    pathologies = ["Cardiomegaly", "Atelectasis"]

    pd.DataFrame({"VolumeName": ["train_vol1.nii.gz"]}).to_csv(
        generated_subsets_dir / "selected_train_volumes.csv", index=False
    )
    pd.DataFrame({"VolumeName": ["valid_vol1.nii.gz"]}).to_csv(
        generated_subsets_dir / "selected_valid_volumes.csv", index=False
    )

    pd.DataFrame(
        {"VolumeName": ["train_vol1.nii.gz"], "Cardiomegaly": [1], "Atelectasis": [0]}
    ).to_csv(labels_dir / "train_predicted_labels.csv", index=False)
    pd.DataFrame(
        {"VolumeName": ["valid_vol1.nii.gz"], "Cardiomegaly": [0], "Atelectasis": [1]}
    ).to_csv(labels_dir / "valid_predicted_labels.csv", index=False)

    dummy_nifti_data = np.zeros((10, 10, 10), dtype=np.float32)
    dummy_nifti_img = nib.Nifti1Image(dummy_nifti_data, np.eye(4))
    nib.save(dummy_nifti_img, train_img_dir / "train_vol1.nii.gz")
    nib.save(dummy_nifti_img, valid_img_dir / "valid_vol1.nii.gz")

    # 3. Set environment variables for the config loader
    monkeypatch.setenv("BASE_PROJECT_DIR", str(project_root))
    monkeypatch.setenv("TRAIN_IMG_DIR", str(train_img_dir))
    monkeypatch.setenv("VALID_IMG_DIR", str(valid_img_dir))
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))

    # 4. Create YAML config file
    config_data = {
        "paths": {
            "train_img_dir": "${TRAIN_IMG_DIR}",
            "valid_img_dir": "${VALID_IMG_DIR}",
            "base_project_dir": str(project_root),
            "cache_dir": "${CACHE_DIR}",
            "data_subsets": {
                "selected_train_volumes": "test_data/generated_subsets/selected_train_volumes.csv",
                "selected_valid_volumes": "test_data/generated_subsets/selected_valid_volumes.csv",
            },
            "labels": {
                "train": "test_data/dataset/multi_abnormality_labels/train_predicted_labels.csv",
                "valid": "test_data/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
            },
            "output_dir": str(output_dir),
        },
        "model": {
            "type": "resnet3d",
            "variant": "18",
            "vit_specific": {"patch_size": [16, 16, 16]},
        },
        "loss_function": {"type": "BCEWithLogitsLoss", "focal_loss": {}},
        "training": {
            "num_epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.01,
            "num_workers": 0,
            "pin_memory": False,
            "early_stopping_patience": 2,
            "resume_from_checkpoint": None,
        },
        "optimization": {
            "gradient_checkpointing": False,
            "mixed_precision": False,
            "use_bf16": False,
        },
        "image_processing": {
            "target_spacing": [1.0, 1.0, 1.0],
            "target_shape_dhw": [16, 16, 16],
            "clip_hu_min": -1000,
            "clip_hu_max": 1000,
            "orientation_axcodes": "LPS",
        },
        "cache": {"use_cache": False},
        "pathologies": {"columns": pathologies},
    }

    config_path = project_root / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    yield {"config_path": config_path, "output_dir": output_dir}


# --- Test Cases ---


@patch("scripts.train.train_model")
def test_train_script_basic_run(mock_train_model, test_environment, monkeypatch):
    """
    Tests a basic invocation of the training script.

    Verifies that `train_model` is called once with a correctly parsed and
    resolved configuration object.
    """
    config_path = test_environment["config_path"]

    # Simulate command-line arguments: `python train.py --config ...`
    monkeypatch.setattr(sys, "argv", ["scripts/train.py", "--config", str(config_path)])

    train_main()

    # Assert that the core training function was called
    mock_train_model.assert_called_once()

    # Inspect the config object passed to the mock
    passed_config = mock_train_model.call_args.args[0]

    assert passed_config.model.type == "resnet3d"
    assert passed_config.model.variant == "18"
    assert passed_config.training.num_epochs == 1
    assert passed_config.training.resume_from_checkpoint is None

    # Verify that paths were resolved correctly and are absolute
    assert passed_config.paths.output_dir.is_absolute()
    assert passed_config.paths.output_dir == test_environment["output_dir"]
    assert passed_config.paths.labels.train.name == "train_predicted_labels.csv"


@patch("scripts.train.train_model")
def test_train_script_model_override(mock_train_model, test_environment, monkeypatch):
    """
    Tests overriding model configuration via command-line arguments.
    """
    config_path = test_environment["config_path"]

    # Simulate CLI arguments with model overrides
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/train.py",
            "--config",
            str(config_path),
            "--model-type",
            "vit3d",
            "--model-variant",
            "base",
        ],
    )

    train_main()

    # Assert that the config passed to train_model reflects the overrides
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

    Verifies that a warning is logged and training starts fresh without
    a checkpoint path in the configuration.
    """
    config_path = test_environment["config_path"]

    # Simulate CLI arguments with the resume flag
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/train.py", "--config", str(config_path), "--resume"],
    )

    train_main()

    # Assertions
    mock_train_model.assert_called_once()
    passed_config = mock_train_model.call_args.args[0]

    # No checkpoint should be set in the config
    assert passed_config.training.resume_from_checkpoint is None

    # A warning should have been logged
    mock_log_warning.assert_called_once()
    assert "no checkpoint was found" in mock_log_warning.call_args[0][0]


@patch("scripts.train.train_model")
def test_train_script_resume_with_checkpoint(
    mock_train_model, test_environment, monkeypatch
):
    """
    Tests the --resume flag when a checkpoint file exists.

    Verifies that the configuration is correctly updated with the path
    to the latest checkpoint.
    """
    config_path = test_environment["config_path"]
    output_dir = test_environment["output_dir"]

    # Create a dummy checkpoint file for the script to find
    checkpoint_path = output_dir / "last_checkpoint.pth"
    checkpoint_path.touch()

    # Simulate CLI arguments with the resume flag
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts/train.py", "--config", str(config_path), "--resume"],
    )

    train_main()

    # Assert that the config passed to train_model has the correct checkpoint path
    mock_train_model.assert_called_once()
    passed_config = mock_train_model.call_args.args[0]

    assert passed_config.training.resume_from_checkpoint == checkpoint_path
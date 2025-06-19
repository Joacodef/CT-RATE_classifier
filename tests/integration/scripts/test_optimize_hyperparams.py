# tests/integration/scripts/test_optimize_hyperparams.py
"""
Unit and integration tests for scripts/optimize_hyperparams.py

This module tests the hyperparameter optimization script, focusing on the
objective function's logic and the main function's setup of the Optuna study.
Extensive mocking is used to avoid running actual training or optimization.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock, ANY
from pathlib import Path
from types import SimpleNamespace
import argparse
import torch
import optuna

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from scripts.optimize_hyperparams import objective, main

# --- Fixtures ---

@pytest.fixture
def mock_base_config(tmp_path: Path) -> SimpleNamespace:
    """Provides a mock baseline configuration object."""
    output_dir = tmp_path / "test_output"
    data_subsets_dir = tmp_path / "data_subsets"
    return SimpleNamespace(
        paths=SimpleNamespace(
            output_dir=output_dir,
            data_subsets=SimpleNamespace(train=data_subsets_dir / "train.csv"),
            labels=SimpleNamespace(train=tmp_path / "labels" / "train.csv")
        ),
        training=SimpleNamespace(
            learning_rate=0.0, weight_decay=0.0, batch_size=0
        ),
        model=SimpleNamespace(type="", variant=""),
        loss_function=SimpleNamespace(
            type="FocalLoss",
            focal_loss=SimpleNamespace(alpha=0.0, gamma=0.0)
        ),
        wandb=SimpleNamespace(enabled=False)
    )

@pytest.fixture
def mock_args() -> argparse.Namespace:
    """Provides mock command-line arguments."""
    return argparse.Namespace(
        trials_on_5_percent=5,
        trials_on_20_percent=10,
        trials_on_50_percent=15,
        study_name="test_study"
    )

@pytest.fixture
def mock_trial() -> MagicMock:
    """Provides a mock Optuna Trial object."""
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 0
    # Make suggest methods return a default value
    trial.suggest_float.return_value = 0.001
    trial.suggest_categorical.return_value = "default"
    trial.params = {'learning_rate': 1e-4, 'model_type': 'resnet3d'}
    return trial


# --- Test Classes ---

class TestObjectiveFunction:
    """Tests the core logic of the Optuna objective function."""

    @patch('scripts.optimize_hyperparams.train_model')
    @patch('pathlib.Path.exists')
    def test_successful_trial_returns_metric(
        self, mock_exists, mock_train_model,
        mock_trial, mock_base_config, mock_args
    ):
        """
        Verifies that a successful trial run returns the best metric from history.
        """
        # Setup: Mock that subset files exist and train_model returns a valid history
        mock_exists.return_value = True
        mock_train_model.return_value = (
            None, # model object
            {'metrics': [{'roc_auc_macro': 0.8}, {'roc_auc_macro': 0.85}]}
        )

        # Execute
        result = objective(mock_trial, mock_base_config, mock_args)

        # Assert
        mock_train_model.assert_called_once()
        assert result == 0.85

    @patch('scripts.optimize_hyperparams.train_model')
    @patch('pathlib.Path.exists')
    def test_staged_optimization_path_selection(
        self, mock_exists, mock_train_model,
        mock_trial, mock_base_config, mock_args
    ):
        """
        Tests that the correct data subset is selected based on the trial number.
        """
        mock_exists.return_value = True
        mock_train_model.return_value = (None, {'metrics': []})

        # Trial 0 should use 5% dataset
        mock_trial.number = 0
        objective(mock_trial, mock_base_config, mock_args)
        called_config = mock_train_model.call_args[1]['config']
        assert called_config.paths.train_subset_path.name == "train_05_percent.csv"

        # Trial 6 should use 20% dataset (between 5 and 10)
        mock_trial.number = 6
        objective(mock_trial, mock_base_config, mock_args)
        called_config = mock_train_model.call_args[1]['config']
        assert called_config.paths.train_subset_path.name == "train_20_percent.csv"

        # Trial 12 should use 50% dataset (between 10 and 15)
        mock_trial.number = 12
        objective(mock_trial, mock_base_config, mock_args)
        called_config = mock_train_model.call_args[1]['config']
        assert called_config.paths.train_subset_path.name == "train_50_percent.csv"

        # Trial 16 should use full dataset (train_subset_path is None)
        mock_trial.number = 16
        objective(mock_trial, mock_base_config, mock_args)
        called_config = mock_train_model.call_args[1]['config']
        assert called_config.paths.train_subset_path is None

    @patch('scripts.optimize_hyperparams.train_model')
    @patch('pathlib.Path.exists')
    def test_cuda_oom_prunes_trial(
        self, mock_exists, mock_train_model,
        mock_trial, mock_base_config, mock_args
    ):
        """
        Verifies that a CUDA OOM error correctly raises a TrialPruned exception.
        """
        mock_exists.return_value = True  # Ensure file checks pass
        mock_train_model.side_effect = torch.cuda.OutOfMemoryError()

        with pytest.raises(optuna.exceptions.TrialPruned):
            objective(mock_trial, mock_base_config, mock_args)

    @patch('scripts.optimize_hyperparams.train_model')
    @patch('pathlib.Path.exists')
    def test_generic_exception_returns_zero(
        self, mock_exists, mock_train_model,
        mock_trial, mock_base_config, mock_args
    ):
        """
        Verifies that any other exception during training returns 0.0.
        """
        mock_exists.return_value = True  # Ensure file checks pass
        mock_train_model.side_effect = ValueError("A generic error")
        
        result = objective(mock_trial, mock_base_config, mock_args)
        assert result == 0.0

class TestMainFunction:
    """Tests the main script execution flow."""

    @patch('argparse.ArgumentParser.parse_args')
    @patch('scripts.optimize_hyperparams.load_config')
    @patch('optuna.create_study')
    def test_main_flow(self, mock_create_study, mock_load_config, mock_parse_args, tmp_path):
        """
        Verifies that the main function correctly parses args, creates a study,
        and starts optimization.
        """
        # Setup: Mock argument parsing
        mock_args = argparse.Namespace(
            config='path/to/config.yaml',
            n_trials=10,
            study_name='test_study',
            storage_db='test.db',
            trials_on_5_percent=2,
            trials_on_20_percent=5,
            trials_on_50_percent=8,
            pruner='median'
        )
        mock_parse_args.return_value = mock_args

        # Mock the config object returned by load_config to control the output path
        mock_cfg = MagicMock()
        mock_cfg.paths.output_dir = tmp_path
        mock_load_config.return_value = mock_cfg

        mock_study = MagicMock()
        mock_best_trial = SimpleNamespace(value=0.9543, params={'param_a': 1})
        mock_study.best_trial = mock_best_trial
        mock_study.get_trials.return_value = []
        mock_study.trials = [mock_best_trial]

        mock_create_study.return_value = mock_study

        # Execute
        with patch('scripts.optimize_hyperparams.objective'), \
             patch('optuna.trial.TrialState'):
                main()

        # Assert
        mock_load_config.assert_called_once_with('path/to/config.yaml')

        # Construct the expected storage path according to the script's logic
        expected_storage_path = tmp_path / 'test.db'
        expected_storage_str = f"sqlite:///{expected_storage_path.resolve()}"

        mock_create_study.assert_called_once_with(
            direction="maximize",
            study_name='test_study',
            storage=expected_storage_str,
            load_if_exists=True,
            pruner=ANY
        )
        mock_study.optimize.assert_called_once()
        call_kwargs = mock_study.optimize.call_args.kwargs
        assert call_kwargs['n_trials'] == 10
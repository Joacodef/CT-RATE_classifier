import sys
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, call

import pytest
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.evaluation.reporting import generate_final_report, generate_csv_report

# --- Fixtures ---

@pytest.fixture
def mock_config(tmp_path: Path) -> SimpleNamespace:
    """Creates a mock SimpleNamespace config object for testing."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()

    config = SimpleNamespace(
        paths=SimpleNamespace(output_dir=output_dir),
        model=SimpleNamespace(type="TestNet3D"),
        pathologies=SimpleNamespace(columns=["Cardiomegaly", "Atelectasis", "Lung nodule"]),
        training=SimpleNamespace(
            learning_rate=1e-4,
            batch_size=4
            ),
        image_processing=SimpleNamespace(
            target_shape_dhw=(96, 128, 128),
            target_spacing=[1.5, 1.5, 1.5]
            )
    )
    return config

@pytest.fixture
def mock_history() -> dict:
    """Creates a mock training history dictionary for three epochs."""
    history = {
        'train_loss': [0.5, 0.4, 0.35],
        'valid_loss': [0.6, 0.45, 0.48],
        'metrics': [
            {  # Epoch 1
                'roc_auc_macro': 0.75, 'roc_auc_micro': 0.80,
                'f1_macro': 0.60, 'f1_micro': 0.65,
                'accuracy': 0.85, 'precision_macro': 0.58, 'recall_macro': 0.62,
                'Cardiomegaly_auc': 0.70, 'Atelectasis_auc': 0.80, 'Lung nodule_auc': 0.75,
                'Cardiomegaly_f1': 0.60, 'Atelectasis_f1': 0.65, 'Lung nodule_f1': 0.55
            },
            {  # Epoch 2 (Best Epoch)
                'roc_auc_macro': 0.85, 'roc_auc_micro': 0.88,
                'f1_macro': 0.70, 'f1_micro': 0.72,
                'accuracy': 0.90, 'precision_macro': 0.68, 'recall_macro': 0.72,
                'Cardiomegaly_auc': 0.80, 'Atelectasis_auc': 0.90, 'Lung nodule_auc': 0.85,
                'Cardiomegaly_f1': 0.70, 'Atelectasis_f1': 0.75, 'Lung nodule_f1': 0.65
            },
            {  # Epoch 3
                'roc_auc_macro': 0.82, 'roc_auc_micro': 0.86,
                'f1_macro': 0.68, 'f1_micro': 0.70,
                'accuracy': 0.88, 'precision_macro': 0.66, 'recall_macro': 0.70,
                'Cardiomegaly_auc': 0.78, 'Atelectasis_auc': 0.88, 'Lung nodule_auc': 0.80,
                'Cardiomegaly_f1': 0.68, 'Atelectasis_f1': 0.72, 'Lung nodule_f1': 0.62
            }
        ]
    }
    return history


# --- Test Classes ---

class TestGenerateFinalReport:
    """Tests for the main visual report generation function."""

    @patch('src.evaluation.reporting.generate_csv_report')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_report_generation_runs_and_calls_dependencies(
        self, mock_close, mock_savefig, mock_generate_csv, mock_history, mock_config
    ):
        """
        Verifies that `generate_final_report` runs without error, calls `savefig`
        for PNG and PDF, and calls `generate_csv_report`.
        """
        best_epoch_idx = 1 
        generate_final_report(mock_history, mock_config, best_epoch_idx)

        # 1. Assert that savefig was called for both PNG and PDF
        assert mock_savefig.call_count == 2
        
        expected_png_path = mock_config.paths.output_dir / 'training_report.png'
        expected_pdf_path = mock_config.paths.output_dir / 'training_report.pdf'

        # Check that the report was saved in both formats
        call_args_list = mock_savefig.call_args_list
        saved_paths = [c[0][0] for c in call_args_list]
        assert expected_png_path in saved_paths
        assert expected_pdf_path in saved_paths

        # 2. Assert that the CSV report generation was called with the correct best_epoch
        mock_generate_csv.assert_called_once_with(mock_history, mock_config, best_epoch_idx)

        # 3. Assert that the plot figure was closed
        mock_close.assert_called_once()

    @patch('src.evaluation.reporting.plt.savefig')
    def test_report_handles_minimal_history(self, mock_savefig, mock_config):
        """
        Ensures the report function can handle a history with only one epoch
        without crashing.
        """
        minimal_history = {
            'train_loss': [0.5],
            'valid_loss': [0.6],
            'metrics': [{'roc_auc_macro': 0.7}]
        }
        
        try:
            generate_final_report(minimal_history, mock_config, best_epoch_idx=0)
        except Exception as e:
            pytest.fail(f"generate_final_report failed with minimal history: {e}")

        # Check if savefig was called, indicating plotting occurred
        assert mock_savefig.call_count > 0


class TestGenerateCsvReport:
    """Tests for the data-based (CSV, JSON) report generation."""

    def test_csv_and_json_files_are_created(self, mock_history, mock_config):
        """
        Verifies that both the detailed CSV and summary JSON files are created.
        """
        best_epoch_idx = 1
        output_dir = mock_config.paths.output_dir
        
        generate_csv_report(mock_history, mock_config, best_epoch_idx)

        expected_csv_path = output_dir / 'training_metrics_detailed.csv'
        expected_json_path = output_dir / 'training_summary.json'
        
        assert expected_csv_path.exists(), "Detailed metrics CSV file was not created."
        assert expected_json_path.exists(), "Training summary JSON file was not created."

    def test_csv_content_is_correct(self, mock_history, mock_config):
        """
        Verifies the content of the generated CSV file.
        """
        best_epoch_idx = 1
        generate_csv_report(mock_history, mock_config, best_epoch_idx)
        
        csv_path = mock_config.paths.output_dir / 'training_metrics_detailed.csv'
        df = pd.read_csv(csv_path)

        # Check number of rows
        assert len(df) == len(mock_history['train_loss'])
        
        # Check key columns exist
        assert 'epoch' in df.columns
        assert 'train_loss' in df.columns
        assert 'valid_loss' in df.columns
        assert 'is_best' in df.columns
        assert 'roc_auc_macro' in df.columns
        assert 'Cardiomegaly_auc' in df.columns
        
        # Check 'is_best' column
        assert df.loc[best_epoch_idx, 'is_best'] == True
        assert df['is_best'].sum() == 1
        
        # Check a specific value
        assert df.loc[best_epoch_idx, 'roc_auc_macro'] == pytest.approx(0.85)
        assert df.loc[0, 'train_loss'] == pytest.approx(0.5)

    def test_json_content_is_correct(self, mock_history, mock_config):
        """
        Verifies the content of the generated summary JSON file.
        """
        best_epoch_idx = 1
        generate_csv_report(mock_history, mock_config, best_epoch_idx)
        
        json_path = mock_config.paths.output_dir / 'training_summary.json'
        with open(json_path, 'r') as f:
            summary = json.load(f)

        # Check existence and type of key fields
        assert isinstance(summary['Total Epochs'], int)
        assert isinstance(summary['Best Epoch'], int)
        assert isinstance(summary['Best Validation Loss'], float)
        assert isinstance(summary['Model Type'], str)

        # Check specific values
        assert summary['Total Epochs'] == 3
        assert summary['Best Epoch'] == best_epoch_idx + 1
        assert summary['Best ROC AUC (Macro)'] == pytest.approx(0.85)
        assert summary['Final Validation Loss'] == pytest.approx(mock_history['valid_loss'][-1])
        assert summary['Model Type'] == "TestNet3D"
        assert summary['Number of Pathologies'] == 3
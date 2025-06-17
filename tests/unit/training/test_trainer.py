# tests/unit/training/test_trainer.py
"""
Unit tests for training/trainer.py
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY
import pandas as pd 
import numpy as np 

# Add project root to path to allow direct imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Mocking os.getenv ---
MOCK_ENV_VARS = {
    "TRAIN_IMG_DIR": "/dummy/train_dir",
    "VALID_IMG_DIR": "/dummy/valid_dir",
    "BASE_PROJECT_DIR": "/dummy/project_dir",
    "CACHE_DIR": "/dummy/cache_dir"
}

def mock_getenv_module_level(variable_name, default=None):
    """Mock for os.getenv that returns predefined paths."""
    return MOCK_ENV_VARS.get(variable_name, default)

getenv_patcher = patch('os.getenv', side_effect=mock_getenv_module_level)
getenv_patcher.start()

# --- Imports ---
import pytest
import torch
import torch.nn as nn
# Import validate_epoch from the trainer module
from training.trainer import create_model, load_and_prepare_data, train_epoch, validate_epoch, train_model
from config import Config
from models.resnet3d import ResNet3D

# --- Test Class Definition for create_model ---
class TestCreateModel:
    # ... (tests for create_model remain the same as before) ...
    def test_create_model_resnet3d_success(self):
        mock_config = Config() 
        mock_config.MODEL_TYPE = "resnet3d"
        mock_config.NUM_PATHOLOGIES = 18
        mock_config.GRADIENT_CHECKPOINTING = False
        model = create_model(mock_config)
        assert isinstance(model, ResNet3D)
        assert model.use_checkpointing == mock_config.GRADIENT_CHECKPOINTING

    def test_create_model_resnet3d_with_checkpointing(self):
        mock_config = Config()
        mock_config.MODEL_TYPE = "resnet3d"
        mock_config.NUM_PATHOLOGIES = 10
        mock_config.GRADIENT_CHECKPOINTING = True
        model = create_model(mock_config)
        assert isinstance(model, ResNet3D)
        assert model.use_checkpointing == True

    def test_create_model_unknown_type_raises_value_error(self):
        mock_config = Config()
        mock_config.MODEL_TYPE = "unknown_model_type"
        mock_config.NUM_PATHOLOGIES = 18
        mock_config.GRADIENT_CHECKPOINTING = False
        with pytest.raises(ValueError, match="Unknown model type: unknown_model_type"):
            create_model(mock_config)

    def test_create_model_num_pathologies_respected(self):
        mock_config = Config()
        mock_config.MODEL_TYPE = "resnet3d"
        mock_config.NUM_PATHOLOGIES = 5
        mock_config.GRADIENT_CHECKPOINTING = False
        model = create_model(mock_config)
        num_output_features = None
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Sequential):
            final_linear_layer = model.fc[-1]
            if isinstance(final_linear_layer, nn.Linear):
                 num_output_features = final_linear_layer.out_features
        elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            num_output_features = model.fc.out_features
        assert num_output_features == mock_config.NUM_PATHOLOGIES


# --- Test Class Definition for load_and_prepare_data ---
@patch('pandas.read_csv') 
class TestLoadAndPrepareData:
    # ... (tests for load_and_prepare_data remain the same as before) ...
    @pytest.fixture
    def mock_config(self):
        config = Config()
        config.SELECTED_TRAIN_VOLUMES_CSV = Path("dummy_train_volumes.csv")
        config.SELECTED_VALID_VOLUMES_CSV = Path("dummy_valid_volumes.csv")
        config.TRAIN_LABELS_CSV = Path("dummy_train_labels.csv")
        config.VALID_LABELS_CSV = Path("dummy_valid_labels.csv")
        config.PATHOLOGY_COLUMNS = ["PathologyA", "PathologyB"]
        return config

    def test_load_and_prepare_data_success(self, mock_read_csv, mock_config):
        mock_train_volumes_df = pd.DataFrame({'VolumeName': ['train_vol1', 'train_vol2']})
        mock_valid_volumes_df = pd.DataFrame({'VolumeName': ['valid_vol1']})
        mock_train_labels_df = pd.DataFrame({
            'VolumeName': ['train_vol1', 'train_vol2', 'train_vol3'],
            'PathologyA': [1, np.nan, 0], 
            'PathologyB': [0, 1, 0]      
        })
        mock_valid_labels_df = pd.DataFrame({
            'VolumeName': ['valid_vol1', 'valid_vol4'],
            'PathologyA': [np.nan, 1], 
            'PathologyB': [0, 0]
        })
        def read_csv_side_effect(path):
            if path == mock_config.SELECTED_TRAIN_VOLUMES_CSV: return mock_train_volumes_df.copy()
            elif path == mock_config.SELECTED_VALID_VOLUMES_CSV: return mock_valid_volumes_df.copy()
            elif path == mock_config.TRAIN_LABELS_CSV: return mock_train_labels_df.copy()
            elif path == mock_config.VALID_LABELS_CSV: return mock_valid_labels_df.copy()
            raise FileNotFoundError(f"Unexpected path for read_csv: {path}")
        mock_read_csv.side_effect = read_csv_side_effect
        train_df, valid_df = load_and_prepare_data(mock_config)
        assert not train_df.empty
        assert len(train_df) == 2 
        assert train_df['PathologyA'].tolist() == [1, 0]
        assert train_df['PathologyB'].tolist() == [0, 1]
        for col in mock_config.PATHOLOGY_COLUMNS: assert train_df[col].dtype == int
        assert not valid_df.empty
        assert len(valid_df) == 1 
        assert valid_df['PathologyA'].tolist() == [0]
        assert valid_df['PathologyB'].tolist() == [0]
        for col in mock_config.PATHOLOGY_COLUMNS: assert valid_df[col].dtype == int
        expected_calls = [
            call(mock_config.SELECTED_TRAIN_VOLUMES_CSV), call(mock_config.SELECTED_VALID_VOLUMES_CSV),
            call(mock_config.TRAIN_LABELS_CSV), call(mock_config.VALID_LABELS_CSV),
        ]
        mock_read_csv.assert_has_calls(expected_calls, any_order=True)

    def test_load_and_prepare_data_file_not_found(self, mock_read_csv, mock_config):
        mock_read_csv.side_effect = FileNotFoundError("Mocked file not found")
        with pytest.raises(FileNotFoundError, match="Required CSV file not found"):
            load_and_prepare_data(mock_config)

    def test_load_and_prepare_data_generic_error_on_load(self, mock_read_csv, mock_config):
        mock_read_csv.side_effect = Exception("Mocked generic CSV loading error")
        with pytest.raises(RuntimeError, match="Error loading data"):
            load_and_prepare_data(mock_config)

    def test_load_and_prepare_data_empty_dataframe_error(self, mock_read_csv, mock_config):
        mock_train_volumes_df = pd.DataFrame({'VolumeName': ['train_vol1']}) 
        mock_valid_volumes_df = pd.DataFrame({'VolumeName': []}) 
        mock_train_labels_df = pd.DataFrame({'VolumeName': ['train_vol1'], 'PathologyA': [1], 'PathologyB': [0]})
        mock_valid_labels_df = pd.DataFrame({'VolumeName': ['valid_vol_unmatch'], 'PathologyA': [1], 'PathologyB': [0]})
        def read_csv_side_effect(path):
            if path == mock_config.SELECTED_TRAIN_VOLUMES_CSV: return mock_train_volumes_df
            elif path == mock_config.SELECTED_VALID_VOLUMES_CSV: return mock_valid_volumes_df
            elif path == mock_config.TRAIN_LABELS_CSV: return mock_train_labels_df
            elif path == mock_config.VALID_LABELS_CSV: return mock_valid_labels_df
            return pd.DataFrame()
        mock_read_csv.side_effect = read_csv_side_effect
        with pytest.raises(ValueError, match="Training or validation dataframe is empty"):
            load_and_prepare_data(mock_config)

    def test_load_and_prepare_data_missing_pathology_columns_error(self, mock_read_csv, mock_config):
        mock_train_volumes_df = pd.DataFrame({'VolumeName': ['train_vol1']})
        mock_valid_volumes_df = pd.DataFrame({'VolumeName': ['valid_vol1']})
        mock_train_labels_df = pd.DataFrame({'VolumeName': ['train_vol1'], 'PathologyA': [1]})
        mock_valid_labels_df = pd.DataFrame({'VolumeName': ['valid_vol1'], 'PathologyA': [0]})
        def read_csv_side_effect(path):
            if path == mock_config.SELECTED_TRAIN_VOLUMES_CSV: return mock_train_volumes_df
            elif path == mock_config.SELECTED_VALID_VOLUMES_CSV: return mock_valid_volumes_df
            elif path == mock_config.TRAIN_LABELS_CSV: return mock_train_labels_df 
            elif path == mock_config.VALID_LABELS_CSV: return mock_valid_labels_df
            return pd.DataFrame()
        mock_read_csv.side_effect = read_csv_side_effect
        mock_config.PATHOLOGY_COLUMNS = ["PathologyA", "PathologyB"] 
        with pytest.raises(ValueError, match="Missing pathology columns in training data: \\['PathologyB'\\]"):
            load_and_prepare_data(mock_config)


# --- Test Class Definition for train_epoch ---
class TestTrainEpoch:
    """
    Tests for the train_epoch function in trainer.py.
    """

    @pytest.fixture
    def mock_dependencies(self, mocker):
        """Fixture to create mock objects for train_epoch."""
        mocks = MagicMock()
        mocks.model = MagicMock(spec=nn.Module)
        mocks.dataloader = MagicMock() 
        mocks.criterion = MagicMock(spec=nn.Module) 
        mocks.optimizer = MagicMock(spec=torch.optim.Optimizer)
        mocks.scaler = MagicMock(spec=torch.cuda.amp.GradScaler) 
        mocks.device = torch.device('cpu') 
        
        mocks.logger = MagicMock()
        mocker.patch('training.trainer.logger', mocks.logger) 

        return mocks

    def _prepare_mock_batch(self, device):
        """Helper to create a mock batch."""
        pixel_values = torch.randn(2, 1, 3, 3, 3).to(device)
        labels = torch.randint(0, 2, (2, 5)).float().to(device)
        return {"pixel_values": pixel_values, "labels": labels}

    def _create_mock_loss_object(self, value, grad_accum_steps=1):
        """
        Helper to create a mock loss object setup.
        Returns a tuple: (loss_object_from_criterion, loss_object_after_division)
        """
        mock_loss_a = MagicMock(name=f"loss_A_val_{value}")
        mock_loss_a.item.return_value = float(value) 

        mock_loss_b = MagicMock(name=f"loss_B_val_{value/grad_accum_steps}")
        mock_loss_b.item.return_value = float(value) / grad_accum_steps
        
        mock_loss_a.__truediv__.return_value = mock_loss_b
        
        return mock_loss_a, mock_loss_b

    def test_train_epoch_basic_run_no_amp_no_accumulation(self, mock_dependencies):
        """Test basic run: no AMP, no gradient accumulation."""
        deps = mock_dependencies
        grad_accum_steps = 1
        batch1 = self._prepare_mock_batch(deps.device)
        batch2 = self._prepare_mock_batch(deps.device)
        deps.dataloader.__iter__.return_value = iter([batch1, batch2])
        deps.dataloader.__len__.return_value = 2

        loss_a1, loss_b1 = self._create_mock_loss_object(1.0, grad_accum_steps=grad_accum_steps)
        loss_a2, loss_b2 = self._create_mock_loss_object(0.5, grad_accum_steps=grad_accum_steps)
        
        # This mock tensor will be the same object returned by deps.model for all calls
        mock_model_output = torch.randn(2, 5).to(deps.device)
        deps.model.return_value = mock_model_output
        deps.criterion.side_effect = [loss_a1, loss_a2] 

        avg_loss = train_epoch(
            model=deps.model, dataloader=deps.dataloader, criterion=deps.criterion,
            optimizer=deps.optimizer, scaler=None, device=deps.device, epoch=0, total_epochs=10,
            gradient_accumulation_steps=grad_accum_steps, use_amp=False
        )

        deps.model.train.assert_called_once()
        assert deps.optimizer.zero_grad.call_count == 3 
        
        # --- Check model calls ---
        assert deps.model.call_count == 2
        model_called_with_batch1_pixels = any(
            torch.equal(call_args[0][0], batch1["pixel_values"]) for call_args in deps.model.call_args_list
        )
        assert model_called_with_batch1_pixels, "Model not called with batch1 pixel_values"
        
        model_called_with_batch2_pixels = any(
            torch.equal(call_args[0][0], batch2["pixel_values"]) for call_args in deps.model.call_args_list
        )
        assert model_called_with_batch2_pixels, "Model not called with batch2 pixel_values"

        # --- Check criterion calls ---
        assert deps.criterion.call_count == 2
        criterion_call_args_list = deps.criterion.call_args_list
        
        # Check call for batch1
        assert torch.equal(criterion_call_args_list[0][0][0], mock_model_output) # Arg 0 of first call
        assert torch.equal(criterion_call_args_list[0][0][1], batch1["labels"])   # Arg 1 of first call
        loss_b1.backward.assert_called_once() 
        
        # Check call for batch2
        assert torch.equal(criterion_call_args_list[1][0][0], mock_model_output) # Arg 0 of second call
        assert torch.equal(criterion_call_args_list[1][0][1], batch2["labels"])   # Arg 1 of second call
        loss_b2.backward.assert_called_once() 

        assert deps.optimizer.step.call_count == 2
        assert avg_loss == pytest.approx((loss_a1.item.return_value + loss_a2.item.return_value) / 2)
        deps.logger.info.assert_any_call(
            f"Epoch [{1}/{10}] Batch [{0}/{2}] Loss: {loss_a1.item.return_value:.4f}"
        )


    def test_train_epoch_with_gradient_accumulation(self, mock_dependencies):
        """Test with gradient accumulation."""
        deps = mock_dependencies
        num_batches = 4
        gradient_accumulation_steps = 2
        batches = [self._prepare_mock_batch(deps.device) for _ in range(num_batches)]
        deps.dataloader.__iter__.return_value = iter(batches)
        deps.dataloader.__len__.return_value = num_batches

        mock_criterion_outputs_A = []
        mock_losses_for_backward_B = []

        for i in range(num_batches):
            loss_val = float(i + 1)
            loss_a, loss_b = self._create_mock_loss_object(loss_val, grad_accum_steps=gradient_accumulation_steps)
            mock_criterion_outputs_A.append(loss_a)
            mock_losses_for_backward_B.append(loss_b)
        
        deps.model.return_value = torch.randn(2, 5).to(deps.device)
        deps.criterion.side_effect = mock_criterion_outputs_A

        avg_loss = train_epoch(
            model=deps.model, dataloader=deps.dataloader, criterion=deps.criterion,
            optimizer=deps.optimizer, scaler=None, device=deps.device, epoch=0, total_epochs=10,
            gradient_accumulation_steps=gradient_accumulation_steps, use_amp=False
        )

        deps.model.train.assert_called_once()
        deps.optimizer.zero_grad.assert_any_call() 
        
        for mock_loss_b_item in mock_losses_for_backward_B:
            mock_loss_b_item.backward.assert_called_once()

        assert deps.optimizer.step.call_count == num_batches / gradient_accumulation_steps 
        assert deps.optimizer.zero_grad.call_count == (num_batches / gradient_accumulation_steps) + 1 

        expected_total_sum_loss_A = sum(l_a.item.return_value for l_a in mock_criterion_outputs_A)
        assert avg_loss == pytest.approx(expected_total_sum_loss_A / num_batches)


    @patch('torch.cuda.amp.autocast') 
    def test_train_epoch_with_amp(self, mock_autocast, mock_dependencies):
        """Test with Automatic Mixed Precision (AMP)."""
        deps = mock_dependencies
        if not torch.cuda.is_available():
             pytest.skip("CUDA not available, skipping AMP test")
        deps.device = torch.device('cuda') 

        batch = self._prepare_mock_batch(deps.device) 
        deps.dataloader.__iter__.return_value = iter([batch])
        deps.dataloader.__len__.return_value = 1
        
        gradient_accumulation_steps = 1
        loss_val = 1.0
        loss_a, loss_b = self._create_mock_loss_object(loss_val, grad_accum_steps=gradient_accumulation_steps)

        mock_model_output_amp = torch.randn(2,5).to(deps.device)
        deps.model.return_value = mock_model_output_amp
        deps.criterion.return_value = loss_a 
        
        mock_scaled_loss_for_scaler_backward = MagicMock(name="scaled_loss_for_scaler_backward")
        deps.scaler.scale.return_value = mock_scaled_loss_for_scaler_backward
        
        avg_loss = train_epoch(
            model=deps.model, dataloader=deps.dataloader, criterion=deps.criterion,
            optimizer=deps.optimizer, scaler=deps.scaler, device=deps.device, epoch=0, total_epochs=10,
            gradient_accumulation_steps=gradient_accumulation_steps, use_amp=True
        )

        mock_autocast.assert_called_once() 
        
        deps.model.assert_called_once()
        called_args_model, _ = deps.model.call_args
        assert torch.equal(called_args_model[0], batch["pixel_values"])

        deps.criterion.assert_called_once()
        crit_args, _ = deps.criterion.call_args
        assert torch.equal(crit_args[0], mock_model_output_amp)
        assert torch.equal(crit_args[1], batch["labels"])

        deps.scaler.scale.assert_called_once_with(loss_b)
        mock_scaled_loss_for_scaler_backward.backward.assert_called_once()
        deps.scaler.step.assert_called_once_with(deps.optimizer)
        deps.scaler.update.assert_called_once()
        
        assert avg_loss == pytest.approx(loss_a.item.return_value / 1)

# --- Test Class Definition for validate_epoch ---
class TestValidateEpoch:
    """
    Tests for the validate_epoch function in trainer.py.
    """

    @pytest.fixture
    def mock_val_dependencies(self, mocker):
        """Fixture to create mock objects for validate_epoch."""
        mocks = MagicMock()
        mocks.model = MagicMock(spec=nn.Module)
        mocks.dataloader = MagicMock() 
        mocks.criterion = MagicMock(spec=nn.Module) 
        mocks.device = torch.device('cpu') 
        return mocks

    def _prepare_mock_val_batch(self, device, batch_size=2, num_pathologies=5):
        """Helper to create a mock validation batch."""
        pixel_values = torch.randn(batch_size, 1, 3, 3, 3).to(device)
        labels = torch.randint(0, 2, (batch_size, num_pathologies)).float().to(device)
        mock_model_output = torch.randn(batch_size, num_pathologies).to(device)
        return {"pixel_values": pixel_values, "labels": labels}, mock_model_output

    def test_validate_epoch_basic_run(self, mock_val_dependencies):
        """Test basic run of validate_epoch."""
        deps = mock_val_dependencies
        batch_size = 2
        num_pathologies = 3 
        
        batch1_data, model_output1 = self._prepare_mock_val_batch(deps.device, batch_size, num_pathologies)
        batch2_data, model_output2 = self._prepare_mock_val_batch(deps.device, batch_size, num_pathologies)
        
        deps.dataloader.__iter__.return_value = iter([batch1_data, batch2_data])
        deps.dataloader.__len__.return_value = 2

        deps.model.side_effect = [model_output1, model_output2]
        
        mock_loss_val1 = MagicMock(spec=torch.Tensor); mock_loss_val1.item.return_value = 1.0
        mock_loss_val2 = MagicMock(spec=torch.Tensor); mock_loss_val2.item.return_value = 0.5
        deps.criterion.side_effect = [mock_loss_val1, mock_loss_val2]

        avg_loss, all_predictions, all_labels = validate_epoch(
            model=deps.model,
            dataloader=deps.dataloader,
            criterion=deps.criterion,
            device=deps.device
        )

        deps.model.eval.assert_called_once()
        assert deps.model.call_count == 2
        
        model_calls = deps.model.call_args_list
        assert torch.equal(model_calls[0][0][0], batch1_data["pixel_values"])
        assert torch.equal(model_calls[1][0][0], batch2_data["pixel_values"])

        criterion_calls = deps.criterion.call_args_list
        assert torch.equal(criterion_calls[0][0][0], model_output1)
        assert torch.equal(criterion_calls[0][0][1], batch1_data["labels"])
        assert torch.equal(criterion_calls[1][0][0], model_output2)
        assert torch.equal(criterion_calls[1][0][1], batch2_data["labels"])

        assert avg_loss == pytest.approx((1.0 + 0.5) / 2)
        
        expected_predictions_np = np.concatenate([model_output1.cpu().numpy(), model_output2.cpu().numpy()], axis=0)
        expected_labels_np = np.concatenate([batch1_data["labels"].cpu().numpy(), batch2_data["labels"].cpu().numpy()], axis=0)
        
        assert isinstance(all_predictions, np.ndarray)
        assert isinstance(all_labels, np.ndarray)
        assert np.array_equal(all_predictions, expected_predictions_np)
        assert np.array_equal(all_labels, expected_labels_np)

    def test_validate_epoch_empty_dataloader(self, mock_val_dependencies):
        """Test validate_epoch with an empty dataloader."""
        deps = mock_val_dependencies
        deps.dataloader.__iter__.return_value = iter([]) 
        deps.dataloader.__len__.return_value = 0 # Mock a dataloader that reports length 0

        # Expect ZeroDivisionError because avg_loss = total_loss / len(dataloader)
        with pytest.raises(ZeroDivisionError):
            validate_epoch(
                model=deps.model,
                dataloader=deps.dataloader,
                criterion=deps.criterion,
                device=deps.device
            )
        
        # model.eval() is called before the potential division by zero
        deps.model.eval.assert_called_once()
        # The model's forward pass and criterion should not be called if the dataloader loop is not entered
        deps.model.assert_not_called() # Checks if model(...) was called
        deps.criterion.assert_not_called()
        
class TestTrainModel:
    @pytest.fixture
    def actual_mock_config(self, tmp_path: Path): # Renamed fixture
        """Creates a Config instance for testing train_model, using tmp_path."""
        config = Config() # Assuming Config() sets up most defaults

        # Override for test speed and isolation
        config.NUM_EPOCHS = 1 # Minimal epochs for testing structure
        config.OUTPUT_DIR = tmp_path / "test_output_dir"
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        config.PATHOLOGY_COLUMNS = ["PathologyA", "PathologyB"] # Simplified
        config.NUM_PATHOLOGIES = len(config.PATHOLOGY_COLUMNS)
        config.ORIENTATION_AXCODES = "LPS_Test" # Use a distinct value for testing

        # Critical: Ensure RESUME_FROM_CHECKPOINT is None by default for other tests,
        # but will be set in the specific test_train_model_resumes_from_checkpoint.
        config.RESUME_FROM_CHECKPOINT = None

        # Provide paths for CSV files within tmp_path
        # Create minimal dummy files so that pandas.read_csv (if not fully mocked) doesn't fail
        dummy_volume_data = {'VolumeName': ['vol_train_1', 'vol_train_2']}
        dummy_label_data_train = {'VolumeName': ['vol_train_1', 'vol_train_2'], 'PathologyA': [1,0], 'PathologyB': [0,1]}
        
        config.SELECTED_TRAIN_VOLUMES_CSV = tmp_path / "selected_train_volumes.csv"
        pd.DataFrame(dummy_volume_data).to_csv(config.SELECTED_TRAIN_VOLUMES_CSV, index=False)
        
        config.TRAIN_LABELS_CSV = tmp_path / "train_predicted_labels.csv"
        pd.DataFrame(dummy_label_data_train).to_csv(config.TRAIN_LABELS_CSV, index=False)

        dummy_volume_data_valid = {'VolumeName': ['vol_valid_1']}
        dummy_label_data_valid = {'VolumeName': ['vol_valid_1'], 'PathologyA': [1], 'PathologyB': [1]}
        config.SELECTED_VALID_VOLUMES_CSV = tmp_path / "selected_valid_volumes.csv"
        pd.DataFrame(dummy_volume_data_valid).to_csv(config.SELECTED_VALID_VOLUMES_CSV, index=False)

        config.VALID_LABELS_CSV = tmp_path / "valid_predicted_labels.csv"
        pd.DataFrame(dummy_label_data_valid).to_csv(config.VALID_LABELS_CSV, index=False)
        
        config.TRAIN_IMG_DIR = tmp_path / "dummy_train_img_dir"
        config.VALID_IMG_DIR = tmp_path / "dummy_valid_img_dir"
        config.CACHE_DIR = tmp_path / "dummy_cache_dir"
        config.TRAIN_IMG_DIR.mkdir(exist_ok=True)
        config.VALID_IMG_DIR.mkdir(exist_ok=True)
        config.CACHE_DIR.mkdir(exist_ok=True)

        config.LOSS_FUNCTION = "FocalLoss" # Default to avoid BCE specific logic for this test
        config.MIXED_PRECISION = True # To ensure scaler is created for the test case
        
        return config


    # Decorators are applied from bottom-up. Innermost is the first mock arg after self.
    # The order of arguments in def test_... MUST match this order.
    @patch('json.load')                             # mock_json_load (19th)
    @patch('builtins.open')                         # mock_builtin_open (18th)
    @patch('training.trainer.generate_final_report')# mock_generate_report (17th)
    @patch('training.trainer.wandb.watch')          # mock_wandb_watch (16th)
    @patch('training.trainer.wandb.init')           # mock_wandb_init (15th)
    @patch('training.trainer.load_checkpoint')      # mock_load_checkpoint_func (14th)
    @patch('training.trainer.Path')                 # mock_path_class (13th)
    @patch('training.trainer.save_checkpoint')      # mock_save_checkpoint (12th)
    @patch('training.trainer.compute_metrics', return_value={'roc_auc_macro': 0.95, 'f1_macro': 0.9}) # mock_compute_metrics (11th)
    @patch('training.trainer.validate_epoch', return_value=(0.4, np.array([[0.1,0.8]]), np.array([[0,1]]))) # mock_validate_epoch (10th)
    @patch('training.trainer.train_epoch', return_value=0.5) # mock_train_epoch (9th)
    @patch('torch.optim.lr_scheduler.CosineAnnealingLR') # mock_scheduler_class (8th)
    @patch('torch.optim.AdamW')                     # mock_optimizer_class (7th)
    @patch('training.trainer.nn.BCEWithLogitsLoss') # mock_bce_loss_fn (6th)
    @patch('training.trainer.FocalLoss')            # mock_focal_loss_fn (5th)
    @patch('training.trainer.create_model')         # mock_create_model (4th)
    @patch('training.trainer.DataLoader')           # mock_dataloader (3rd)
    @patch('training.trainer.CTDataset3D')          # mock_ctdataset3d (2nd)
    @patch('training.trainer.load_and_prepare_data')# mock_load_and_prepare_data_func (1st)
    def test_train_model_resumes_from_checkpoint(
        self, 
        # Arguments must be in order of patches from innermost to outermost
        mock_load_and_prepare_data_func, 
        mock_ctdataset3d,
        mock_dataloader,
        mock_create_model,
        mock_focal_loss_fn,
        mock_bce_loss_fn,
        mock_optimizer_class,
        mock_scheduler_class,
        mock_train_epoch,
        mock_validate_epoch,
        mock_compute_metrics,
        mock_save_checkpoint,
        mock_path_class,
        mock_load_checkpoint_func,
        mock_wandb_init,
        mock_wandb_watch,
        mock_generate_report,
        mock_builtin_open, 
        mock_json_load,    
        actual_mock_config # This is the pytest fixture, correctly named and positioned
    ):
        # --- Setup for resuming ---
        checkpoint_path_str = "dummy_checkpoint.pth" # Relative path string
        actual_mock_config.RESUME_FROM_CHECKPOINT = checkpoint_path_str
        
        # Configure the mock Path object factory (mock_path_class)
        mock_checkpoint_path_obj = MagicMock(spec=Path)
        mock_checkpoint_path_obj.exists.return_value = True # Checkpoint file "exists"
        
        mock_history_path_obj = MagicMock(spec=Path)
        mock_history_path_obj.exists.return_value = False # History file "does not exist" for this test

        def path_side_effect(path_arg):
            # Convert path_arg to string for comparison, as it could be Path object or string
            path_arg_str = str(path_arg)
            if path_arg_str == checkpoint_path_str:
                return mock_checkpoint_path_obj
            elif path_arg_str == str(actual_mock_config.OUTPUT_DIR / 'training_history.json'):
                return mock_history_path_obj
            # If Path() is called with the OUTPUT_DIR itself (which is already a Path object from tmp_path)
            elif path_arg is actual_mock_config.OUTPUT_DIR:
                 return actual_mock_config.OUTPUT_DIR # Return the real tmp_path Path object
            
            # Default behavior for other Path() instantiations: return a new MagicMock for Path
            # This allows calls like `Path(...).mkdir()` without pre-configuring every possible path.
            new_mock_path = MagicMock(spec=Path)
            new_mock_path.__str__ = MagicMock(return_value=path_arg_str) # So str(Path(...)) works
            new_mock_path.name = Path(path_arg_str).name
            new_mock_path.stem = Path(path_arg_str).stem
            new_mock_path.parent = MagicMock(spec=Path) # Mock parent to allow parent.mkdir
            new_mock_path.parent.mkdir.return_value = None
            new_mock_path.__truediv__.side_effect = lambda other: Path(path_arg_str) / other # Support / operator
            new_mock_path.exists.return_value = False # Default for other paths
            return new_mock_path

        mock_path_class.side_effect = path_side_effect
        
        # Configure return values for mocked functions
        resumed_epoch = 5
        resumed_metrics = {'roc_auc_macro': 0.88} 
        mock_load_checkpoint_func.return_value = (resumed_epoch, resumed_metrics)

        mock_scheduler_instance = MagicMock()
        mock_scheduler_class.return_value = mock_scheduler_instance
        
        mock_load_and_prepare_data_func.return_value = (
            pd.DataFrame({'VolumeName': ['vol1'], actual_mock_config.PATHOLOGY_COLUMNS[0]: [1], actual_mock_config.PATHOLOGY_COLUMNS[1]: [0]}),
            pd.DataFrame({'VolumeName': ['vol2'], actual_mock_config.PATHOLOGY_COLUMNS[0]: [0], actual_mock_config.PATHOLOGY_COLUMNS[1]: [1]})
        )
        
        # --- Mocking model and its .to() call ---
        mock_model_instance_original = MagicMock(spec=torch.nn.Module, name="model_original")
        mock_model_instance_original.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        
        mock_model_after_to = MagicMock(spec=torch.nn.Module, name="model_after_to_device")
        # Ensure the .to() mock also supports load_state_dict if it's called on the .to() result by load_checkpoint
        # This is important if load_checkpoint itself tries to call model.load_state_dict
        mock_model_after_to.load_state_dict = MagicMock()


        mock_create_model.return_value = mock_model_instance_original
        mock_model_instance_original.to.return_value = mock_model_after_to
        
        actual_mock_config.NUM_EPOCHS = resumed_epoch + 2 # e.g., train for 2 more epochs

        # Mock wandb.init to return a mock run object
        mock_wandb_run = MagicMock()
        mock_wandb_init.return_value = mock_wandb_run
        
        # --- Call train_model ---
        train_model(actual_mock_config)

        # --- Assertions for resuming ---
        # Assert load_checkpoint was called correctly
        # The scaler is created if actual_mock_config.MIXED_PRECISION is True
        # If it is, a GradScaler instance will be passed. ANY handles this.
        mock_load_checkpoint_func.assert_called_once_with(
            # Path(checkpoint_path_str) would be called inside train_model,
            # which our mock_path_class.side_effect handles and returns mock_checkpoint_path_obj.
            # However, train_model receives the string from config.
            # The actual argument passed to load_checkpoint is this string.
            checkpoint_path_str, # Check if it's the string or Path object
            mock_model_after_to, 
            mock_optimizer_class.return_value, 
            ANY # For the scaler (can be None or GradScaler instance)
        )
        
        # Assert scheduler was advanced correctly
        # In train_model: for _ in range(start_epoch): scheduler.step()
        # start_epoch = checkpoint_epoch + 1 = resumed_epoch + 1
        expected_initial_scheduler_steps = resumed_epoch + 1
        
        # The training loop runs for (actual_mock_config.NUM_EPOCHS - start_epoch) times
        # Each loop iteration also calls scheduler.step()
        expected_loop_iterations = actual_mock_config.NUM_EPOCHS - (resumed_epoch + 1)
        
        total_expected_scheduler_steps = expected_initial_scheduler_steps + expected_loop_iterations
        assert mock_scheduler_instance.step.call_count == total_expected_scheduler_steps

        # Assert train_epoch call count
        assert mock_train_epoch.call_count == expected_loop_iterations

        # Assert wandb.watch was called with the model after .to(device)
        mock_wandb_watch.assert_called_with(mock_model_after_to, log="gradients", log_freq=100)


# --- Cleanup ---
def teardown_module(module):
    """Stop the module-level patcher."""
    getenv_patcher.stop()
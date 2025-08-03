import torch
import logging
from pathlib import Path
from typing import Optional
logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Monitors a metric and stops training when it stops improving.

    This utility halts the training process once the model's performance on a
    validation metric ceases to show improvement over a specified number of
    epochs. This helps prevent overfitting and saves computational resources.

    Attributes:
        patience (int):
            The number of epochs to wait for an improvement before stopping.
        min_delta (float):
            The minimum change in the monitored metric to be considered an
            improvement.
        mode (str):
            One of 'min' or 'max'. In 'min' mode, training stops when the
            monitored quantity stops decreasing. In 'max' mode, it stops
            when the quantity stops increasing.
        counter (int):
            The number of epochs that have passed without improvement.
        best_value (Optional[float]):
            The best score observed for the monitored metric so far.
        early_stop (bool):
            A flag that becomes True when training should be stopped.
    """
    
    def __init__(self, patience=5, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, value):
        """
        Evaluates the current metric value and updates the early stopping state.

        Args:
            value (float): The latest value of the metric to monitor (e.g.,
                           validation loss or AUC).

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


def save_checkpoint(model, optimizer, scaler, epoch, best_metrics, checkpoint_path):
    """
    Saves a training checkpoint to a file.

    The checkpoint includes the model's state, the optimizer's state, the AMP
    GradScaler's state (if used), the current epoch, and the latest validation
    metrics. This allows for the complete restoration of a training session.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scaler (Optional[torch.cuda.amp.GradScaler]): The GradScaler used for
                                                       mixed-precision training.
        epoch (int): The last completed epoch number.
        metrics (Dict[str, Any]): A dictionary of performance metrics from the
                                  last validation run.
        checkpoint_path (Path): The file path where the checkpoint will be saved.
    """

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metrics': best_metrics,
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None):
    """
    Loads a training checkpoint from a file.

    This function restores the state of the model, optimizer, and GradScaler
    from a saved checkpoint. It is designed to work whether loading for inference
    (model only) or for resuming training. The checkpoint is loaded onto the CPU
    first to prevent GPU memory issues.

    Args:
        checkpoint_path (Path): The path to the checkpoint file.
        model (torch.nn.Module): The model instance to load the state into.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer instance to
                                                     restore.
        scaler (Optional[torch.cuda.amp.GradScaler]): The GradScaler instance to
                                                      restore.

    Returns:
        Tuple[int, Dict[str, Any]]: A tuple containing the epoch number from the
                                    checkpoint and a dictionary of the saved metrics.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('best_metrics', {})


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """
    Finds the most recent checkpoint file in a given directory.

    The search prioritizes 'last_checkpoint.pth'. If not found, it searches for
    checkpoints named 'checkpoint_epoch_*.pth' and returns the one with the
    highest epoch number. This simplifies the process of resuming training.

    Args:
        output_dir (Path): The directory to search for checkpoints.

    Returns:
        Optional[Path]: The path to the latest checkpoint file, or None if no
                        checkpoints are found.
    """
    
    # Prioritize the 'last_checkpoint.pth' for immediate resumption.
    last_checkpoint = output_dir / 'last_checkpoint.pth'
    if last_checkpoint.exists():
        return last_checkpoint
    
    # If not found, search for the highest epoch-numbered checkpoint.
    checkpoint_pattern = "checkpoint_epoch_*.pth"
    checkpoints = list(output_dir.glob(checkpoint_pattern))
    
    if not checkpoints:
        return None
    
    # Extract epoch numbers and find the path corresponding to the latest one.
    epoch_numbers = []
    for cp in checkpoints:
        try:
            # Assumes filename format '..._epoch_NUM.pth'
            epoch_num = int(cp.stem.split('_')[-1])
            epoch_numbers.append((epoch_num, cp))
        except ValueError:
            continue
    
    if epoch_numbers:
        # Sort by epoch number in descending order and take the first one.
        epoch_numbers.sort(key=lambda x: x[0])
        return epoch_numbers[-1][1]
    
    return None

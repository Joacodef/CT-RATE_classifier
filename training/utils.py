import torch
import logging
from pathlib import Path
from typing import Optional
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=5, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, value):
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
    """Save model checkpoint"""
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
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('best_metrics', {})


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in the output directory"""
    
    # First check for last_checkpoint.pth
    last_checkpoint = output_dir / 'last_checkpoint.pth'
    if last_checkpoint.exists():
        return last_checkpoint
    
    # Otherwise, find the highest numbered epoch checkpoint
    checkpoint_pattern = "checkpoint_epoch_*.pth"
    checkpoints = list(output_dir.glob(checkpoint_pattern))
    
    if not checkpoints:
        return None
    
    # Extract epoch numbers and find the latest
    epoch_numbers = []
    for cp in checkpoints:
        try:
            epoch_num = int(cp.stem.split('_')[-1])
            epoch_numbers.append((epoch_num, cp))
        except ValueError:
            continue
    
    if epoch_numbers:
        epoch_numbers.sort(key=lambda x: x[0])
        return epoch_numbers[-1][1]
    
    return None

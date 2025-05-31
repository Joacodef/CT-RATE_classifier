import torch
import os

def setup_torch_optimizations():
    """Configure PyTorch for optimal performance"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(min(8, os.cpu_count()))
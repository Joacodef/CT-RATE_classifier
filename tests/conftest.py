# E:\ProyectoRN\code\conftest.py
import sys
from pathlib import Path

# Add the project root directory to sys.path
# This allows imports like 'from data.dataset import ...'
# or 'from models.resnet3d import ...' from any test file.
sys.path.insert(0, str(Path(__file__).parent.resolve()))
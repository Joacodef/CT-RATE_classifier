# tests/conftest.py

import sys
from pathlib import Path

# Add project root and src directory to sys.path to allow imports
# from modules in both locations (e.g., 'config', 'data', 'models', 'training').
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
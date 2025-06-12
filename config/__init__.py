# config/__init__.py
"""
Configuration module for CT 3D Classifier.

This module attempts to load the user-specific 'config.py' file. If it is
not found (e.g., in a CI/CD environment or on initial setup), it falls back
to the 'config_sample.py' file, which provides a default configuration.
"""

try:
    # Prioritize the user-specific, untracked configuration file.
    from .config import Config
except ImportError:
    # Fallback to the sample configuration if the primary file is not present.
    from .config_sample import Config

# Expose the loaded Config class at the package level.
__all__ = ['Config']
# src/config/__init__.py
"""
Configuration loading module for the CT 3D Classifier.

This file exposes the primary `load_config` function, making it directly
importable from the `src.config` package. This provides a convenient
and centralized entry point for accessing application configuration.
"""

from .config import load_config

__all__ = ["load_config"]
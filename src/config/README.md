# Configuration Module (`/src/config`)

This directory contains the project's configuration management system. Its purpose is to provide a centralized and robust way to load all settings, paths, and hyperparameters from external files, keeping configuration separate from the application code.

The system is designed to read human-readable **YAML files** for general configuration and **`.env` files** for environment-specific variables (like file paths and API keys).

## Core Components

The primary component of this module is the `config.py` script.

---

### `config.py`

This file defines the logic for loading, parsing, and processing the configuration files. It exposes a single main function, `load_config`.

**Key Functions:**

* **`load_config(config_path)`**: This is the central function that orchestrates the entire configuration loading process. It performs several critical steps:
    1.  **Loads `.env` File**: It automatically finds and loads the `.env` file from the project root, making environment variables available to the application.
    2.  **Parses YAML**: It reads the specified YAML configuration file.
    3.  **Substitutes Environment Variables**: It searches the loaded YAML content for placeholders like `${VAR_NAME}` and replaces them with the corresponding values from the environment variables. This allows sensitive information or machine-specific paths to be kept out of version control.
    4.  **Converts to Namespace**: The configuration dictionary is recursively converted into a `SimpleNamespace` object. This provides a more convenient and IDE-friendly way to access settings using dot notation (e.g., `config.model.type`) instead of dictionary syntax (`config['model']['type']`).
    5.  **Resolves Paths**: All paths specified in the configuration are resolved into absolute `pathlib.Path` objects. This makes the code robust and runnable from any directory.
    6.  **Type Conversion**: It performs necessary type conversions for specific parameters, such as converting lists of numbers into NumPy arrays or tuples and parsing the `torch_dtype` string into a `torch.dtype` object.

### Usage Example

To use the configuration system, simply import and call the `load_config` function at the beginning of a script.

```python
from src.config import load_config

# Load the configuration from a YAML file
config = load_config("configs/my_experiment.yaml")

# Access settings using dot notation
learning_rate = config.training.learning_rate
model_type = config.model.type
output_directory = config.paths.output_dir

print(f"Starting training for {model_type} with LR: {learning_rate}")
print(f"Output will be saved to: {output_directory}")
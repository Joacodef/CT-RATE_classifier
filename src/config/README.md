# Configuration Module (`/src/config`)

This directory contains the project's configuration management system. Its purpose is to load settings, paths, and hyperparameters from external files, separating configuration from the application code.

The system reads YAML files for general configuration and `.env` files for environment-specific variables.

## Core Components

The primary component of this module is the `config.py` script.

---

### `config.py`

This file defines the logic for loading, parsing, and processing the configuration files. It exposes a single main function, `load_config`.

**Key Functions:**

* **`load_config(config_path)`**: This function orchestrates the configuration loading process. It performs several steps:
    1.  **Loads `.env` File**: It finds and loads the `.env` file from the project root, making environment variables available to the application.
    2.  **Parses YAML**: It reads the specified YAML configuration file.
    3.  **Substitutes Environment Variables**: It searches the loaded YAML content for placeholders like `${VAR_NAME}` and replaces them with the corresponding values from the environment variables.
    4.  **Converts to Namespace**: The configuration dictionary is recursively converted into a `SimpleNamespace` object, which allows for accessing settings using dot notation (e.g., `config.model.type`).
    5.  **Resolves Paths**: All paths specified in the configuration are resolved into absolute `pathlib.Path` objects.
    6.  **Type Conversion**: It performs type conversions for specific parameters, such as converting lists of numbers into NumPy arrays or tuples and parsing the `torch_dtype` string into a `torch.dtype` object.
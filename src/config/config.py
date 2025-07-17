import os
import re
import yaml
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv
import numpy as np

def _substitute_env_vars(data):
    """
    Recursively substitutes environment variable placeholders in the config data.
    e.g., '${VAR_NAME}' is replaced by the value of the 'VAR_NAME' env var.
    """
    # Regex to find patterns like ${VAR_NAME}
    env_var_pattern = re.compile(r"\$\{(\w+)\}")

    if isinstance(data, dict):
        return {k: _substitute_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars(i) for i in data]
    elif isinstance(data, str):
        # Find all placeholders in the string
        matches = env_var_pattern.findall(data)
        for var_name in matches:
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(
                    f"Configuration Error: Environment variable '{var_name}' "
                    f"is not set but is required by the config."
                )
            # Replace the placeholder with the actual environment variable value
            data = data.replace(f"${{{var_name}}}", env_value)
        return data
    else:
        # Return data of other types as is
        return data

def _dict_to_namespace(data):
    """
    Recursively converts a dictionary to a nested SimpleNamespace object.
    This allows for attribute-style access (e.g., config.model.type).
    """
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = _dict_to_namespace(value)
        return SimpleNamespace(**data)
    elif isinstance(data, list):
        return [_dict_to_namespace(item) for item in data]
    return data

def load_config(config_path: str | Path) -> SimpleNamespace:
    """
    Loads, parses, and processes a YAML configuration file.

    This function performs the following steps:
    1. Loads the project's .env file to make environment variables available.
    2. Loads the specified YAML configuration file.
    3. Substitutes environment variable placeholders (e.g., ${VAR}) in the config.
    4. Resolves all relative paths into absolute pathlib.Path objects.
    5. Converts the configuration into a nested SimpleNamespace for easy access.
    6. Performs necessary type conversions (e.g., lists to numpy arrays).

    Args:
        config_path (str | Path): The path to the YAML configuration file.

    Returns:
        SimpleNamespace: A nested object containing the processed configuration.
    """
    # Find the project root (assuming src/config/config.py)
    project_root = Path(__file__).resolve().parents[2]
    # Load environment variables from the .env file at the project root
    load_dotenv(dotenv_path=project_root / ".env")

    # 1. Load the raw YAML file
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # 2. Substitute environment variables
    subst_config = _substitute_env_vars(raw_config)

    # 3. Convert dictionary to a nested SimpleNamespace
    cfg = _dict_to_namespace(subst_config)

    # 4. Resolve and process paths
    cfg.paths.base_project_dir = Path(cfg.paths.base_project_dir).resolve()
    cfg.paths.img_dir = Path(cfg.paths.img_dir).resolve()
    cfg.paths.cache_dir = Path(cfg.paths.cache_dir).resolve()

    cfg.paths.output_dir = Path(cfg.paths.output_dir).resolve()

    data_dir = Path(cfg.paths.data_dir).resolve()
    cfg.paths.data_dir = data_dir

    # Resolve paths relative to the data directory
    cfg.paths.data_subsets.train = (
        data_dir / cfg.paths.data_subsets.train
    )
    cfg.paths.data_subsets.valid = (
        data_dir / cfg.paths.data_subsets.valid
    )
    cfg.paths.labels.train = (
        data_dir / cfg.paths.labels.train
    )
    cfg.paths.labels.valid = (
        data_dir / cfg.paths.labels.valid
    )
    cfg.paths.reports.train = (
        data_dir / cfg.paths.reports.train
    )
    cfg.paths.reports.valid = (
        data_dir / cfg.paths.reports.valid
    )
    cfg.paths.metadata.train = (
        data_dir / cfg.paths.metadata.train
    )
    cfg.paths.metadata.valid = (
        data_dir / cfg.paths.metadata.valid
    )

    # 5. Perform final type conversions and add computed values
    cfg.image_processing.target_spacing = np.array(
        cfg.image_processing.target_spacing
    )
    cfg.model.vit_specific.patch_size = tuple(
        cfg.model.vit_specific.patch_size
    )
    cfg.image_processing.target_shape_dhw = tuple(
        cfg.image_processing.target_shape_dhw
    )
    cfg.pathologies.num_pathologies = len(cfg.pathologies.columns)

    return cfg
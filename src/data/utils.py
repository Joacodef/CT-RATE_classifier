from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_dynamic_image_path(base_dir: Path, volume_name: str, dir_structure: str) -> Path:
    """
    Constructs the full path to an existing NIfTI volume based on the source directory structure.

    Args:
        base_dir (Path): The root directory where the images are stored.
        volume_name (str): The volume identifier, which can be with or without the extension.
        dir_structure (str): The directory structure mode. Can be 'nested' or 'flat'.
                           This should correspond to the `paths.dir_structure` config value.

    Returns:
        Path: The full, resolved path to the NIfTI file.

    Raises:
        ValueError: If an unsupported mode is provided.
    """
    # Correctly handle volume names that may or may not have an extension.
    volume_stem = volume_name.replace(".nii.gz", "").replace(".nii", "")
    volume_filename = f"{volume_stem}.nii.gz"

    if dir_structure == 'nested':
        # Handles reading from a hierarchical structure.
        # e.g., train_1_a_1 -> base_dir/train_1/train_1_a/train_1_a_1.nii.gz
        parts = volume_stem.split('_')
        if len(parts) >= 3:
            subject_session = f"{parts[0]}_{parts[1]}"
            subject_session_scan = f"{parts[0]}_{parts[1]}_{parts[2]}"
            return base_dir / subject_session / subject_session_scan / volume_filename
        else:
            # If the name doesn't fit the nested pattern, assume it's in the base directory.
            logger.warning(
                f"Volume name '{volume_stem}' does not match the expected nested pattern. "
                f"Falling back to a flat search in the base directory."
            )
            return base_dir / volume_filename

    elif dir_structure == 'flat':
        # Handles reading from a simple, flat directory structure.
        return base_dir / volume_filename

    else:
        raise ValueError(f"Unsupported dir_structure: '{dir_structure}'. Choose 'nested' or 'flat'.")


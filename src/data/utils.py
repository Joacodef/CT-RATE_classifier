from pathlib import Path

def get_dynamic_image_path(base_img_dir: Path, volume_filename: str, mode: str = 'nested') -> Path:
    """
    Construct the full path to a NIfTI volume using different modes.

    Args:
        base_img_dir (Path): The root directory for the images.
        volume_filename (str): The volume identifier.
        mode (str): The path generation mode. Can be 'nested' or 'flat'.
                    - 'nested': For reading original files from a hierarchical
                                structure (e.g., base/subj/scan/file.nii.gz).
                                Falls back to flat structure for simple names.
                    - 'flat': For writing processed files to a flat directory,
                              appending '__transformed.nii.gz'.

    Returns:
        Path: The full, resolved path to the NIfTI file.

    Raises:
        ValueError: If an unsupported mode is provided.
    """
    if mode == 'nested':
        # This logic handles reading from a hierarchical structure.
        # It ensures the filename has the correct extension.
        if not volume_filename.endswith('.nii.gz'):
            if volume_filename.endswith('.nii'):
                volume_filename = volume_filename[:-4]  # Avoid .nii.nii.gz
            volume_filename += '.nii.gz'

        name_without_ext = volume_filename.replace('.nii.gz', '')
        parts = name_without_ext.split('_')

        if len(parts) >= 3:
            # Create hierarchical path for filenames with sufficient parts.
            subject_session = f"{parts[0]}_{parts[1]}"
            subject_session_scan = f"{parts[0]}_{parts[1]}_{parts[2]}"
            return base_img_dir / subject_session / subject_session_scan / volume_filename
        else:
            # Fallback to a flat structure for simpler filenames.
            return base_img_dir / volume_filename

    elif mode == 'flat':
        # This logic handles writing to a flat structure with a transformed suffix.
        filename_stem = Path(volume_filename).name
        if filename_stem.endswith(".nii.gz"):
            filename_stem = filename_stem[:-7]
        elif filename_stem.endswith(".nii"):
            filename_stem = filename_stem[:-4]

        transformed_filename = f"{filename_stem}__transformed.nii.gz"
        return base_img_dir / transformed_filename

    else:
        raise ValueError(f"Unsupported mode: '{mode}'. Choose 'nested' or 'flat'.")


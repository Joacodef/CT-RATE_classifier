from pathlib import Path

def get_dynamic_image_path(base_img_dir: Path, volume_filename_gz: str) -> Path:
    """
    Construct the full path to a NIfTI volume in a flat directory structure.

    This function takes a base directory and a volume identifier, which may
    contain slashes from an old directory structure. It extracts the final
    filename component, appends the standard .nii.gz suffix, and joins it
    with the base directory.

    Args:
        base_dir (Path): The root directory where all .nii.gz files are stored.
        volume_name (str): The volume identifier, e.g., "train/study/series_1".

    Returns:
        Path: The full, resolved path to the NIfTI file.
    
    Example:
        >>> get_image_path(Path("/data/processed"), "train/study/series_1")
        Path("/data/processed/series_1.nii.gz")
    """
    # Extract the final component (basename) from the volume_name string.
    # This makes the function robust to volume names that contain path separators.
    filename_stem = Path(volume_filename_gz).name

    # Construct the full path by joining the base directory and the processed filename.
    return base_img_dir / f"{filename_stem}"





# OLD:
# def get_dynamic_image_path(base_img_dir: Path, volume_filename_gz: str) -> Path:
#     """
#     Construct the full path to a NIfTI volume file based on a hierarchical directory structure.
    
#     This function handles a specific directory organization where volumes are stored in
#     nested folders based on their filename components. It's designed for datasets where
#     files follow a naming convention like "subject_session_scan.nii.gz" and are organized
#     in a hierarchy like: base_dir/subject_session/subject_session_scan/file.nii.gz
    
#     Args:
#         base_img_dir: Root directory containing all image volumes
#         volume_filename_gz: Filename of the NIfTI volume (with or without .nii.gz extension)
    
#     Returns:
#         Path: Full path to the volume file
    
#     Examples:
#         >>> get_dynamic_image_path(Path("/data"), "SUBJ01_SESS1_T1.nii.gz")
#         Path("/data/SUBJ01_SESS1/SUBJ01_SESS1_T1/SUBJ01_SESS1_T1.nii.gz")
        
#         >>> get_dynamic_image_path(Path("/data"), "SUBJ01_SESS1_T1")  # Extension added automatically
#         Path("/data/SUBJ01_SESS1/SUBJ01_SESS1_T1/SUBJ01_SESS1_T1.nii.gz")
        
#         >>> get_dynamic_image_path(Path("/data"), "simple_file.nii.gz")  # Fewer than 3 parts
#         Path("/data/simple_file.nii.gz")
    
#     Directory Structure Expected:
#         base_img_dir/
#         ├── SUBJ01_SESS1/
#         │   ├── SUBJ01_SESS1_T1/
#         │   │   └── SUBJ01_SESS1_T1.nii.gz
#         │   └── SUBJ01_SESS1_T2/
#         │       └── SUBJ01_SESS1_T2.nii.gz
#         └── simple_file.nii.gz  # Files with < 3 parts go in root
#     """
#     # Ensure the filename has the correct extension
#     # This allows flexibility in input while maintaining consistency
#     if not volume_filename_gz.endswith(".nii.gz"):
#         volume_filename_gz += ".nii.gz"
    
#     # Extract the base name without extension for parsing
#     name_without_ext = volume_filename_gz.replace(".nii.gz", "")
    
#     # Split filename into components using underscore as delimiter
#     # Expected format: "subject_session_scan" or similar
#     parts = name_without_ext.split('_')
    
#     # Build hierarchical path if filename has at least 3 components
#     if len(parts) >= 3:
#         # Create nested directory structure:
#         # Level 1: subject_session (first two parts)
#         # Level 2: subject_session_scan (first three parts)
#         # File: full filename with extension
#         subject_session = f"{parts[0]}_{parts[1]}"
#         subject_session_scan = f"{parts[0]}_{parts[1]}_{parts[2]}"
        
#         full_path = base_img_dir / subject_session / subject_session_scan / volume_filename_gz
#     else:
#         # For files with fewer than 3 parts, place directly in base directory
#         # This handles edge cases and simple filenames gracefully
#         full_path = base_img_dir / volume_filename_gz
    
#     return full_path
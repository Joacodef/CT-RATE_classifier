from pathlib import Path

def get_dynamic_image_path(base_img_dir: Path, volume_filename_gz: str) -> Path:
    """Get the full path to the volume file"""
    if not volume_filename_gz.endswith(".nii.gz"):
        volume_filename_gz += ".nii.gz"
        
    name_without_ext = volume_filename_gz.replace(".nii.gz", "")
    parts = name_without_ext.split('_')
    
    if len(parts) >= 3:
        full_path = base_img_dir / f"{parts[0]}_{parts[1]}" / f"{parts[0]}_{parts[1]}_{parts[2]}" / volume_filename_gz
    else:
        full_path = base_img_dir / volume_filename_gz
        
    return full_path
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# Add project root to Python path to allow src imports
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from scripts.data_preparation.create_filtered_dataset import (
    create_filtered_dataset,
    natural_sort_key,
)


@pytest.fixture
def mock_config_and_files(tmp_path: Path) -> SimpleNamespace:
    """
    Creates a temporary directory structure with mock data files and a
    corresponding config object for testing the filtering script.
    """
    data_dir = tmp_path / "data"
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create mock metadata files
    pd.DataFrame(
        {
            "VolumeName": [
                "train_10_scan_C",  # Should be kept, tests sorting
                "train_2_scan_A",  # Should be kept, tests sorting
                "brain_scan_to_exclude",  # Exact match exclusion
                "manual_prefix_1_scan_A",  # Prefix exclusion
            ]
        }
    ).to_csv(metadata_dir / "train_metadata.csv", index=False)

    pd.DataFrame(
        {
            "VolumeName": [
                "valid_1_scan_B",  # Should be kept
                "missing_z_scan_to_exclude.nii.gz",  # Exact match exclusion
                "manual_prefix_2_scan_B",  # Prefix exclusion
            ]
        }
    ).to_csv(metadata_dir / "valid_metadata.csv", index=False)

    # 2. Create mock exclusion files
    (data_dir / "brain_scans.txt").write_text("brain_scan_to_exclude")
    (data_dir / "missing_z_space.txt").write_text("missing_z_scan_to_exclude")
    pd.DataFrame(
        {"VolumeName": ["manual_prefix_1", "manual_prefix_2"]}
    ).to_csv(data_dir / "all_manual_labels.csv", index=False)

    # 3. Create the mock config object pointing to the temporary files
    config = SimpleNamespace(
        paths=SimpleNamespace(
            data_dir=data_dir,
            metadata=SimpleNamespace(
                train=metadata_dir / "train_metadata.csv",
                valid=metadata_dir / "valid_metadata.csv",
            ),
            exclusion_files=SimpleNamespace(
                brain_scans=data_dir / "brain_scans.txt",
                missing_z=data_dir / "missing_z_space.txt",
                manual_labels=data_dir / "all_manual_labels.csv",
            ),
            full_dataset_csv=data_dir / "filtered_master_list.csv",
        )
    )
    return config


def test_create_filtered_dataset_logic(mock_config_and_files: SimpleNamespace):
    """
    Tests the core filtering logic of the create_filtered_dataset script.

    It verifies that:
    - The script correctly identifies and keeps desired volumes.
    - Both exact-match and prefix-based exclusions work as expected.
    - The final output list is naturally sorted.
    - The output file is created at the location specified in the config.
    """
    # --- 1. Arrange ---
    config = mock_config_and_files
    # This is the expected list of volumes *after* filtering but *before* sorting.
    # The final assertion will compare against the sorted version of this list.
    expected_volumes = ["train_2_scan_A", "train_10_scan_C", "valid_1_scan_B"]
    expected_volumes.sort(key=natural_sort_key)  # Apply the same sorting as the script

    # --- 2. Act ---
    create_filtered_dataset(config)

    # --- 3. Assert ---
    output_path = config.paths.full_dataset_csv
    assert (
        output_path.exists()
    ), "The output CSV file was not created at the expected path."

    result_df = pd.read_csv(output_path)
    actual_volumes = result_df["VolumeName"].tolist()

    assert (
        actual_volumes == expected_volumes
    ), "The list of volumes in the output file does not match the expected result."
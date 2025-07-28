import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from scripts.data_preparation.create_filtered_dataset import (
    create_filtered_dataset,
    get_patient_id,
    natural_sort_key,
)


@pytest.mark.parametrize(
    "volume_name, expected_id",
    [
        ("train_123_a_1", "123"),
        ("valid_55_scanB_recon2", "55"),
        ("test_9_c", "9"),
        ("malformed", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_get_patient_id(volume_name, expected_id):
    """
    Tests the get_patient_id function with various formats.
    """
    assert get_patient_id(volume_name) == expected_id


@pytest.fixture
def mock_config_and_files(tmp_path: Path) -> SimpleNamespace:
    """
    Creates a temporary directory structure with mock data files and a
    corresponding config object for testing the patient-level filtering script.
    """
    data_dir = tmp_path / "data"
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create mock metadata files
    # Patient '5' has a scan here and another in the exclusion list.
    # Patient '8' has a scan here and another in an exact-match exclusion list.
    pd.DataFrame(
        {
            "VolumeName": [
                "train_10_scan_C",          # Keep
                "train_2_scan_A",           # Keep
                "train_5_scan_B",           # Exclude (patient-level)
                "train_8_scan_X",           # Exclude (patient-level)
            ]
        }
    ).to_csv(metadata_dir / "train_metadata.csv", index=False)

    pd.DataFrame(
        {
            "VolumeName": [
                "valid_1_scan_B",           # Keep
                "valid_5_scan_C",           # Manually labeled (causes exclusion)
                "valid_99_scan_Z.nii.gz",   # In missing_z list (causes exclusion)
            ]
        }
    ).to_csv(metadata_dir / "valid_metadata.csv", index=False)

    # 2. Create mock exclusion files
    # This file uses full filenames, which will be normalized by the script
    (data_dir / "brain_scans.txt").write_text("train_8_scan_X.nii.gz")
    (data_dir / "missing_z_space.txt").write_text("valid_99_scan_Z")
    
    # This file contains VolumeNames that act as prefixes.
    pd.DataFrame(
        {"VolumeName": ["valid_5_scan_C"]}
    ).to_csv(data_dir / "all_manual_labels.csv", index=False)

    config = SimpleNamespace(
        paths=SimpleNamespace(
            data_dir=data_dir, # This remains a full path
            output_filename="filtered_master_list.csv",
            metadata=SimpleNamespace(
                train=metadata_dir / "train_metadata.csv",
                valid=metadata_dir / "valid_metadata.csv",
            ),
            exclusion_files=SimpleNamespace(
                brain_scans="brain_scans.txt",
                missing_z="missing_z_space.txt",
                manual_labels="all_manual_labels.csv",
            ),
        )
    )
    return config


def test_create_filtered_dataset_patient_level_logic(mock_config_and_files: SimpleNamespace):
    """
    Tests the core patient-level filtering logic of the script.

    It verifies that if any scan from a patient is in an exclusion list,
    all scans from that same patient are removed from the final dataset.
    """
    # --- 1. Arrange ---
    config = mock_config_and_files
    
    # Expected volumes after patient-level filtering and natural sorting.
    # Patients 5, 8, and 99 should be completely removed.
    expected_volumes = ["valid_1_scan_B", "train_2_scan_A", "train_10_scan_C"]
    expected_volumes.sort(key=natural_sort_key)

    # --- 2. Act ---
    create_filtered_dataset(config)

    # --- 3. Assert ---
    output_path = config.paths.data_dir / config.paths.output_filename
    assert (
        output_path.exists()
    ), "The output CSV file was not created at the expected path."

    result_df = pd.read_csv(output_path)
    actual_volumes = result_df["VolumeName"].tolist()

    assert (
        actual_volumes == expected_volumes
    ), "The final list of volumes does not correctly reflect patient-level exclusion."
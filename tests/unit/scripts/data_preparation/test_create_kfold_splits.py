import pandas as pd
import pytest
from types import SimpleNamespace 
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from scripts.data_preparation.create_kfold_splits import create_kfold_splits


@pytest.fixture
def mock_data_and_config(tmp_path):
    """
    Creates mock CSV files and a corresponding config object for testing.

    The mock data includes multiple scans for several patients to ensure
    the grouping logic can be tested effectively.
    """
    # 1. Create Mock DataFrames
    master_data = []
    # Create 20 patients, some with multiple scans
    for i in range(20):
        num_scans = (i % 3) + 1  # Patients will have 1, 2, or 3 scans
        for j in range(num_scans):
            volume_name = f"train_P{i}_S{j}_R1"
            master_data.append({"VolumeName": volume_name})
    df_master = pd.DataFrame(master_data)

    labels_data = {
        "VolumeName": df_master["VolumeName"],
        # Create some alternating binary labels for stratification testing
        "pathology_1": [i % 2 for i in range(len(df_master))],
        "pathology_2": [(i + 1) % 2 for i in range(len(df_master))],
    }
    df_labels = pd.DataFrame(labels_data)

    # 2. Save mock CSVs to a temporary directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    master_csv_path = data_dir / "master_list.csv"
    labels_csv_path = data_dir / "labels.csv"
    df_master.to_csv(master_csv_path, index=False)
    df_labels.to_csv(labels_csv_path, index=False)

    # 3. Create Mock Config using SimpleNamespace
    config = SimpleNamespace(
        paths=SimpleNamespace(
            data_dir=str(data_dir),
            full_dataset_csv="master_list.csv",
            labels=SimpleNamespace(all="labels.csv"),
        ),
        pathologies=SimpleNamespace(columns=["pathology_1", "pathology_2"]),
    )

    return config, data_dir


def test_create_kfold_splits_no_patient_leakage(mock_data_and_config):
    """
    Tests that create_kfold_splits correctly partitions the data without
    leaking patients between training and validation sets.
    """
    # --- 1. Arrange ---
    config, data_dir = mock_data_and_config
    n_splits = 3
    output_dir = data_dir / "splits"

    # --- 2. Act ---
    create_kfold_splits(config, n_splits, output_dir)

    # --- 3. Assert ---
    assert output_dir.exists()
    all_patients = set([f"P{i}" for i in range(20)])

    for i in range(n_splits):
        train_path = output_dir / f"train_fold_{i}.csv"
        valid_path = output_dir / f"valid_fold_{i}.csv"

        # Check that split files were created
        assert train_path.exists()
        assert valid_path.exists()

        df_train = pd.read_csv(train_path)
        df_valid = pd.read_csv(valid_path)

        # Extract patient IDs from each split
        train_patient_ids = set(
            df_train["VolumeName"].apply(lambda x: x.split("_")[1])
        )
        valid_patient_ids = set(
            df_valid["VolumeName"].apply(lambda x: x.split("_")[1])
        )

        # --- Core Assertions ---
        # 1. No patient IDs are shared between train and validation sets
        assert train_patient_ids.isdisjoint(valid_patient_ids), (
            f"Fold {i} has patient leakage. "
            f"Overlap: {train_patient_ids.intersection(valid_patient_ids)}"
        )

        # 2. The union of patients in train and valid equals all patients
        assert train_patient_ids.union(valid_patient_ids) == all_patients, (
            f"Fold {i} does not contain all patients."
        )
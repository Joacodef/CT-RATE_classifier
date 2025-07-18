import os
import shutil
import tempfile
import unittest
import torch
import sys
from pathlib import Path

from monai.data import PersistentDataset, Dataset
from monai.transforms import Compose, EnsureChannelFirstd, Spacingd

# Add the project root to the Python path to allow src imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.data.cache_utils import deterministic_hash, get_or_create_cache_subdirectory


class TestCachingIntegration(unittest.TestCase):
    """
    Integration test to verify the interaction between PersistentDataset
    and the custom hashing and caching utility functions using in-memory data.
    """

    def setUp(self):
        """
        Set up a temporary directory for the cache.
        """
        self.temp_dir = tempfile.mkdtemp()
        # Ensure cache_dir is a Path object for compatibility with the utility function.
        self.cache_dir = Path(self.temp_dir) / "persistent_cache"
        self.cache_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """
        Remove the temporary directory after the test.
        """
        shutil.rmtree(self.temp_dir)

    def test_cache_creation_with_in_memory_data(self):
        """
        Tests if PersistentDataset correctly caches in-memory data
        using the custom hashing functions.
        """
        # 1. Create a mock dataset in memory.
        mock_data = [
            {"image": torch.rand(10, 10, 10), "VolumeName": "vol_001"},
            {"image": torch.rand(10, 10, 10), "VolumeName": "vol_002"},
            {"image": torch.rand(10, 10, 10), "VolumeName": "vol_003"},
        ]
        in_memory_dataset = Dataset(data=mock_data)

        # 2. Define a transform pipeline.
        transforms = Compose(
            [
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ]
        )

        # 3. Manually create the expected cache subdirectory.
        split = "train"
        cache_path = get_or_create_cache_subdirectory(self.cache_dir, transforms, split)

        # 4. Wrap with PersistentDataset.
        persistent_dataset = PersistentDataset(
            data=in_memory_dataset,
            transform=transforms,
            cache_dir=cache_path,
            hash_func=deterministic_hash,
        )

        # 5. Trigger caching by iterating through the dataset.
        for item in persistent_dataset:
            self.assertIsNotNone(item)
            self.assertEqual(item['image'].shape[0], 1)

        # 6. Assert correctness.
        self.assertTrue(cache_path.is_dir())

        # Assert that the cached files exist with correctly hashed names.
        for item_info in in_memory_dataset:
            # The hash must be decoded from bytes to a string to match the filename.
            item_hash_str = deterministic_hash(item_info).decode('utf-8')
            expected_file_path = cache_path / f"{item_hash_str}.pt"
            self.assertTrue(
                expected_file_path.exists(),
                f"Cached file '{expected_file_path}' was not found for item '{item_info['VolumeName']}'.",
            )

        # Verify the number of files. The cache dir contains one .pt file per item
        # plus the cache_params.json file.
        expected_file_count = len(in_memory_dataset) + 1
        self.assertEqual(len(list(cache_path.iterdir())), expected_file_count)


if __name__ == "__main__":
    unittest.main()
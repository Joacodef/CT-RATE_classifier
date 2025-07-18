import tempfile
import unittest
from pathlib import Path
import torch
import shutil
import sys

from monai.transforms import Compose, Spacingd, EnsureChannelFirstd

# Add the project root to the Python path to allow src imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.data.cache_utils import deterministic_hash, get_or_create_cache_subdirectory


class TestHashingInvariance(unittest.TestCase):
    """
    Unit tests to verify the stability and specificity of the hashing logic.
    """

    def setUp(self):
        """
        Set up a temporary directory to be used as a base cache path.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def tearDown(self):
        """
        Remove the temporary directory after tests.
        """
        shutil.rmtree(self.temp_dir)

    def test_item_hash_is_invariant_to_data_changes(self):
        """
        Confirms that the item hash only depends on the 'VolumeName'
        and ignores other changes in the data dictionary.
        """
        # Two items with the same volume_name but different image data.
        item1 = {"volume_name": "vol_-123", "image": torch.rand(10, 10)}
        item2 = {"volume_name": "vol_-123", "image": torch.rand(20, 20)}

        # Calculate the hash for both items.
        hash1 = deterministic_hash(item1)
        hash2 = deterministic_hash(item2)

        # The hashes should be identical because the VolumeName is the same.
        self.assertEqual(hash1, hash2, "Hashes should be identical when only irrelevant data changes.")

        # A third item with a different VolumeName.
        item3 = {"volume_name": "vol_456", "image": torch.rand(10, 10)}
        hash3 = deterministic_hash(item3)

        # This hash should be different.
        self.assertNotEqual(hash1, hash3, "Hashes should be different for different VolumeNames.")

    def test_directory_hash_is_stable(self):
        """
        Confirms that the transform directory hash is stable and not affected
        by non-functional changes, like explicitly setting default parameters.
        """
        # Define two transform pipelines that are functionally identical.
        # `mode` defaults to "bilinear", and `channel_dim` to None in this context.
        transforms1 = Compose([
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0))
        ])

        transforms2 = Compose([
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear") # Explicit default
        ])

        # Get the cache directory path for both transform sets.
        path1 = get_or_create_cache_subdirectory(self.cache_dir, transforms1, "train")
        path2 = get_or_create_cache_subdirectory(self.cache_dir, transforms2, "train")

        # The paths should be identical because the underlying parameters are the same.
        self.assertEqual(path1, path2, "Cache paths should be identical for functionally equivalent transforms.")

        # Now, create a transform pipeline that is functionally different.
        transforms3 = Compose([
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0)) # Different pixdim
        ])
        path3 = get_or_create_cache_subdirectory(self.cache_dir, transforms3, "train")

        # This path should be different.
        self.assertNotEqual(path1, path3, "Cache paths should be different for different transform parameters.")


if __name__ == "__main__":
    unittest.main()
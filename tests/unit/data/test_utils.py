"""
Unit tests for data/utils.py

Tests cover:
- Path construction for various filename formats.
- Handling of filenames with and without extensions.
- Correct formation of hierarchical paths based on filename parts.
"""

import sys
from pathlib import Path

# Add project root to path for imports if running tests from a different directory
# This assumes the tests directory is structured as tests/unit/data/test_utils.py
# and the project root is four levels up.
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import pytest

# Ensure the module to be tested can be imported
# If 'data' is a package, this should work.
# If utils.py is directly under 'data', this is correct.
from data.utils import get_dynamic_image_path

class TestGetDynamicImagePath:
    """Tests for the get_dynamic_image_path function."""

    def test_filename_with_three_parts(self):
        """
        Test with a filename that has exactly three parts,
        expecting a hierarchical path.
        Example: SUBJ01_SESS1_T1.nii.gz -> base_dir/SUBJ01_SESS1/SUBJ01_SESS1_T1/SUBJ01_SESS1_T1.nii.gz
        """
        base_dir = Path("/test_data/ct_scans")
        volume_filename = "SUBJ01_SESS1_T1.nii.gz"
        expected_path = base_dir / "SUBJ01_SESS1" / "SUBJ01_SESS1_T1" / "SUBJ01_SESS1_T1.nii.gz"
        
        result_path = get_dynamic_image_path(base_dir, volume_filename)
        assert result_path == expected_path

    def test_filename_with_more_than_three_parts(self):
        """
        Test with a filename that has more than three parts,
        expecting a hierarchical path based on the first three parts.
        Example: SUBJ01_SESS1_T1_EXTRA.nii.gz -> base_dir/SUBJ01_SESS1/SUBJ01_SESS1_T1/SUBJ01_SESS1_T1_EXTRA.nii.gz
        The function logic uses the first three parts for directory structure,
        but the full filename for the actual file.
        """
        base_dir = Path("/archive/images")
        volume_filename = "P001_STUDY2_SCAN3_DWI.nii.gz"
        # According to the function logic:
        # subject_session = "P001_STUDY2"
        # subject_session_scan = "P001_STUDY2_SCAN3"
        # full_path = base_dir / "P001_STUDY2" / "P001_STUDY2_SCAN3" / "P001_STUDY2_SCAN3_DWI.nii.gz"
        expected_path = base_dir / "P001_STUDY2" / "P001_STUDY2_SCAN3" / "P001_STUDY2_SCAN3_DWI.nii.gz"
        
        result_path = get_dynamic_image_path(base_dir, volume_filename)
        assert result_path == expected_path

    def test_filename_with_two_parts(self):
        """
        Test with a filename that has two parts,
        expecting the file to be directly in the base directory.
        Example: STUDY01_CT.nii.gz -> base_dir/STUDY01_CT.nii.gz
        """
        base_dir = Path("/datasets")
        volume_filename = "STUDY01_CT.nii.gz"
        expected_path = base_dir / "STUDY01_CT.nii.gz"
        
        result_path = get_dynamic_image_path(base_dir, volume_filename)
        assert result_path == expected_path

    def test_filename_with_one_part(self):
        """
        Test with a filename that has only one part,
        expecting the file to be directly in the base directory.
        Example: image123.nii.gz -> base_dir/image123.nii.gz
        """
        base_dir = Path("/raw_files")
        volume_filename = "image123.nii.gz"
        expected_path = base_dir / "image123.nii.gz"
        
        result_path = get_dynamic_image_path(base_dir, volume_filename)
        assert result_path == expected_path

    def test_filename_without_nii_gz_extension(self):
        """
        Test with a filename missing the .nii.gz extension.
        The function should add it.
        Example: SUBJ02_SESSA_T2 -> base_dir/SUBJ02_SESSA/SUBJ02_SESSA_T2/SUBJ02_SESSA_T2.nii.gz
        """
        base_dir = Path("/images_input")
        volume_filename = "SUBJ02_SESSA_T2" # No extension
        expected_path = base_dir / "SUBJ02_SESSA" / "SUBJ02_SESSA_T2" / "SUBJ02_SESSA_T2.nii.gz"
        
        result_path = get_dynamic_image_path(base_dir, volume_filename)
        assert result_path == expected_path

    def test_filename_without_nii_gz_extension_few_parts(self):
        """
        Test with a filename missing .nii.gz and having few parts.
        Example: scan_45 -> base_dir/scan_45.nii.gz
        """
        base_dir = Path("/images_input")
        volume_filename = "scan_45" # No extension, two parts
        expected_path = base_dir / "scan_45.nii.gz"
        
        result_path = get_dynamic_image_path(base_dir, volume_filename)
        assert result_path == expected_path

    def test_filename_with_nii_extension_only(self):
        """
        Test with a filename having only .nii extension.
        The function's logic results in the .nii part becoming part of an intermediate directory name,
        and the final filename will have .nii.nii.gz.
        Example: IMG_001_FLAIR.nii -> base_dir/IMG_001/IMG_001_FLAIR.nii/IMG_001_FLAIR.nii.nii.gz
        """
        base_dir = Path("/data/testing")
        volume_filename_input = "IMG_001_FLAIR.nii"

        # Trace the function's logic to determine the expected path:
        # 1. volume_filename_gz (final filename part) becomes "IMG_001_FLAIR.nii.nii.gz"
        final_filename_component = "IMG_001_FLAIR.nii.nii.gz"

        # 2. name_without_ext for directory parts becomes "IMG_001_FLAIR.nii"
        # 3. parts = ["IMG", "001", "FLAIR.nii"]
        # 4. Intermediate directory parts from these:
        intermediate_dir1 = "IMG_001" # parts[0]_parts[1]
        intermediate_dir2 = "IMG_001_FLAIR.nii" # parts[0]_parts[1]_parts[2]

        expected_path = base_dir / intermediate_dir1 / intermediate_dir2 / final_filename_component

        result_path = get_dynamic_image_path(base_dir, volume_filename_input)
        assert result_path == expected_path, \
               f"Result path: {result_path}, Expected: {expected_path}. " \
               "The test expectation should match the function's behavior where '.nii' can be part of a directory name."


    def test_base_dir_with_trailing_slash(self):
        """
        Test if the base_dir path handling is robust (Pathlib usually handles this).
        """
        base_dir = Path("/data/with_slash/") # Pathlib normalizes this
        volume_filename = "TEST_01_SCAN.nii.gz"
        expected_path = base_dir / "TEST_01" / "TEST_01_SCAN" / "TEST_01_SCAN.nii.gz"
        
        result_path = get_dynamic_image_path(base_dir, volume_filename)
        assert result_path == expected_path

    def test_empty_filename_parts(self):
        """
        Test with filename that might result in empty parts if split by '_'.
        Example: __.nii.gz
        The function splits by '_'. parts will be ['', '', ''].
        subject_session = "_"
        subject_session_scan = "__"
        Path should be base_dir/_/__/__.nii.gz
        """
        base_dir = Path("/special_cases")
        volume_filename = "__.nii.gz" # Three parts, all empty strings
        expected_path = base_dir / "_" / "__" / "__.nii.gz"
        
        result_path = get_dynamic_image_path(base_dir, volume_filename)
        assert result_path == expected_path

    def test_filename_with_leading_underscores(self):
        """
        Test filename with leading underscores.
        Example: _SUBJ_SESS_SCAN.nii.gz -> base_dir/_SUBJ/_SUBJ_SESS/_SUBJ_SESS_SCAN.nii.gz
        """
        base_dir = Path("/more_cases")
        volume_filename = "_SUBJ_SESS_SCAN.nii.gz"
        # parts = ['', 'SUBJ', 'SESS', 'SCAN']
        # subject_session = "_SUBJ"
        # subject_session_scan = "_SUBJ_SESS"
        expected_path = base_dir / "_SUBJ" / "_SUBJ_SESS" / "_SUBJ_SESS_SCAN.nii.gz"

        result_path = get_dynamic_image_path(base_dir, volume_filename)
        assert result_path == expected_path

# To run these tests, you would typically use pytest from your terminal:
# pytest tests/unit/data/test_utils.py
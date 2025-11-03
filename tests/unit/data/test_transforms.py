import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from monai.transforms import (
    LoadImaged,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    Resized,
    EnsureTyped,
    Compose,
)

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.transforms import KeyCleanerD, get_preprocessing_transforms


class TestKeyCleanerD:
    def test_only_requested_keys_remain(self):
        transform = KeyCleanerD(keys_to_keep=["image", "label"])
        sample = {
            "image": torch.ones(1),
            "label": torch.tensor([1.0]),
            "image_meta_dict": {"affine": torch.eye(4)},
        }

        cleaned = transform(sample)

        assert set(cleaned.keys()) == {"image", "label"}
        assert cleaned["image"] is sample["image"]
        assert torch.equal(cleaned["label"], sample["label"])

    def test_transform_info_is_sorted(self):
        transform = KeyCleanerD(keys_to_keep=["b", "a"])

        info = transform.get_transform_info()

        assert info == {"class": "KeyCleanerD", "keys_to_keep": ["a", "b"]}


class TestGetPreprocessingTransforms:
    @pytest.fixture
    def config(self):
        image_processing = SimpleNamespace(
            orientation_axcodes=("L", "P", "S"),
            target_spacing=(1.0, 1.5, 2.0),
            clip_hu_min=-300.0,
            clip_hu_max=300.0,
            scale_b_min=0.1,
            scale_b_max=0.9,
            target_shape_dhw=(64, 96, 128),
        )
        return SimpleNamespace(image_processing=image_processing, torch_dtype=torch.float16)

    def test_pipeline_sequence_and_parameters(self, config):
        pipeline = get_preprocessing_transforms(config)

        assert isinstance(pipeline, Compose)
        transforms = pipeline.transforms

        assert [type(t) for t in transforms] == [
            LoadImaged,
            Orientationd,
            Spacingd,
            ScaleIntensityRanged,
            Resized,
            EnsureTyped,
            KeyCleanerD,
        ]

        loader = transforms[0]
        assert loader.keys == ("image",)

        orientation = transforms[1]
        assert tuple(orientation.ornt_transform.axcodes) == config.image_processing.orientation_axcodes

        spacing = transforms[2]
        assert tuple(spacing.spacing_transform.pixdim) == config.image_processing.target_spacing
        spacing_mode = spacing.mode if isinstance(spacing.mode, str) else spacing.mode[0]
        assert spacing_mode == "bilinear"

        scaling = transforms[3]
        assert scaling.scaler.a_min == config.image_processing.clip_hu_min
        assert scaling.scaler.a_max == config.image_processing.clip_hu_max
        assert scaling.scaler.b_min == config.image_processing.scale_b_min
        assert scaling.scaler.b_max == config.image_processing.scale_b_max
        assert scaling.scaler.clip is True

        resize = transforms[4]
        assert tuple(resize.resizer.spatial_size) == config.image_processing.target_shape_dhw
        resize_mode = resize.mode if isinstance(resize.mode, str) else resize.mode[0]
        assert resize_mode == "trilinear"

        ensure_typed = transforms[5]
        dtype = ensure_typed.dtype[0] if isinstance(ensure_typed.dtype, tuple) else ensure_typed.dtype
        assert dtype == config.torch_dtype

        cleaner = transforms[6]
        assert cleaner.keys_to_keep == {"image", "image_meta_dict"}
        assert cleaner.get_transform_info()["keys_to_keep"] == ["image", "image_meta_dict"]

# tests/unit/feature_extraction/test_feature_extraction.py
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from scripts.feature_extraction.generate_features_ct_clip import adapt_model_to_feature_extractor
from src.models.resnet3d import resnet18_3d
from src.models.densenet3d import densenet121_3d
from src.models.vit3d import vit_tiny_3d

# --- Test Cases ---

class TestAdaptModelToFeatureExtractor:
    """
    Unit tests for the `adapt_model_to_feature_extractor` function.
    """

    @pytest.mark.parametrize(
        "model_fn, model_type, original_classifier_attr",
        [
            (resnet18_3d, "resnet3d", "fc"),
            (densenet121_3d, "densenet3d", "classifier"),
            (vit_tiny_3d, "vit3d", "head"),
        ],
    )
    def test_adaptation_replaces_classifier_with_identity(
        self, model_fn, model_type, original_classifier_attr
    ):
        """
        Verifies that the final classification layer of each model type
        is correctly replaced with an nn.Identity layer.
        """
        # 1. Arrange: Create a standard model instance
        model = model_fn(num_classes=18)
        # Verify the original classifier exists and is not an Identity layer
        original_classifier = getattr(model, original_classifier_attr)
        assert not isinstance(original_classifier, nn.Identity)
        assert isinstance(original_classifier[-1], nn.Linear)

        # 2. Act: Adapt the model
        feature_extractor = adapt_model_to_feature_extractor(model, model_type)

        # 3. Assert: Check that the classifier is now an Identity layer
        modified_classifier = getattr(feature_extractor, original_classifier_attr)
        assert isinstance(
            modified_classifier, nn.Identity
        ), f"The '{original_classifier_attr}' of the {model_type} model was not replaced."

    @pytest.mark.parametrize(
        "model_fn, model_type, expected_feature_dim",
        [
            (resnet18_3d, "resnet3d", 512),
            (densenet121_3d, "densenet3d", 1024),
            (vit_tiny_3d, "vit3d", 192),
        ],
    )
    def test_adapted_model_produces_feature_vector(
        self, model_fn, model_type, expected_feature_dim
    ):
        """
        Verifies that the adapted model outputs a 1D feature vector of the
        correct dimension, not classification logits.
        """
        # 1. Arrange
        dummy_input_size = (32, 64, 64) # Define the size for clarity
        if model_type == "vit3d":
            # For ViT, we must specify the volume size to match the dummy input
            model = model_fn(num_classes=18, volume_size=dummy_input_size)
        else:
            # Other models are not dependent on input size for initialization
            model = model_fn(num_classes=18)
        feature_extractor = adapt_model_to_feature_extractor(model, model_type)
        
        # Create a dummy input tensor
        # Using a smaller spatial size for faster test execution
        dummy_input = torch.randn(2, 1, *dummy_input_size)

        # 2. Act
        with torch.no_grad():
            output = feature_extractor(dummy_input)

        # 3. Assert
        assert isinstance(output, torch.Tensor)
        assert output.ndim == 2  # Should be a 2D tensor (batch_size, feature_dim)
        assert output.shape[0] == 2  # Batch size should be preserved
        assert output.shape[1] == expected_feature_dim

    def test_unsupported_model_type_raises_value_error(self):
        """
        Tests that a ValueError is raised when an unsupported model type is passed.
        """
        # 1. Arrange
        class UnsupportedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 2)
        
        model = UnsupportedModel()
        model_type = "unsupported_type"

        # 2. Act & Assert
        with pytest.raises(ValueError, match=f"Unsupported model type for feature extraction: {model_type}"):
            adapt_model_to_feature_extractor(model, model_type)
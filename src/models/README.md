# Models Module (`/src/models`)

This directory contains the neural network architectures for the 3D CT scan classification task. The models are self-contained, adapted for 3D volumetric data, and built directly on PyTorch, enabling features like gradient checkpointing for memory efficiency.

## Core Components

The module is organized into files based on the model architecture. Each file provides a consistent interface for creating models.

1.  `resnet3d.py`: A self-contained 3D adaptation of the ResNet architecture.
2.  `densenet3d.py`: A self-contained 3D adaptation of the DenseNet architecture.
3.  `vit3d.py`: A self-contained 3D adaptation of the Vision Transformer (ViT) architecture.

-----

### `resnet3d.py`

This file provides a 3D ResNet (Residual Network) implementation built from scratch for this project. It is configured for single-channel volumetric inputs and includes a custom classification head to match the project's requirements.

**Key Features:**

  * **`ResNet3D`**: The main class for the 3D ResNet model, implemented with standard PyTorch modules.

**Available Variants:**

  * `resnet18_3d`
  * `resnet34_3d`

-----

### `densenet3d.py`

This file provides a self-contained 3D implementation of the DenseNet (Densely Connected Convolutional Network) architecture. It is adapted for 3D from the official PyTorch 2D version and does not rely on external libraries for its core structure.

**Key Features:**

  * **`DenseNet3D`**: The main class for the 3D DenseNet model, implemented with standard PyTorch modules.

**Available Variants:**

  * Standard: `densenet121_3d`, `densenet161_3d`, `densenet169_3d`, `densenet201_3d`
  * Custom Compact: `densenet_small_3d`, `densenet_tiny_3d` for different memory constraints.

-----

### `vit3d.py`

This file implements a 3D Vision Transformer (ViT) from scratch. It adapts the powerful transformer architecture for volumetric classification tasks and is not dependent on MONAI.

**Key Features:**

  * **`VisionTransformer3D`**: The main ViT model class, which includes all components like patch embedding, multi-head attention, and transformer blocks built with PyTorch.

**Available Variants:**

  * `vit_tiny_3d`
  * `vit_small_3d`
  * `vit_base_3d`
  * `vit_large_3d`
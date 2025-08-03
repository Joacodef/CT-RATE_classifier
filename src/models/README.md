# Models Module (`/src/models`)

This directory contains the neural network architectures for the 3D CT scan classification task. The models are adapted for 3D volumetric data and are built as wrappers around the standardized implementations provided by the `monai` library.

## Core Components

The module is organized into files based on the model architecture. Each file provides a consistent interface for creating models while leveraging the robust `monai` backends.

1.  [**`resnet3d.py`**](https://www.google.com/search?q=%5Bhttps://www.google.com/search%3Fq%3D%2523resnet3dpy%5D\(https://www.google.com/search%3Fq%3D%2523resnet3dpy\)): A 3D adaptation of the ResNet architecture.
2.  [**`densenet3d.py`**](https://www.google.com/search?q=%5Bhttps://www.google.com/search%3Fq%3D%2523densenet3dpy%5D\(https://www.google.com/search%3Fq%3D%2523densenet3dpy\)): A 3D adaptation of the DenseNet architecture.
3.  [**`vit3d.py`**](https://www.google.com/search?q=%5Bhttps://www.google.com/search%3Fq%3D%2523vit3dpy%5D\(https://www.google.com/search%3Fq%3D%2523vit3dpy\)): A 3D adaptation of the Vision Transformer (ViT) architecture.

-----

### `resnet3d.py`

This file provides a 3D ResNet (Residual Network) by wrapping `monai.networks.nets.ResNet`. It is configured for single-channel volumetric inputs and includes a custom classification head to match the project's requirements.

**Key Features:**

  * **`ResNet3D`**: The base class that wraps the MONAI ResNet model and attaches a custom classification head.

**Available Variants:**

  * `resnet18_3d`
  * `resnet34_3d`

-----

### `densenet3d.py`

This file provides a 3D implementation of the DenseNet (Densely Connected Convolutional Network) architecture by wrapping `monai.networks.nets.DenseNet`. The wrapper ensures the model can be used seamlessly within the existing training framework.

**Key Features:**

  * **`DenseNet3D`**: The main class for the 3D DenseNet model, which integrates MONAI's DenseNet backbone with a custom classification head.

**Available Variants:**

  * Standard: `densenet121_3d`, `densenet161_3d`, `densenet169_3d`, `densenet201_3d`
  * Custom Compact: `densenet_small_3d`, `densenet_tiny_3d` for different memory constraints.

-----

### `vit3d.py`

This file implements a 3D Vision Transformer (ViT) by wrapping `monai.networks.nets.ViT`. It adapts the powerful transformer architecture for volumetric classification tasks.

**Key Features:**

  * **`VisionTransformer3D`**: The main ViT model class, which combines the MONAI ViT backbone with a custom classification head suitable for this project.

**Available Variants:**

  * `vit_tiny_3d`
  * `vit_small_3d`
  * `vit_base_3d`
  * `vit_large_3d`
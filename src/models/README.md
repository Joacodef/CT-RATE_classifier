# Models Module (`/src/models`)

This directory contains the neural network architectures and custom loss functions used for the 3D CT scan classification task. The models are adapted for 3D volumetric data and include memory-saving optimizations like gradient checkpointing to handle large CT volumes.

## Core Components

The module is organized into files based on the model architecture:

1.  [**`resnet3d.py`**](#resnet3dpy): A 3D adaptation of the classic ResNet architecture.
2.  [**`densenet3d.py`**](#densenet3dpy): A 3D adaptation of the DenseNet architecture.
3.  [**`vit3d.py`**](#vit3dpy): A 3D adaptation of the Vision Transformer (ViT) architecture.
4.  [**`losses.py`**](#lossespy): Contains custom loss functions for training.

---

### `resnet3d.py`

This file implements a 3D version of the ResNet (Residual Network) architecture, which is well-suited for deep learning tasks on volumetric data. The implementation includes memory optimization through gradient checkpointing.

**Key Features:**

* **`ResNet3D`**: The base class for the 3D ResNet model.
* **`BasicBlock3D`**: The 3D residual block used in the shallower ResNet variants.
* **Gradient Checkpointing**: The `use_checkpointing` flag can be set to trade compute for memory, which is essential for training on large 3D volumes.

**Available Variants:**

* `resnet18_3d`
* `resnet34_3d`

---

### `densenet3d.py`

This file provides a 3D implementation of the DenseNet (Densely Connected Convolutional Network) architecture. DenseNets connect each layer to every other layer in a feed-forward fashion, which encourages feature reuse and strengthens feature propagation.

**Key Features:**

* **`DenseNet3D`**: The main class for the 3D DenseNet model.
* **`_DenseLayer` & `_DenseBlock`**: The core components that define the dense connectivity pattern.
* **Memory Efficiency**: The implementation includes two memory-saving options:
    1.  `memory_efficient`: A highly optimized implementation within the dense layers.
    2.  `use_checkpointing`: Applies gradient checkpointing at the block level for further memory reduction.

**Available Variants:**

* Standard: `densenet121_3d`, `densenet161_3d`, `densenet169_3d`, `densenet201_3d`
* Custom Compact: `densenet_small_3d`, `densenet_tiny_3d` for more memory-constrained environments.

---

### `vit3d.py`

This file implements a 3D Vision Transformer (ViT), adapting the transformer architecture, originally designed for natural language processing, to volumetric image classification. It works by dividing a 3D volume into non-overlapping patches and treating them as a sequence of tokens.

**Key Features:**

* **`VisionTransformer3D`**: The main ViT model class.
* **`PatchEmbed3D`**: A layer that converts 3D patches into linear embeddings.
* **`MultiHeadAttention3D`**: The self-attention mechanism adapted for 3D patch sequences.
* **Gradient Checkpointing**: Includes the `use_checkpointing` flag to reduce memory usage during training.

**Available Variants:**

* `vit_tiny_3d`
* `vit_small_3d`
* `vit_base_3d`
* `vit_large_3d`

---

### `losses.py`

This module contains custom loss functions used during training.

**Key Components:**

* **`FocalLoss`**: An implementation of the Focal Loss function. It is a modification of the standard binary cross-entropy loss that helps address class imbalance by down-weighting the loss assigned to well-classified examples, thereby focusing the model's training on "harder" examples.
# Utilities Module (`/src/utils`)

This directory contains general-purpose utility functions that provide support across the entire project. These helpers are not specific to modeling, data handling, or training but offer functionalities like logging configuration and performance optimization.

## Core Components

1.  [**`logging_config.py`**](https://www.google.com/search?q=%23logging_configpy): A module to standardize logging configuration.
2.  [**`torch_utils.py`**](https://www.google.com/search?q=%23torch_utilspy): A module to configure PyTorch for performance.

-----

### `logging_config.py`

This file provides a centralized function to set up consistent logging for any script in the application.

**Key Functions:**

  * **`setup_logging`**: Initializes the root logger to output messages of `INFO` level and above. It configures two handlers:
      * `FileHandler`: Writes all log messages to a specified file (e.g., `ct_3d_training.log`).
      * `StreamHandler`: Prints all log messages to the console.

### `torch_utils.py`

This file contains a utility function to apply several performance-related settings to the PyTorch backend.

**Key Functions:**

  * **`setup_torch_optimizations`**: This function configures PyTorch settings for hardware like NVIDIA Ampere architecture and newer. It enables:
      * `torch.backends.cudnn.benchmark = True`: Allows cuDNN to find the best algorithm for the hardware.
      * `torch.backends.cuda.matmul.allow_tf32 = True`: Enables the use of TensorFloat-32 cores for matrix multiplication.
      * `torch.backends.cudnn.allow_tf32 = True`: Enables the use of TensorFloat-32 cores for cuDNN convolutions.
      * It also sets the number of threads used by PyTorch for CPU operations.
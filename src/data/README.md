# Data Module (`/src/data`)

This directory is responsible for all data handling, from locating raw files to applying complex transformations and managing a multi-level caching system. The design separates data loading, preprocessing, and augmentation into distinct, composable steps.

The core principle is a two-stage caching pipeline (disk and RAM) managed by MONAI, where only the label-agnostic, preprocessed image tensors are cached. Labels and on-the-fly augmentations are applied after data is retrieved from the cache.

## Core Components

The data module consists of four key files:

1.  [**`dataset.py`**](https://www.google.com/search?q=%23datasetpy): Defines the custom PyTorch `Dataset` classes that form the building blocks of the data pipeline.
2.  [**`transforms.py`**](https://www.google.com/search?q=%23transformspy): Centralizes the MONAI transformation pipelines for consistent preprocessing.
3.  [**`cache_utils.py`**](https://www.google.com/search?q=%23cache_utilspy): Contains the logic for the deterministic caching system, including hashing and directory management.
4.  [**`utils.py`**](https://www.google.com/search?q=%23utilspy): Provides utility functions for data-related tasks, such as resolving file paths.

-----

### `dataset.py`

This file implements the `Dataset` classes that work together to create a flexible and cache-friendly data pipeline.

  * **`CTMetadataDataset`**: This is the first step in the pipeline. Its role is to take a dataframe and an index, and return the absolute file path to the corresponding CT volume and its volume name. It is label-agnostic to serve as a clean input to the caching mechanism.
  * **`LabelAttacherDataset`**: This wrapper is used after the caching and preprocessing stages. It takes a source dataset that yields processed images (e.g., from a `CacheDataset`) and dynamically attaches the correct pathology labels from a pandas DataFrame based on the index. This design ensures that labels are not stored in the cache.
  * **`ApplyTransforms`**: A generic wrapper that applies a set of transformations (typically augmentations) to the data provided by a source dataset. This allows for on-the-fly augmentations to be performed after the base data has been retrieved from the cache.

### `transforms.py`

This module defines the transformation pipeline used for consistent data preprocessing.

  * **`get_preprocessing_transforms`**: This is the primary function, which constructs and returns a MONAI `Compose` object containing the entire preprocessing pipeline. This pipeline is responsible for:
      * Loading NIfTI images and their metadata (`LoadImaged`).
      * Reorienting volumes to a standard orientation (`Orientationd`).
      * Resampling volumes to a target voxel spacing (`Spacingd`).
      * Clipping and normalizing voxel intensity values (`ScaleIntensityRanged`).
      * Resizing volumes to a fixed target shape (`Resized`).
  * **`KeyCleanerD`**: A custom transform that runs at the end of the preprocessing pipeline to remove all intermediate data from the dictionary, keeping only the final image tensor and its metadata. This ensures the cached items are minimal and self-contained.

### `cache_utils.py`

This file contains the logic for the caching system. It ensures that any change in preprocessing parameters results in a new, unique cache, which supports reproducibility.

  * **`get_or_create_cache_subdirectory`**: This function generates a unique hash based on the parameters of the entire preprocessing transform pipeline. This hash is used as the directory name for the `PersistentDataset`. If the directory already exists, the cached data is reused; otherwise, a new one is created.
  * **`get_transform_params`**: A utility that uses Python's `inspect` module to recursively parse a MONAI `Compose` object and all its constituent transforms, extracting every public parameter into a serializable dictionary.
  * **`deterministic_hash`**: A hashing function with two modes:
    1.  When given a data item, it hashes only the `volume_name` to create a stable filename for that item's cache file.
    2.  When given a configuration object (like the transform parameters), it creates a stable JSON string and hashes it to produce the cache directory hash.
  * **`worker_init_fn`**: A helper for `DataLoader` that patches `torch.load` in each worker process. This is required to deserialize complex MONAI objects (like `MetaTensor`) that are stored in the cache.

### `utils.py`

This file contains general-purpose utility functions related to data handling.

  * **`get_dynamic_image_path`**: A helper function that constructs the full, absolute path to a CT volume based on its name and the directory structure (`nested` or `flat`) specified in the configuration. This abstracts the file system layout from the dataset classes.# Data Module (`/src/data`)

This directory is responsible for all data handling, from locating raw files to applying complex transformations and managing a multi-level caching system. The design separates data loading, preprocessing, and augmentation into distinct, composable steps.

The core principle is a two-stage caching pipeline (disk and RAM) managed by MONAI, where only the label-agnostic, preprocessed image tensors are cached. Labels and on-the-fly augmentations are applied after data is retrieved from the cache.

## Core Components

The data module consists of four key files:

1.  [**`dataset.py`**](https://www.google.com/search?q=%23datasetpy): Defines the custom PyTorch `Dataset` classes that form the building blocks of the data pipeline.
2.  [**`transforms.py`**](https://www.google.com/search?q=%23transformspy): Centralizes the MONAI transformation pipelines for consistent preprocessing.
3.  [**`cache_utils.py`**](https://www.google.com/search?q=%23cache_utilspy): Contains the logic for the deterministic caching system, including hashing and directory management.
4.  [**`utils.py`**](https://www.google.com/search?q=%23utilspy): Provides utility functions for data-related tasks, such as resolving file paths.

-----

### `dataset.py`

This file implements the `Dataset` classes that work together to create a flexible and cache-friendly data pipeline.

  * **`CTMetadataDataset`**: This is the first step in the pipeline. Its role is to take a dataframe and an index, and return the absolute file path to the corresponding CT volume and its volume name. It is label-agnostic to serve as a clean input to the caching mechanism.
  * **`LabelAttacherDataset`**: This wrapper is used after the caching and preprocessing stages. It takes a source dataset that yields processed images (e.g., from a `CacheDataset`) and dynamically attaches the correct pathology labels from a pandas DataFrame based on the index. This design ensures that labels are not stored in the cache.
  * **`ApplyTransforms`**: A generic wrapper that applies a set of transformations (typically augmentations) to the data provided by a source dataset. This allows for on-the-fly augmentations to be performed after the base data has been retrieved from the cache.

### `transforms.py`

This module defines the transformation pipeline used for consistent data preprocessing.

  * **`get_preprocessing_transforms`**: This is the primary function, which constructs and returns a MONAI `Compose` object containing the entire preprocessing pipeline. This pipeline is responsible for:
      * Loading NIfTI images and their metadata (`LoadImaged`).
      * Reorienting volumes to a standard orientation (`Orientationd`).
      * Resampling volumes to a target voxel spacing (`Spacingd`).
      * Clipping and normalizing voxel intensity values (`ScaleIntensityRanged`).
      * Resizing volumes to a fixed target shape (`Resized`).
  * **`KeyCleanerD`**: A custom transform that runs at the end of the preprocessing pipeline to remove all intermediate data from the dictionary, keeping only the final image tensor and its metadata. This ensures the cached items are minimal and self-contained.

### `cache_utils.py`

This file contains the logic for the caching system. It ensures that any change in preprocessing parameters results in a new, unique cache, which supports reproducibility.

  * **`get_or_create_cache_subdirectory`**: This function generates a unique hash based on the parameters of the entire preprocessing transform pipeline. This hash is used as the directory name for the `PersistentDataset`. If the directory already exists, the cached data is reused; otherwise, a new one is created.
  * **`get_transform_params`**: A utility that uses Python's `inspect` module to recursively parse a MONAI `Compose` object and all its constituent transforms, extracting every public parameter into a serializable dictionary.
  * **`deterministic_hash`**: A hashing function with two modes:
    1.  When given a data item, it hashes only the `volume_name` to create a stable filename for that item's cache file.
    2.  When given a configuration object (like the transform parameters), it creates a stable JSON string and hashes it to produce the cache directory hash.
  * **`worker_init_fn`**: A helper for `DataLoader` that patches `torch.load` in each worker process. This is required to deserialize complex MONAI objects (like `MetaTensor`) that are stored in the cache.

### `utils.py`

This file contains general-purpose utility functions related to data handling.

  * **`get_dynamic_image_path`**: A helper function that constructs the full, absolute path to a CT volume based on its name and the directory structure (`nested` or `flat`) specified in the configuration. This abstracts the file system layout from the dataset classes.
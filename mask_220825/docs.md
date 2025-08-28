## Documentation: 3D Segmentation Module with Coarse-to-Fine Architecture

### Table of Contents
1.  [**Introduction and General Concept**](###Introduction and General Concept)
    *   [1.1. Module Purpose](####1.1. Module Purpose)
    *   [1.2. Key Idea: Coarse-to-Fine Architecture](####1.2. Key Idea: Coarse-to-Fine Architecture)
2.  [**Architecture and Workflow Pipeline**](###Architecture and Workflow Pipeline)
    *   [2.1. General Pipeline Diagram](####21-general-pipeline-diagram)
    *   [2.2. Detailed Training Stage Pipeline](####22-detailed-training-stage-pipeline)
3.  [**Module Descriptions**](###3-module-descriptions)
    *   [3.1. `data_units.py`: Data Preparation and Loading](####31-data_unitspy-data-preparation-and-loading)
    *   [3.2. `models.py`: Neural Network Architectures](####32-modelspy-neural-network-architectures)
    *   [3.3. `inference.py`: Model Application Logic](####33-inferencepy-model-application-logic)
    *   [3.4. `training_pipeline3.py`: Training Process Orchestrator](####34-training_pipeline3py-training-process-orchestrator)
    *   [3.5. `augmentations.py`: Data Augmentation](####35-augmentationspy-data-augmentation)
4.  [**Practical Guide**](###4-practical-guide)
    *   [4.1. Step 1: Preprocessing Raw Data](####41-step-1-preprocessing-raw-data)
    *   [4.2. Step 2: Running the Training Process](####42-step-2-running-the-training-process)
    *   [4.3. Step 3: Results and Checkpoints](####43-step-3-results-and-checkpoints)
5.  [**Key Configuration Parameters**](####5-key-configuration-parameters)

---

### 1. Introduction and General Concept

#### 1.1. Module Purpose

This module is designed to solve the task of semantic segmentation on 3D data.

The main complexity of the task lies in the wide variety of input data:
*   **Different MRI modalities** (T1, T2, FLAIR, DWI, etc.).
*   **Varying spatial resolutions** and image sizes.
*   The presence of artifacts and noise.

The module is designed to create a single, robust (adaptive) model capable of effectively handling this diversity.

#### 1.2. Key Idea: Coarse-to-Fine Architecture

At the core of the module is a two-stage architecture that mimics the human approach to image analysis: first a general overview, then a focus on details.

1.  **Coarse Stage:**
    *   A lightweight and fast model (`CoarseUNet_Medium`) analyzes the **entire, but downscaled**, 3D image.
    *   Its task is to create a **global context map**: an approximate, low-detail mask of the entire object. This map provides an understanding of the object's general shape and location in space.

2.  **Fine Stage:**
    *   The main, more powerful model (`MultiTask_FineUNet_MoE`) works with **high-resolution patches**.
    *   It receives an **"information packet"** as input, consisting of:
        1.  A patch of the original image (local details).
        2.  The corresponding patch from the "global context map" (anatomical location).
        3.  (Optional) A patch from coordinate "heatmaps" (absolute mathematical location).
    *   Using all this information, the model makes a final, high-precision prediction for each patch.

This approach allows the `Fine` model to make more informed decisions, as it "knows" which part of the larger object it is currently analyzing.

### 2. Architecture and Workflow Pipeline

#### 2.1. General Pipeline Diagram

```
[Raw .nii.gz data] -> [Preprocessing (data_units.py)] -> [Single H5 file] -> [Training (training_pipeline3.py)] -> [Trained model (.pth)]
```

#### 2.2. Detailed Training Stage Pipeline

During each training iteration (`AdvancedTrainer.train_epoch`), the following cycle occurs:

1.  **Loading:** The `FullImageDataset` loads **one complete 3D scan** into memory.
2.  **Coarse Model Pass:**
    *   The full scan is downscaled to `coarse_input_size`.
    *   `CoarseUNet_Medium` makes a **single** prediction, creating the "global map."
    *   `loss_coarse` is calculated.
3.  **Fine Model Pass (in a loop):**
    *   `patches_per_volume` random patches are extracted from the full scan and the full "global map" in a loop.
    *   For each patch, a 2-channel or 5-channel input is assembled (image, map, optionally 3 coordinate channels).
    *   `MultiTask_FineUNet_MoE` makes a prediction for each patch.
    *   The losses `loss_fine_seg` and `loss_fine_cls` are accumulated and averaged.
4.  **Optimization Step:** All losses are summed, and the optimizer performs a **single** backpropagation step, updating the weights of both models simultaneously.

### 3. Module Descriptions

#### 3.1. `data_units.py`: Data Preparation and Loading

*   **Main Task:** Centralized management of all data operations.
*   **Key Components:**
    *   `create_prepared_synthstrip_h5`: A script for **one-time preprocessing**. It scans a directory with raw `.nii.gz` files, resamples each scan to an isotropic resolution (`target_voxel_spacing`) and a specified size (`target_shape` from the config), and then saves everything into a single HDF5 file optimized for reading.
    *   `FullImageDataset`: A specialized PyTorch `Dataset` used in the `AdvancedTrainer`. Its unique feature is that it loads **entire 3D volumes** instead of patches, which is necessary for the Coarse-to-Fine approach.
    *   `pad_or_crop_to_shape`, `normalize_mri`: Helper functions for standardizing the size and intensity of voxels.

#### 3.2. `models.py`: Neural Network Architectures

*   **`CoarseUNet_Medium`:** A simple, lightweight 3D U-Net. Designed for fast generation of the "global map" on downscaled images.
*   **`MultiTask_FineUNet_MoE`:** The "heart" of the project. A powerful 3D U-Net with the following features:
    *   **Multi-Task:** It has two "heads" and simultaneously predicts a segmentation mask and the modality class. This acts as a strong regularizer, forcing the model to learn more general and useful features.
    *   **Mixture of Experts (MoE):** Instead of standard convolutional blocks, it uses an `MoELayer`, where multiple "experts" (in this case, `ResBlock3D`) work in parallel, and a special `gating_network` dynamically decides which expert to "trust" with processing the current data. This significantly increases the model's capacity without a proportional increase in computational cost.
    *   **Multi-channel Input:** Designed to accept a 2-channel or 5-channel input, allowing it to receive all contextual information.

#### 3.3. `inference.py`: Model Application Logic

*   **`UnifiedPatchedModel`:** A key component for the **validation** and **real-world application** stages. This is a wrapper model that encapsulates both trained models (`Coarse` and `Fine`) and implements **adaptive inference logic**:
    *   If the input image is larger than the patch size, it applies a "sliding window" technique, slicing the image into overlapping patches and carefully stitching the results together.
    *   If the input image is smaller than or equal to the patch size, it pads the image to the patch size, makes a prediction in a single pass, and crops the result.
    *   **Important:** This model can also generate and use coordinate maps if `use_coord_maps=True`, ensuring full consistency between the training and validation logic.

#### 3.4. `training_pipeline3.py`: Training Process Orchestrator

*   **`AdvancedTrainer`:** The main class that manages the entire process. Its responsibilities:
    *   Accepts a configuration dictionary as input.
    *   Splits the data into training and validation sets.
    *   Initializes the models, optimizer, scheduler, and loss functions.
    *   Implements the main `train` loop, which alternately calls training (`train_epoch`) and validation (`validate_epoch`).
    *   Updates the learning rate and saves the best checkpoints (`save_checkpoint`) based on validation metrics (Dice score).
    *   Logs progress to the console.

#### 3.5. `augmentations.py`: Data Augmentation

*   Contains the `get_augmentations_transform` function, which creates a 3D augmentation pipeline using the `torchio` library. It includes random flipping, affine transformations (scaling, rotation, translation), elastic deformations, noise addition, blurring, and gamma adjustments. Augmentations are applied "on the fly" to the loaded data to improve the model's generalization ability.

### 4. Practical Guide

#### 4.1. Step 1: Preprocessing Raw Data

Before starting the training, you need to prepare the H5 file.
1.  Open the `data_units.py` file.
2.  Configure the paths and parameters in the `SYNTHSTRIP_PROCESSING_CONFIG` dictionary. **Important:** The current training and validation pipeline is adapted to work with different sizes.
3.  Run the `get_prepared_synthstrip_dataset` function (or `create_prepared_synthstrip_h5` directly), specifying the path to the raw data and the path to save the H5 file.
4.  Wait for the process to complete.

#### 4.2. Step 2: Running the Training Process

1.  Open the `training_pipeline3.py` file.
2.  Find the `main_advanced_training` function.
3.  In the `config` dictionary, specify the correct path to your H5 file (`h5_path`).
4.  Configure the other training parameters (see Section 5).
5.  Run the script: `python training_pipeline3.py`.

#### 4.3. Step 3: Results and Checkpoints

*   Training progress will be printed to the console, including training losses and validation metrics (Dice, loss) after each epoch.
*   Checkpoints will be saved to the directory specified in `save_dir`.
*   The `best_model.pth` file contains the weights of the model that achieved the best Dice score on the validation set.

### 5. Key Configuration Parameters

The main parameters are configured in the `config` dictionary in `main_advanced_training`:
*   `h5_path`: **(Required)** The path to your H5 file.
*   `save_dir`: The directory for saving checkpoints.
*   `device`: `"cuda"` or `"cpu"`.
*   `coarse_input_size`: The size to which the full image is downscaled for the `Coarse` model.
*   `fine_patch_size`: The size of the patches on which the `Fine` model is trained.
*   `use_coord_maps`: `True` or `False`. Enables/disables the use of coordinate "heatmaps." **Important:** The model will be compiled with the corresponding number of input channels.
*   `patches_per_volume`: How many random patches to extract from one full 3D scan in a single training step.
*   `num_epochs`: The total number of training epochs.
*   `learning_rate`: The initial learning rate.
*   `validation_split`: The fraction of data to be set aside for validation (e.g., `0.15` = 15%).
*   `model_checkpoint_path`: The path to a saved model (models and additional info).
*   `load_optim`: `True` or `False`. Enables/disables loading the saved state of the optimizer.
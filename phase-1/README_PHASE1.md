# PHASE-1
## Overview
- This phase focused on preparing and normalizing raw DICOM images from the LIDC-IDRI dataset for deep learning. 
- The goal was to extract uniform 3D patches of lung nodules along with their clinical metadata to feed into a multimodal neural network.
- Tools and Libraries Used
    - SimpleITK & pydicom: Used for ingesting raw CT scans and resampling volumes to a uniform 1.0 mm isotropic spacing. This ensures that the physical dimensions of the nodules are consistent across all patients regardless of the original scanner settings.
    - pylidc: An object-relational mapping library specifically designed for the LIDC-IDRI dataset. It was used to aggregate consensus nodule centroids, calculate average malignancy ratings across multiple radiologists, and extract semantic scores (subtlety, sphericity, margin, spiculation, and texture).
    - NumPy & Pandas: Used for applying lung Hounsfield Unit (HU) windowing ([-1000, 400] HU) to focus purely on lung tissue and normalizing voxel intensities to a [0, 1] range. Pandas was utilized to generate and store the final tracking manifest.csv.
    - PyTorch (torch): Used to convert the extracted $64 \times 64 \times 64$ numpy arrays into 3D tensors and save them directly as .pt files. This drastically speeds up I/O operations during the training phase.
## Process Workflow
    - DICOM Parsing & Resampling: Loaded the raw 3D DICOM CT scans and resampled them.
    - Windowing & Normalization: Clipped the radiodensity values to the lung window to discard irrelevant tissues (like bone and outside air) and normalized the scale.
    - Annotation Aggregation: Queried the SQLite database via pylidc to find nodule annotations, calculated consensus centroids, and classified the malignancy score.
    - Patch Extraction: Cropped a $64 \times 64 \times 64$ voxel 3D subvolume centered on each nodule's centroid.
    - Export: Saved the patches as PyTorch tensors and logged all metadata into the manifest.
## Challenges and Solutions
    - Data Imbalance: The LIDC dataset is highly skewed towards non-malignant nodules. To mitigate this, oversampling techniques (duplicating minority class samples) and data augmentation strategies (rotations, flips, and intensity shifts) were applied during training to prevent the model from defaulting to the majority class.  
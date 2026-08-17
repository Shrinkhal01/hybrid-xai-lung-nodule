# Phase 1: Medical CT Preprocessing & 3D Patch Extraction

This directory contains the pipeline for ingesting, resampling, normalizing, and extracting 3D nodule patches from the LIDC-IDRI dataset. The generated patches and consensus malignancy labels serve as the foundation for the subsequent deep learning and radiomics models in Phase 2.

---

## 📂 File Structure

| File | Type | Description |
| :--- | :--- | :--- |
| [`preprocess.py`](file:///Users/shrinkhals/Shrinkhal-Github/Projects/hybrid-xai-lung-nodule/phase-1/preprocess.py) | Python Script | Command-line script to run batch preprocessing locally or on Colab. |
| [`test_preprocess.py`](file:///Users/shrinkhals/Shrinkhal-Github/Projects/hybrid-xai-lung-nodule/phase-1/test_preprocess.py) | Python Script | Unit tests verifying windowing, padding, coordinates, and saving. |
| [`phase1preprocess.ipynb`](file:///Users/shrinkhals/Shrinkhal-Github/Projects/hybrid-xai-lung-nodule/phase-1/phase1preprocess.ipynb) | Jupyter Notebook | Interactive version for Google Colab with 2D/3D orthogonal slice visualization. |

---

## 🛠️ Library Stack & Rationale

We utilize a medical imaging stack to ensure data integrity during spatial and intensity transformation:

1. **`SimpleITK (sitk)`**: 
   - *Why?* Standard image readers fail to account for spatial attributes (origin, pixel spacing, direction cosines). SimpleITK maintains the physical coordinate space and resamples arbitrary voxel grids (e.g., $0.68\,\text{mm} \times 0.68\,\text{mm} \times 1.25\,\text{mm}$) into isotropic spaces ($1.0\,\text{mm} \times 1.0\,\text{mm} \times 1.0\,\text{mm}$) using linear/B-spline interpolation filters.
2. **`pylidc`**: 
   - *Why?* LIDC-IDRI annotations are stored in custom XML files. `pylidc` queries these annotations, clusters annotations from multiple radiologists referring to the same nodule, and provides the consensus malignancy rating and physical coordinate centroids.
3. **`pydicom`**: 
   - *Why?* Handles low-level DICOM header reading and metadata parsing.
4. **`PyTorch (torch)`**: 
   - *Why?* Patches are serialized directly as PyTorch 3D tensors (`.pt`) so they can be loaded directly into neural networks without conversion overhead.
5. **`h5py`**: 
   - *Why?* Allows storing hundreds of 3D volume matrices in a single compressed HDF5 binary archive for fast disk I/O.

---

## 🔍 Preprocessing Pipeline Details

```
Raw DICOM Scan -> SimpleITK Loader -> Isotropic Resampler -> HU Windowing & Normalization -> Centroid Mapping & Crop -> Save Patch
```

### 1. DICOM Volume Ingestion
- `load_dicom_volume_sitk(dicom_dir)`: Resolves primary CT series folder and reads the 3D volume with correct orientation and slice sorting.

### 2. Isotropic Resampling
- `resample_volume_isotropic(image, target_spacing=(1.0, 1.0, 1.0))`: Resamples voxels to $(1\,\text{mm}, 1\,\text{mm}, 1\,\text{mm})$ so that bounding boxes and 3D spatial features are uniform across all patients.

### 3. Lung HU Windowing & Normalization
- `apply_lung_window_and_normalize(volume_np, min_hu=-1000.0, max_hu=400.0)`: 
  - Clips CT voxel intensities to a standard lung parenchyma window of $[-1000, 400]$ Hounsfield Units.
  - Normalizes intensity values linearly to $[0.0, 1.0]$ using:
    $$\text{Normalized} = \frac{\text{clipped\_HU} + 1000.0}{1400.0}$$

### 4. Annotation Clustering & Consensus Malignancy
- `parse_scan_nodules(scan, resampled_sitk_img, original_sitk_img)`:
  - Groups annotations of 1-4 radiologists for each physical nodule.
  - Computes the consensus malignancy score (average radiologist rating, 1-5 scale).
  - Assigns binary class targets:
    - **`0` (Benign)**: Consensus malignancy $< 3.0$
    - **`1` (Malignant)**: Consensus malignancy $> 3.0$
    - **`-1` (Indeterminate)**: Consensus malignancy $= 3.0$
  - Maps physical annotation coordinates to the resampled 3D voxel space.

### 5. 3D Patch Extraction
- `extract_3d_patch(volume_np, centroid_zyx, patch_size=(64, 64, 64), pad_value=0.0)`: Extracts a sub-volume centered at the nodule centroid. If the nodule is near the volume boundary, constant background padding (`0.0` or equivalent air HU) is applied.

---

## 🚀 Execution & Configuration

### pylidc Database Setup
To query LIDC-IDRI annotations, `pylidc` requires a configuration file pointing to your dataset root folder directory.

1. A file named `~/.pylidcrc` is automatically created with the following format:
   ```ini
   [pylidc]
   path = /path/to/raw_data
   warn = False
   ```
2. On first execution, `pylidc` automatically compiles an SQLite database compiling all DICOM directories and annotation XML files.

### Running Batch Preprocessing
```bash
python preprocess.py \
    --raw_data_dir "/content/drive/MyDrive/Lung_Nodule_Project/raw_data" \
    --output_dir "/content/drive/MyDrive/Lung_Nodule_Project/processed_patches" \
    --save_format pt
```

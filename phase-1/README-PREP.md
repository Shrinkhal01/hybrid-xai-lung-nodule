# Phase 1: Full-Scale Data Preprocessing & 3D Patch Extraction

## 📌 Overview
Phase 1 forms the foundational data pipeline for the Hybrid Explainable AI Framework for Accurate Lung Nodule Detection. It processes the massive 133 GB LIDC-IDRI Dataset, converting raw clinical DICOM CT scans into standardized, AI-ready 3D PyTorch tensors and radiologist-quantified tabular metadata.

Because medical imaging data is highly variable (different slice thicknesses, pixel spacings, and radiologist opinions), this pipeline ensures spatial uniformity, consensus-based ground truth labeling, and memory-efficient batching for downstream model training.

---

## 🛠️ Technologies Used & Why

| Technology | Purpose in Pipeline | Why It Was Chosen |
| :--- | :--- | :--- |
| **`tcia_utils`** | Dataset Acquisition | Bypasses local storage limits by streaming the 1,010 patient CT series (133 GB) directly from the NBIA REST API to Google Drive via Colab. |
| **`SimpleITK`** | 3D Volume Processing | Essential for handling complex spatial metadata (origin, direction, spacing) inherent in DICOM files, which standard image libraries (like OpenCV or PIL) cannot process. |
| **`pylidc`** | Annotation Parsing | The LIDC-IDRI dataset contains annotations from 4 different radiologists. `pylidc` uses clustering algorithms to merge these into a single consensus nodule and extracts critical morphological traits (spiculation, texture, etc.). |
| **`PyTorch`** | Patch Serialization | Converts the cropped 3D NumPy arrays into mathematical tensors (`.pt`) for seamless, high-speed loading during Phase 4 GPU training. |
| **`Pandas`** | Manifest Generation | Compiles all nodule metadata, physical world coordinates, class labels, and file paths into a single structured registry (`manifest.csv`). |

---

## ⚙️ How It Works (The Pipeline Architecture)

The preprocessing executes sequentially across all 1,010 subjects through the following steps:

1. **Directory Restructuring**
   Cloud downloaders retrieve DICOMs using raw Series UIDs (e.g., `1.3.6.1...`). The pipeline first maps and restructures these folders into the official standard expected by the `pylidc` library (`LIDC-IDRI-XXXX`).

2. **Isotropic Resampling**
   CT scans come from different hospital machines with varying voxel sizes (e.g., $0.6 \times 0.6 \times 1.5\,\text{mm}$). The pipeline uses SimpleITK to mathematically resample every 3D volume to a universal isotropic spacing of $1.0 \times 1.0 \times 1.0\,\text{mm}$. This ensures the CNN learns physical shapes consistently, regardless of the original scanner's settings.

3. **Hounsfield Unit (HU) Windowing**
   The raw pixel values in CT scans represent radiodensity (Hounsfield Units). The script applies a Lung Window `[-1000, 400]` HU to filter out irrelevant tissues (like bone and background air) and highlights lung parenchyma and pulmonary nodules. The values are then min-max normalized to a `[0, 1]` scale.

4. **Consensus Clustering & Ground Truth**
   For each scan, `pylidc` clusters the annotations from the 4 radiologists:
   - **Target Labeling**: If the mean malignancy score is $<3.0$, it is labeled **Benign (0)**. If $>3.0$, it is labeled **Malignant (1)**.
   - **Tabular Feature Extraction**: Morphological assessments (subtlety, sphericity, margin, spiculation, texture) are averaged across radiologists to feed the Tabular MLP branch of the hybrid model.

5. **3D Patch Extraction**
   Using the continuous physical world coordinates transformed to voxel indices, the pipeline crops a strict $64 \times 64 \times 64$ voxel 3D patch centered precisely on the nodule. Padding is applied if the nodule is near the lung border.

6. **Serialization**
   Each 3D patch is saved as an individual PyTorch tensor (`.pt`) to prevent RAM overload during model training. The metadata is written to `manifest.csv`.

---

## 📂 Output Directory Structure

Upon successful completion, the output directory will be structured as follows:

```text
processed_patches/
│
├── patches/
│   ├── LIDC-IDRI-0001_nodule_000.pt
│   ├── LIDC-IDRI-0002_nodule_000.pt
│   ├── LIDC-IDRI-0002_nodule_001.pt
│   └── ... (thousands of 3D tensors)
│
└── manifest.csv   <-- The master index used by the PyTorch Dataset class
```

---

## 🚀 Execution Notes

- **Runtime Environment**: Designed to run on Google Colab (CPU/GPU) with Google Drive mounted.
- **Time Complexity**: Processing the full 133 GB cohort takes approximately 4 to 7 hours depending on cloud I/O speeds.
- **Safety Mechanism**: The pipeline holds the data dictionary in memory to maximize speed. If the environment times out at the very end of the run, a rescue block is provided in the notebook to remount the drive and flush the `manifest_df` from Python's live memory to the CSV file without recalculating the dataset.

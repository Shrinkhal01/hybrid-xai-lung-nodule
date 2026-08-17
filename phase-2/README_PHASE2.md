# Phase 2: Hybrid Deep Learning + Radiomics Model Architecture

This directory implements the multi-modal machine learning pipeline. It combines deep spatial features from 3D CT volumes with handcrafted radiomic features and radiologist clinical semantic attributes to predict lung nodule malignancy.

---

## 📂 File Structure

| File | Type | Description |
| :--- | :--- | :--- |
| [`radiomics_extractor.py`](file:///Users/shrinkhals/Shrinkhal-Github/Projects/hybrid-xai-lung-nodule/phase-1/phase-2/radiomics_extractor.py) | Python Module | Extracts shape, first-order intensity, and slice GLCM texture features from 3D patches. |
| [`model.py`](file:///Users/shrinkhals/Shrinkhal-Github/Projects/hybrid-xai-lung-nodule/phase-1/phase-2/model.py) | Python Module | Defines the `HybridMalignancyNet` architecture combining 3D CNN, MLP, and Gated Multimodal Fusion. |
| [`train.py`](file:///Users/shrinkhals/Shrinkhal-Github/Projects/hybrid-xai-lung-nodule/phase-1/phase-2/train.py) | Python Script | Coordinates training using 5-Fold patient-grouped validation splits. |
| [`test_phase2.py`](file:///Users/shrinkhals/Shrinkhal-Github/Projects/hybrid-xai-lung-nodule/phase-1/phase-2/test_phase2.py) | Python Script | Unit tests verifying model output shapes, mask estimations, and GLCM metrics. |
| [`phase2_model.ipynb`](file:///Users/shrinkhals/Shrinkhal-Github/Projects/hybrid-xai-lung-nodule/phase-1/phase-2/phase2_model.ipynb) | Jupyter Notebook | Interactive Colab-ready notebook with feature caching, training loops, and validation metrics. |

---

## 🛠️ Library Stack & Rationale

1. **`scipy.ndimage`**:
   - *Why?* Used to perform 3D morphology and label analysis on voxel arrays. It enables automatic identification and isolation of the target nodule at the center of the patch by extracting the connected component closest to the spatial center `(32, 32, 32)`.
2. **`numpy`**:
   - *Why?* Serves as the mathematical engine for gray-level co-occurrence matrix (GLCM) probability mapping, gray-level discretization (binning), and 3D distance constraints.
3. **`scikit-learn`**:
   - *Why?* Essential for diagnostic validation metrics (ROC-AUC, sensitivity, specificity, accuracy) and implementing `GroupKFold` patient-grouped splitting.
4. **`PyTorch (torch)`**:
   - *Why?* Powers the deep neural network blocks (3D convolutions, batch normalization, MLPs, gated fusion, cross-entropy, and mean-squared-error backward passes).

---

## 🔍 Module Details

### 1. Tabular Radiomics Feature Extractor (`radiomics_extractor.py`)
Because the extracted 3D CT patch is centered exactly on the nodule, we dynamically estimate the binary nodule mask using an intensity threshold ($>0.35$ normalized HU for soft tissue) and a spatial distance constraint ($r \leq 20.0\,\text{mm}$ from the patch center).
It extracts **16 radiomic features**:
* **First-Order Intensity (8d)**: Mean, standard deviation, energy, entropy, skewness, kurtosis, range, uniformity.
* **3D Shape (4d)**: Volume (voxel count), surface area, sphericity, compactness.
* **Texture (4d)**: GLCM contrast, correlation, homogeneity, and energy computed on the nodule's central axial slice.

### 2. Multi-Modal Network Architecture (`model.py`)
```
[3D CT Image]   ---> [Spatial 3D CNN Encoder]          \
                                                        ===> [Gated Multimodal Fusion] ---> [Predictor Head]
[Tabular Vector] ---> [Radiomics/Concept MLP Encoder]   /
```
* **Spatial 3D CNN Encoder**: Passes the 3D CT patch through four layers of 3D Conv + BatchNorm + ReLU + MaxPool blocks, followed by Global Average Pooling (GAP) to project it to a 128-dimensional latent vector.
* **Tabular Feature Encoder**: A multi-layer MLP projecting the 16 radiomic features and 5 radiologist clinical attributes (subtlety, sphericity, margin, spiculation, texture) to a 64-dimensional vector.
* **Gated Multimodal Fusion Layer**: Projects tabular features to the 128-dimensional spatial latent space, then computes attention gating weights $g_{\text{cnn}}$ and $g_{\text{rad}}$ to dynamically prioritize imaging vs. tabular features:
  $$f_{\text{fused}} = g_{\text{cnn}} \odot f_{\text{cnn}} + g_{\text{rad}} \odot f_{\text{rad\_proj}}$$
* **Multi-Task Heads**: Predicts binary malignancy logits (classification) and auxiliary continuous malignancy ratings (regression).

### 3. Patient-Grouped Training Loop (`train.py`)
* **Preventing Data Leakage**: In medical ML, we must split train/validation sets by Patient ID, otherwise there is high data leakage (multiple nodules from the same patient scan would end up in both training and testing sets). We enforce strict patient separation using `GroupKFold` on the `patient_id` column.
* **RAM Caching**: Loads and pre-computes all 3D patch radiomic features at runtime initialization, caching them in memory to accelerate GPU epoch cycles.
* **Multi-Task Loss**: Minimizes Cross-Entropy (for classification) and MSE (for continuous consensus ratings).
* Tracks AUC-ROC, Sensitivity, Specificity, F1-Score, and Accuracy on validation folds.

---

## 🚀 Execution & Setup

### Running Local Training
Execute the training script from the CLI:
```bash
python train.py \
    --manifest_path "/content/drive/MyDrive/Lung_Nodule_Project/processed_patches/manifest.csv" \
    --output_dir "/content/drive/MyDrive/Lung_Nodule_Project/models" \
    --epochs 15 \
    --batch_size 16 \
    --learning_rate 0.0003
```

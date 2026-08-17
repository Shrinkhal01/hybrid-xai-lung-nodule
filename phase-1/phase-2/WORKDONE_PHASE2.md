# Phase 2: Hybrid Malignancy Classifier

Implemented a hybrid, multi-modal, multi-task deep learning architecture designed to combine raw volumetric imaging data with clinical and morphological features.

---

## 🛠️ 1. Model Architecture

The network builds a model called `HybridMalignancyNet` that processes two distinct streams of data:

### 📹 Imaging Stream (3D Spatial CNN)
* **Input**: Resampled, Hounsfield Unit (HU) normalized 3D CT patch of size `(64, 64, 64)` centered on the nodule.
* **Process**: Passes it through four 3D convolutional blocks (each containing `Conv3D` + `BatchNorm3D` + `ReLU` + `MaxPool3D`), then applies Global Average Pooling (GAP) and a projection layer.
* **Output**: A **128-dimensional** spatial latent vector.

### 📊 Tabular Stream (MLP Encoder)
* **Input**: A **21-dimensional** vector comprising:
  * **16 handcrafted radiomic features** (8 first-order intensity statistics, 4 shape descriptors like sphericity, and 4 GLCM texture features computed from the central axial slice).
  * **5 radiologist clinical attributes** (subtlety, sphericity, margin, spiculation, and texture ratings) from the LIDC-IDRI dataset annotations.
* **Process**: Passes this vector through a multi-layer MLP.
* **Output**: A **64-dimensional** tabular latent vector.

### 🔀 Gated Multimodal Fusion
Learns attention gating weights ($g_{\text{cnn}}$ and $g_{\text{rad}}$) to dynamically weigh and combine the imaging and tabular modalities: 
$$f_{\text{fused}} = g_{\text{cnn}} \odot f_{\text{cnn}} + g_{\text{rad}} \odot f_{\text{rad\_proj}}$$

### 🎯 Multi-Task Heads
* **Classification Head**: Predicts binary malignancy logits (benign vs. malignant).
* **Regression Head**: Auxiliary task predicting the continuous average radiologist rating (1 to 5 scale).

```
[3D CT Patch (64x64x64)] ---> [3D CNN Encoder] ---\
                                                   ===> [Gated Multimodal Fusion] ===> [Classification Head (Malignancy)]
[Tabular Vector (21d)]   ---> [MLP Encoder]    ---/                               ===> [Regression Head (1-5 Score)]
```

---

## ⚙️ 2. Dataset & Optimization Parameters

* **Dataset Size**: **71 nodules** were cached and processed. 
  > [!NOTE]
  > This is a very small dataset subset, likely used to verify that the pipeline runs without out-of-memory (OOM) errors.
* **Validation Strategy**: **5-Fold Patient-Grouped Cross-Validation** (`GroupKFold` split on `patient_id`). This is critical in medical imaging to prevent data leakage (ensuring different nodules from the same patient scan do not end up in both training and validation splits).
* **Feature Scaling**: Tabular features scaled via `StandardScaler` fitted only on training fold slices and applied to validation slices.
* **Multi-Task Loss**:
  $$\text{Total Loss} = \text{Loss}_{\text{clf}} + 0.5 \times \text{Loss}_{\text{reg}}$$
  * `Loss_clf` uses weighted Cross-Entropy to address class imbalance.
  * `Loss_reg` uses Mean Squared Error (MSE).
* **Optimizer**: `AdamW` with a learning rate of `3e-4` ($0.0003$) and weight decay of `1e-4`.

---

## 📈 3. Training & Convergence

The training was executed for **15 epochs** across all 5 folds on a GPU (`cuda`).

### Loss Trajectory
Across all folds, the training loss steadily decreased, proving that the model successfully converged during training:

| Fold | Initial Loss (Epoch 1) | Final Loss (Epoch 15) | Notes |
| :--- | :---: | :---: | :--- |
| **Fold 1** | `5.3111` | `0.9849` | Lowest loss: `0.7049` at epoch 14 |
| **Fold 2** | `4.8981` | `0.7944` | Steady convergence |
| **Fold 3** | `5.7775` | `0.8002` | Steady convergence |
| **Fold 4** | `4.8580` | `0.8060` | Lowest loss: `0.8494` at epoch 14 |
| **Fold 5** | `5.7801` | `0.7021` | Strongest overall convergence |

---

## ⚠️ 4. Limitations of the Current Phase-2 Run

> [!WARNING]
> Before moving to Phase 3, keep in mind these current validation and resource constraints:
>
> * **Metrics Tracking**: While accuracy and validation metrics are calculated mathematically, they were not printed or plotted during the Colab training loop. Only training loss was printed.
> * **Model Saving**: The model checkpoints (`.pt` files) were not serialized or saved to `MODEL_DIR` at the end of training.
> * **Data Volume**: The run was limited to 71 nodules. To generalize, training must be scaled up to a larger slice of the LIDC-IDRI dataset.

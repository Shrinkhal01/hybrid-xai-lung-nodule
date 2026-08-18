## NOTES
- This project is a comprehensive end-to-end deep learning pipeline for lung nodule malignancy classification, utilizing the LIDC-IDRI dataset.
- The goal is to accurately classify lung nodules as either benign or malignant by combining 3D medical imaging (CT scans) with tabular clinical features. 
- The project is structured into four distinct phases :
1. **Phase 1: Data Acquisition & Preprocessing**

    > **Downloading:** The downloader.ipynb script securely fetches the raw LIDC-IDRI CT scan series directly from The Cancer Imaging Archive (TCIA) into your Google Drive.
    > **Preprocessing:** The phase1preprocess.ipynb notebook handles the heavy lifting of medical image processing. It converts raw DICOM files into 3D volumes, resamples them to an isotropic resolution, and applies "lung windowing" (clipping Hounsfield Units) to isolate lung tissue.
    > **Extraction:** It uses the pylidc library to parse expert annotations, extracts targeted 64x64x64 3D patches around the nodules, and gathers tabular features (subtlety, sphericity, margin, spiculation, texture) to create a structured dataset (manifest.csv).

2. **Phase 2: Model Architecture & Prototyping**

    > **Hybrid Architecture:** In phase2.ipynb, a Hybrid Gated Fusion Model is built. It uses a two-pronged approach:
            > **3D CNN branch:** Processes the spatial 3D image patches.
            > **Tabular MLP branch:** Processes the clinical metadata.
    > **Gated Fusion:** The features from both branches are concatenated and passed through a gating mechanism to dynamically weight the importance of the image versus the tabular data before making a final classification.
    > **Validation:** The model is prototyped using a 5-fold GroupKFold cross-validation strategy to ensure no data leakage occurs at the patient level.

3. **Phase 3: Evaluation & Explainability (XAI)**

    > **Ensemble Inference:** phase3_evaluation.ipynb evaluates the trained models from all 5 folds together, calculating an ensemble ROC-AUC score and using Youden's J Statistic to find the mathematically optimal probability threshold.
    > **Explainable AI (XAI):** It features a custom 3D Grad-CAM implementation to generate heatmaps over the CT scans. This visualizes exactly where the model is looking when it predicts a nodule is malignant, ensuring clinical trust.
    > **Ablation Study:** A modality ablation study is conducted (zeroing out images, then zeroing out tabular data) to scientifically prove that the hybrid combination outperforms using either data source on its own.

4. **Phase 4: Full-Scale Optimization**

    > **Optimized Training:** The phase4_full_scale.ipynb notebook scales the training loop to the entire dataset (over 1,200 nodules). It implements Automatic Mixed Precision (AMP) to drastically reduce memory consumption and speed up training on the GPU.
    > **Benchmarking:** Finally, it compares the model's performance against established literature (like Traditional SVMs, Standard 3D CNNs, and SOTA models like NoduleX), tracking its trajectory to highly competitive AUC scores.


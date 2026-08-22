# Hybrid XAI Lung Nodule Analysis
![hybrid-xai-lung-nodule.jpg]
- Out of your 1,608 total nodules, exactly 366 of them had an indeterminate score of 3.0.
```
1,608 - 366 = 1,242
```
- This is why your Sanity Check cell printed out:Total valid nodules for binary classification: 1242

- The line that instructs the AI to drop the ambiguous Class -1 (Indeterminate) nodules so it can learn a clear, binary "Benign vs. Malignant" classification is : 
```
self.manifest = self.manifest[self.manifest['malignancy_class'] != -1].reset_index(drop=True)
```

- These 1,608 nodules are grouped into three classes based on the 4 radiologists' average malignancy scores:
```
Class 0 (Benign): Score < 3.0
Class 1 (Malignant): Score > 3.0
Class -1 (Indeterminate): Score exactly = 3.0 (borderline cases where the radiologists were evenly split).
```

---

## Model Performance & Evaluation Metrics

The final ensemble model was evaluated across **1,242 consensus-labeled nodules** using 5-fold cross-validation.

### Performance Summary Table

| Metric | Result | Description |
| :--- | :---: | :--- |
| **Overall Accuracy** | **86.00%** | Proportion of all nodules correctly classified as benign or malignant. |
| **Ensemble ROC-AUC** | **0.9042** | Diagnostic discrimination capability across all classification thresholds. |
| **Calibrated Threshold** | **0.5783** | Optimal decision boundary derived via Youden's J statistic. |
| **Malignant (Precision / Recall / F1)** | **0.77 / 0.76 / 0.77** | Solid detection of cancerous nodules despite dataset class imbalance. |
| **Benign (Precision / Recall / F1)** | **0.89 / 0.90 / 0.90** | High specificity and precision for non-malignant scans. |

---

### Detailed Metric Breakdown

#### 1. Overall Accuracy (86.00%)
* **Definition:** The ratio of correct predictions (both True Benign and True Malignant) to the total number of evaluated samples (1,242 nodules).
* **Clinical Significance:** 86% of patient scans receive an accurate automated diagnosis.

#### 2. Ensemble ROC-AUC (0.9042)
* **Definition:** The **Area Under the Receiver Operating Characteristic Curve**. It measures the model's ability to rank a randomly chosen malignant nodule higher than a randomly chosen benign nodule across every possible decision threshold.
* **Clinical Significance:** A score above **0.90** indicates excellent diagnostic discrimination, confirming that the ensemble generalizes well across unseen patient folds.

#### 3. Calibrated Decision Threshold (0.5783 via Youden's J)
* **Definition:** Rather than using an arbitrary 50% ($0.50$) default cutoff, the optimal threshold was mathematically calculated using **Youden's Index** ($J = \text{Sensitivity} + \text{Specificity} - 1$).
* **Clinical Significance:** Calibrates the consensus probability cutoff to maximize sensitivity (catching cancer) while minimizing false-positive alarms (reducing unnecessary invasive biopsies).

#### 4. Malignant (Cancerous) Detection
* **Precision (0.77):** When the model flags a nodule as malignant, it is correct **77%** of the time.
* **Recall / Sensitivity (0.76):** The model successfully identifies **76%** of all actual malignant nodules in the cohort.
* **F1-Score (0.77):** The harmonic mean of precision and recall ($\frac{2 \cdot P \cdot R}{P + R}$), providing a balanced benchmark for the minority malignant class.

#### 5. Benign (Non-Cancerous) Detection
* **Precision (0.89):** High reliability—**89%** of nodules predicted as benign are truly non-cancerous.
* **Recall / Specificity (0.90):** Successfully detects **90%** of all benign nodules, preventing misdiagnoses.
* **F1-Score (0.90):** Strong overall harmonic balance for the benign class.

---

### Multimodal Ablation Study

To evaluate the contribution of each branch, an ablation study was conducted:

| Model Configuration | ROC-AUC | Clinical Interpretation |
| :--- | :---: | :--- |
| **3D CNN Only** | `0.6901` | Spatial 3D image features alone struggle due to visual complexity and patch variance. |
| **Tabular MLP Only** | `0.8985` | Strong baseline using handcrafted radiologist features (subtlety, sphericity, margin, etc.). |
| **Hybrid Gated Fusion** | `0.9042` | **Best performance.** Multimodal gating dynamically weights visual + radiomic cues for superior classification. |

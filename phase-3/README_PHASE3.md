# Phase 3: Model Evaluation, Calibration & 3D Grad-CAM (XAI)

## Overview
This phase focused on rigorously testing the trained Hybrid Gated Fusion model from Phase 2. The objective was to evaluate its predictive performance using ensemble inference, calibrate the decision threshold for clinical reliability, visualize its internal decision-making process, and scientifically prove the value of combining both 3D images and tabular metadata.

---

## 🛠️ Tools and Libraries Used

- **PyTorch (`torch`, `torch.nn.functional`)**: Used to load the saved fold weights, run inference without tracking gradients (`torch.no_grad()`), and execute the forward/backward hooks required to extract gradients for the Grad-CAM heatmaps.
- **Scikit-Learn (`sklearn.metrics`)**: Essential for calculating the evaluation metrics, specifically generating the Receiver Operating Characteristic (ROC) curve, calculating the Area Under the Curve (AUC), and generating the comprehensive classification report (precision, recall, f1-score).
- **SciPy (`scipy.ndimage`)**: Utilized to upsample the small $4 \times 4 \times 4$ activation maps from the deep convolutional layers back to the original $64 \times 64 \times 64$ voxel resolution for the Grad-CAM overlays.
- **Matplotlib**: Used to plot the ROC curve, generate the three-plane (Axial, Coronal, Sagittal) Grad-CAM visual heatmaps, and chart the results of the ablation study.
- **NumPy**: Handled the mathematical array operations, specifically finding the maximum index for Youden's J statistic and averaging the gradient weights.

---

## 🔄 Process Workflow

1. **Ensemble Inference**: Instead of relying on a single model, predictions were aggregated across all 5 cross-validation models generated in Phase 2. The outputs were averaged to produce a highly robust probability score for each nodule, minimizing the variance and overfitting of any single fold.

2. **Threshold Calibration (Youden's J Statistic)**: Standard classification defaults to a 0.50 probability cutoff. To optimize the balance between detecting true cancers (Sensitivity) and avoiding false alarms (Specificity), the optimal threshold was calculated using Youden's J Statistic:
   $$J = \text{Sensitivity} + \text{Specificity} - 1$$
   This mathematical calibration identified the exact probability cutoff that maximized clinical effectiveness. (For the prototype run, this yielded an impressive ROC-AUC of 0.8760 at a threshold of 0.4824).

3. **3D Grad-CAM Explainability (XAI)**: To ensure the model was functioning as a trustworthy medical tool rather than a "black box," Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented. By capturing the gradients from the final 3D convolutional layer during backpropagation, a heatmap was generated and overlaid onto the original CT slices. This visually verified that the network was focusing on the actual anatomical borders of the malignant nodules to make its predictions.

4. **Modality Ablation Study**: To scientifically validate the hybrid architecture, an ablation study was conducted. The model's performance was evaluated under three conditions:
   - **Full Hybrid Model** (Images + Tabular)
   - **Tabular MLP Only** (Images artificially zeroed out)
   - **3D CNN Only** (Tabular features artificially zeroed out)
   
   This proved that the Gated Fusion Layer was actively utilizing both data streams, as removing either modality resulted in a measurable drop in the ROC-AUC score.
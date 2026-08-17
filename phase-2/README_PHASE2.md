## Overview
- This phase focused on designing, building, and training a dual-branch neural network capable of processing both 3D medical images and tabular clinical metadata. The model was trained using a robust cross-validation strategy on the prototype subset to ensure generalized learning and prevent data leakage.

### Tools and Libraries Used
- PyTorch (torch, torch.nn): The primary deep learning framework. It was used to construct the custom dataset classes, define the neural network layers (3D convolutions, multi-layer perceptrons, gating mechanisms), and manage the training loop and gradient optimization.
- Pandas: Used to load the manifest.csv tracking file, filter out indeterminate nodules (class -1), and manage the tabular radiomic features before feeding them into the network.
- Scikit-Learn (GroupKFold, StandardScaler): GroupKFold was essential for splitting the data while ensuring that multiple nodules from the exact same patient did not bleed across the training and validation sets. StandardScaler was used to normalize the tabular features dynamically within each fold to prevent data leakage.
- NumPy: Used for calculating class weights to handle the inherent imbalance between benign and malignant nodules in the dataset.

### Process Workflow
- PyTorch Dataset Definition: Created a custom LungNoduleDataset class to seamlessly load the 3D .pt image tensors and the corresponding 5-element tabular feature vectors (subtlety, sphericity, margin, spiculation, texture). PyTorch's weights_only=False flag was utilized to bypass security blocks on custom tensor dictionary files.
- Hybrid Architecture Construction:
  - 3D CNN Branch: Engineered to extract volumetric spatial features from the 64×64×64 CT patches using cascaded 3D convolutions and max-pooling layers.
  - Tabular MLP Branch: A Multi-Layer Perceptron designed to encode the numerical clinical attributes into dense feature vectors.
  - Gated Fusion Layer: A dynamic integration mechanism using a sigmoid activation gate. This allows the network to learn input-dependent weights, adaptively modulating the importance of image versus tabular features before the final classification head.
- Sanity Verification: Executed a single-batch forward pass to strictly validate that the PyTorch DataLoader outputs aligned perfectly with the input dimension requirements of the dual-branch network.
- Cross-Validation Training: Implemented a 5-fold GroupKFold training loop. The cross-entropy loss function was weighted to penalize misclassifications of minority classes. The model was successfully trained on the 85-patch prototype subset, generating 5 independent sets of fold weights (hybrid_model_fold1.pt to fold5.pt).
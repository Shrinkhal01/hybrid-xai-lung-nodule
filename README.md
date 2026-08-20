# Hybrid XAI Lung Nodule Analysis

!(hybrid-xai-lung-nodule.jpg)

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

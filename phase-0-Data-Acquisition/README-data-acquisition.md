# Phase 0: Cloud-to-Cloud Data Acquisition

## 📌 Overview
Phase 0 handles the foundational step of the Hybrid Explainable AI Framework for Accurate Lung Nodule Detection: acquiring the raw medical data.

- The official LIDC-IDRI Dataset consists of 1,010 patient subjects spanning 1,018 CT series, totaling approximately 133 GB of high-resolution DICOM files.
- Downloading this massive volume of data to a local machine with limited storage (and subsequently re-uploading it to the cloud for processing) is highly inefficient.
- This script (`downloader.ipynb`) solves that bottleneck by executing a direct cloud-to-cloud data stream, pulling the data straight from the Cancer Imaging Archive's REST API into a mounted Google Drive workspace.

---

## 🛠️ Technologies Used & Why

| Technology | Purpose in Pipeline | Why It Was Chosen |
| :--- | :--- | :--- |
| **Google Colab** | Cloud Execution Environment | Provides high-speed cloud bandwidth and direct integration with Google Drive, ensuring zero gigabytes touch local hardware storage. |
| **`tcia_utils`** | NBIA API Communication | The official Python wrapper for the TCIA REST API. It dynamically queries the live database for the exact CT series UIDs and streams the binary files flawlessly. |
| **Google Drive** | Cloud Storage Destination | Acts as the centralized, persistent storage vault (`raw_data/`) that will be accessed by all subsequent PyTorch processing phases. |

---

## ⚙️ How It Works (The Pipeline Architecture)

The `downloader.ipynb` notebook executes the following automated steps:

1. **Environment Setup & Mounting**
   Secures a connection between the Google Colab temporary runtime and the persistent Google Drive storage.

2. **Dynamic API Querying**
   Queries the TCIA server specifically for `collection="LIDC-IDRI"` and `modality="CT"`, returning a complete manifest of all 1,018 target series.

3. **Cloud-to-Cloud Streaming**
   Iterates through the API response and downloads the raw DICOM zip files directly into the specified Drive folder (`/content/drive/MyDrive/Lung_Nodule_Project/raw_data`).

4. **Automatic Extraction**
   Unzips the DICOM files into individual directories named by their unique Series Instance UIDs (e.g., `1.3.6.1.4.1.14519.5.2.1...`).

5. **Resume Safety Logic**
   The script checks the destination folder before downloading. If the Colab runtime disconnects or times out, re-running the script safely skips already-downloaded series, preventing duplicated work and corrupted files.

---

## 📂 Output Directory Structure

Upon successful completion, the output directory (`raw_data`) will be populated with over 1,000 raw series folders, ready for Phase 1 restructuring and preprocessing:

```text
raw_data/
│
├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192/
│   ├── 000001.dcm
│   ├── 000002.dcm
│   └── ... 
├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.129007566048223160327836686225/
│
└── ... (1,018 total CT series folders)
```

---

## 🚀 Execution Notes

- **Network Speed**: The complete 133 GB transfer takes approximately 1.5 to 3 hours depending on Colab's allocated cloud bandwidth and TCIA server traffic.
- **Storage Requirements**: Ensure the target Google Drive account has at least 150 GB of available storage space before initiating the script.
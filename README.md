# Med-Chain: Secure Medical AI & Blockchain Integration

**Med-Chain** is a decentralized healthcare platform designed to diagnose Pneumonia from Chest X-rays using Deep Learning (MobileNetV2). To ensure data integrity and privacy, the system integrates IPFS for decentralized storage and Ethereum Blockchain for securing medical records.

---

## Project Overview

Med-Chain combines three major technology domains:

- **Artificial Intelligence** — MobileNetV2 transfer learning model trained on chest X-ray images to classify Pneumonia vs Normal cases.
- **Explainable AI (XAI)** — Grad-CAM heatmaps overlaid on the original X-ray to visually explain which lung regions the AI focused on during diagnosis.
- **Blockchain + Decentralized Storage** — SHA-256 image hashing, AES-256 encryption, IPFS storage (via IPFS Kubo), and Ethereum smart contract integration (via Ganache + Truffle) to permanently secure and verify medical records.

---

## Step-by-Step Implementation Process

1. **Data Acquisition** — Utilized the Chest X-ray dataset (Pneumonia vs Normal) from Kaggle for training and validation.
2. **AI Model Development** — Implemented Transfer Learning using the MobileNetV2 architecture to achieve high accuracy with minimal computational resources.
3. **Explainable AI (XAI)** — Integrated Grad-CAM to generate heatmaps, allowing clinicians to visualize the specific lung regions the AI focused on during diagnosis.
4. **Security Layer** — Developed a hashing module using SHA-256 to create a unique digital fingerprint for every medical image. AES-256 encryption is applied to protect the raw image file.
5. **Decentralized Storage** — Integrated IPFS to store high-resolution X-ray images, retrieving a unique CID (Content Identifier) for each upload.
6. **Blockchain Integration** — Authored a Solidity Smart Contract to record the IPFS CID and image hash on a local Ethereum network (Ganache).
7. **Web Dashboard** — Developed a real-time diagnostic interface using Streamlit.

---

## Technology Stack

| Layer | Technology |
| :--- | :--- |
| AI Model | TensorFlow 2.x, MobileNetV2, Keras |
| Explainable AI | Grad-CAM (via TensorFlow GradientTape) |
| Image Processing | OpenCV, Pillow, NumPy |
| Security | SHA-256 (hashlib), AES-256 (PyCryptodome) |
| Decentralized Storage | IPFS Kubo (HTTP API on port 5001) |
| Blockchain | Ethereum, Ganache (local testnet), Truffle, Web3.py |
| Web Interface | Streamlit |
| Language | Python 3.11 |

---

## Bug Log & Troubleshooting

| S.No | Issue Encountered | Resolution Strategy |
| :--- | :--- | :--- |
| 1 | **Environment Conflicts** | Encountered `ModuleNotFoundError` for TensorFlow. TensorFlow only supports Python 3.9 to 3.11. Resolved by installing Python 3.11 specifically and creating a dedicated virtual environment. |
| 2 | **Python Version Incompatibility** | Python 3.13 and 3.14 are not supported by TensorFlow. Any attempt to install TensorFlow on these versions returns `No matching distribution found`. Must use Python 3.11. |
| 3 | **IPFS API Incompatibility** | Recent updates in IPFS Kubo caused issues with standard Python libraries. Resolved by using direct HTTP POST requests to the IPFS API at `http://127.0.0.1:5001/api/v0/add`. |
| 4 | **Model Prediction Bias** | Model initially showed a bias towards Pneumonia due to class imbalance in the dataset. Corrected by implementing Class Weights `{NORMAL: 2.5, PNEUMONIA: 1.0}` during the training phase. |
| 5 | **Git Remote Configuration** | A syntax error in the remote URL (extra characters) prevented code pushes. Resolved by manually editing the `.git/config` file and using GitHub Desktop for a clean sync. |
| 6 | **GitHub File Size Limits** | The dataset exceeded GitHub's 100MB limit. Resolved by configuring a `.gitignore` file to exclude large binary data while keeping the source code intact. |

---

## Project File Structure

```
Med-Chain-Project/
|
|-- app.py                    # Main Streamlit web application (entry point)
|-- train_local.py            # Script to train the MobileNetV2 model on the dataset
|-- check_model.py            # Quick script to verify model architecture loads correctly
|-- load_data.py              # Helper script to validate dataset loading
|-- secure_data.py            # AES-256 encryption and SHA-256 hashing module
|-- blockchain_sync.py        # Script to push CID and hash to Ethereum blockchain
|-- upload_to_ipfs.py         # Script to upload encrypted X-ray to local IPFS node
|-- local_model_weights.h5    # Saved trained model weights (~10MB)
|-- encrypted_xray.bin        # Example AES-256 encrypted X-ray output
|-- docs/
|   |-- documentation.md      # Full technical documentation
|-- README.md
```

---

## Installation & Usage

### Prerequisites

Before running the application, ensure the following are installed and running:

- **Python 3.11** (strictly required — TensorFlow does not support 3.12+ yet)
- **IPFS Desktop** — https://docs.ipfs.tech/install/ipfs-desktop/ (must be running on port 5001)
- **Ganache** — https://trufflesuite.com/ganache/ (local Ethereum, must be running on port 7545)
- **Truffle** — Install via `npm install -g truffle` (requires Node.js)

### 1. Environment Setup

```bash
# Create a virtual environment using Python 3.11 specifically
py -3.11 -m venv med_env

# Activate it (Windows)
med_env\Scripts\activate

# Install all dependencies
pip install tensorflow streamlit pillow numpy opencv-python requests web3 pycryptodome matplotlib
```

### 2. Verify Model Loads

```bash
python check_model.py
```

Expected output: `--- Success! Model Initialized ---` followed by the model layer summary.

### 3. Local Blockchain Setup

Ensure Ganache is running, then deploy the smart contract:

```bash
cd blockchain_layer
truffle migrate --reset
```

Copy the deployed contract address from the Truffle output and update it in `app.py` and `blockchain_sync.py`.

### 4. Launch App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

The sidebar shows live connection status for IPFS and Blockchain. The AI diagnosis and Grad-CAM heatmap features work independently without IPFS or Blockchain running.

---

## How to Test (AI Features Only)

1. Activate the virtual environment: `med_env\Scripts\activate`
2. Run the app: `streamlit run app.py`
3. Upload any chest X-ray image (JPG/PNG)
4. The app will display the diagnosis (NORMAL or PNEUMONIA) and a Grad-CAM heatmap
5. The "Secure & Sync to Blockchain" button requires both IPFS and Ganache to be running

A sample dataset for testing is available at:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## Notes

- The `blockchain_layer/` folder containing the Solidity smart contract is not tracked in this repository due to build artifact size. It must be set up locally using Truffle.
- The `local_model_weights.h5` file contains only the trained weights, not the full model. The architecture is rebuilt in code in `app.py` before weights are loaded.
- IPFS must be running locally before clicking "Secure & Sync". The app uses the local IPFS HTTP API, not a public gateway.

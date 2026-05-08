# Full Technical Documentation: Med-Chain (AI + Blockchain)

---

## 1. Project Architecture

The system processes a chest X-ray image through five sequential layers:

1. **Input** — A doctor or patient uploads a Chest X-ray image via the Streamlit web interface.
2. **AI Layer** — The MobileNetV2 model preprocesses the image and predicts whether the case is Pneumonia or Normal, along with a confidence score.
3. **XAI Layer** — Grad-CAM generates a heatmap overlaid on the original image, highlighting the lung regions that most influenced the AI's decision.
4. **Storage Layer** — The image is encrypted using AES-256 and uploaded to IPFS. IPFS returns a unique CID (Content Identifier) for retrieval.
5. **Blockchain Layer** — The image's SHA-256 hash, IPFS CID, and AI diagnosis are recorded permanently on a local Ethereum network (Ganache) via a Solidity smart contract.

### System Flow Diagram

```
User uploads X-ray
        |
        v
[AI Layer: MobileNetV2]
  - Resize image to 224x224
  - Normalize pixel values
  - Predict: NORMAL or PNEUMONIA
  - Output confidence score
        |
        v
[XAI Layer: Grad-CAM]
  - Compute gradients from last conv layer (Conv_1)
  - Generate heatmap
  - Overlay heatmap on original image
        |
        v
[Security Layer]
  - SHA-256 hash of raw image (tamper detection)
  - AES-256 encryption of image file
        |
        v
[Storage Layer: IPFS]
  - Upload encrypted image via HTTP POST to local IPFS node
  - Receive CID (e.g., QmTaJoVMiV...)
        |
        v
[Blockchain Layer: Ethereum/Ganache]
  - Call addRecord() on Solidity smart contract
  - Store: CID + image hash permanently on-chain
  - Returns transaction hash as proof
```

---

## 2. Phase 1: AI Model & Preprocessing (Deep Learning)

### A. Why MobileNetV2?

MobileNetV2 is a lightweight convolutional neural network originally trained by Google on the ImageNet dataset (1.4 million images, 1000 classes). Instead of training from scratch, which requires massive datasets and weeks of GPU time, Transfer Learning allows us to reuse the feature-extraction layers of MobileNetV2 and only train a small custom output layer on our X-ray dataset.

This approach achieves high accuracy with significantly less data and compute.

### B. Model Architecture

```
Input (224 x 224 x 3 RGB image)
        |
MobileNetV2 Base (pre-trained on ImageNet, weights frozen)
        |
GlobalAveragePooling2D (reduce spatial dimensions)
        |
Dense(128, activation='relu')  -- custom layer
        |
Dense(2, activation='softmax') -- output: [NORMAL, PNEUMONIA]
```

The final output is a probability vector. For example: `[0.06, 0.94]` means 6% Normal, 94% Pneumonia.

### C. Image Preprocessing

MobileNetV2 requires input images of exactly 224x224 pixels with pixel values between 0 and 1.

Steps applied to every uploaded image:
1. Convert to RGB (handles grayscale X-rays that have only 1 channel)
2. Resize to (224, 224)
3. Convert to NumPy array
4. Normalize by dividing by 255.0
5. Add batch dimension: shape becomes (1, 224, 224, 3)

### D. Training Details (train_local.py)

| Parameter | Value |
| :--- | :--- |
| Base Model | MobileNetV2 (ImageNet weights) |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Epochs | 15 |
| Batch Size | 16 |
| Validation Split | 20% |
| Class Weights | NORMAL: 2.5, PNEUMONIA: 1.0 |
| Augmentation | Rotation (10 deg), Zoom (10%), Horizontal Flip |

Class weights were applied because the Chest X-ray dataset contains significantly more Pneumonia samples than Normal. Without this correction, the model is biased toward always predicting Pneumonia. Giving a higher weight to NORMAL forces the model to pay equal attention to both classes.

The trained weights are saved to `local_model_weights.h5`. The full model is not saved as a single file because the architecture is rebuilt in code each time the app starts, then the weights are loaded into it.

---

## 3. Phase 2: Explainable AI (Grad-CAM)

### Why Explainability Matters

A black-box AI model in a medical context is dangerous. If the model says "Pneumonia" but the doctor cannot see why, there is no basis for trust. Grad-CAM (Gradient-weighted Class Activation Mapping) solves this by producing a visual explanation.

### How Grad-CAM Works

Grad-CAM measures how much each spatial region in the last convolutional layer's feature map contributed to the final prediction. Regions with a high positive gradient are colored red/yellow on the heatmap — these are the areas the AI "looked at" most strongly.

### Steps in the Implementation

1. Create a sub-model that outputs both the last conv layer (`Conv_1` in MobileNetV2) and the final prediction.
2. Run a forward pass with `tf.GradientTape()` to record operations.
3. Compute the gradient of the predicted class score with respect to the last conv layer's output.
4. Average the gradients across all channels using `tf.reduce_mean`.
5. Multiply the averaged gradients with the feature map to get the heatmap.
6. Apply ReLU to keep only positive contributions.
7. Resize the heatmap to match the original image dimensions (224x224).
8. Apply the JET colormap using OpenCV (`cv2.COLORMAP_JET`) — red = high activation, blue = low.
9. Blend the heatmap with the original image using `cv2.addWeighted` (60% original, 40% heatmap).

The output is displayed directly in the Streamlit interface alongside the original X-ray.

---

## 4. Phase 3: Security Layer

### SHA-256 Hashing (secure_data.py)

Every medical image gets a SHA-256 digital fingerprint. This is a 64-character hexadecimal string derived from the raw bytes of the image file. If even a single pixel is changed, the hash changes completely. This makes it possible to verify at any time that the image stored on IPFS is identical to the one originally uploaded.

Example hash:
```
de2079fb45d4bb0e4c6a48481e7b6ee31ab7601e83b43cd07820168b93de1d13
```

### AES-256 Encryption (secure_data.py)

Before uploading to IPFS, the raw image is encrypted using AES-256 in EAX mode (authenticated encryption). This ensures that the stored file is unreadable to anyone who does not hold the 256-bit encryption key.

The encrypted output (`encrypted_xray.bin`) contains three parts written sequentially:
- **Nonce** — 16-byte random value used to initialize the cipher
- **Tag** — authentication tag to detect tampering
- **Ciphertext** — the actual encrypted image bytes

---

## 5. Phase 4: Decentralized Storage (IPFS)

### Why IPFS Instead of a Central Server?

Storing medical images on a centralized server creates a single point of failure and a single target for attackers. IPFS distributes the data across many nodes. More importantly, IPFS uses content-addressed storage — the CID is derived from the content itself, so the same file always produces the same CID. This means you can always independently verify that the file you retrieved is exactly what was stored.

### How the Upload Works (upload_to_ipfs.py)

The IPFS Python library (`ipfshttpclient`) had compatibility issues with newer versions of IPFS Kubo. The solution was to bypass the library entirely and make a direct HTTP POST request to the local IPFS node's API.

Endpoint used:
```
POST http://127.0.0.1:5001/api/v0/add
```

The response contains a `Hash` field which is the CID. Example:
```
QmTaJoVMiVeEvMQZdpUX1Jv5jCqG94RHgQ9g4ZW58BwuqN
```

This CID is then publicly accessible via any IPFS gateway:
```
https://ipfs.io/ipfs/QmTaJoVMiVeEvMQZdpUX1Jv5jCqG94RHgQ9g4ZW58BwuqN
```

---

## 6. Phase 5: Blockchain Implementation (Ethereum/Ganache)

### The Smart Contract (Solidity)

The Solidity smart contract defines what data gets stored on the blockchain and exposes two functions: one to add a record, and one to retrieve it.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MedicalRecords {
    struct Record {
        string ipfsCID;
        string imageHash;
        uint256 timestamp;
    }

    Record[] public records;

    function addRecord(string memory _cid, string memory _hash) public {
        records.push(Record(_cid, _hash, block.timestamp));
    }

    function getRecord(uint256 index) public view returns (string memory, string memory, uint256) {
        Record memory r = records[index];
        return (r.ipfsCID, r.imageHash, r.timestamp);
    }
}
```

Data stored per record:
- `ipfsCID` — the IPFS address of the encrypted image
- `imageHash` — the SHA-256 fingerprint of the original image
- `timestamp` — the Unix timestamp when the record was created (added automatically by the blockchain)

Once written, this data cannot be deleted or modified by anyone. This is the core value of blockchain for medical records.

### Deployment Process

1. Write the contract in `blockchain_layer/contracts/MedicalRecords.sol`
2. Configure `truffle-config.js` to point to Ganache at `http://127.0.0.1:7545`
3. Run `truffle migrate --reset` to deploy
4. Truffle outputs the deployed contract address (e.g., `0x9dA1982739cba28e609bD0cB8A1323A1841BBfDA`)
5. Copy this address into `app.py` and `blockchain_sync.py`
6. The contract ABI (interface definition) is read from `blockchain_layer/build/contracts/MedicalRecords.json`

### Python to Blockchain Connection (Web3.py)

```
Ganache (local Ethereum node) running at http://127.0.0.1:7545
        |
Web3.py connects via HTTPProvider
        |
Contract loaded using address + ABI from Truffle build output
        |
contract.functions.addRecord(cid, img_hash).transact({'from': account})
        |
Returns a transaction hash as proof of the write operation
```

---

## 7. Phase 6: Web Interface (Streamlit)

Streamlit converts Python scripts into interactive web apps without requiring any frontend code. The entire UI is defined in `app.py`.

### Interface Components

| Component | Purpose |
| :--- | :--- |
| Sidebar: IPFS status | Live check of connection to `http://127.0.0.1:5001` |
| Sidebar: Blockchain status | Live check of connection to Ganache at `http://127.0.0.1:7545` |
| File uploader | Accepts JPG, JPEG, PNG chest X-ray images |
| Left column | Displays the original uploaded X-ray |
| Right column | Displays AI diagnosis, confidence score, progress bar |
| Grad-CAM section | Displays the heatmap overlaid on the X-ray |
| Sync button | Triggers SHA-256 hash, IPFS upload, and blockchain write |
| Result code block | Shows the IPFS CID and Ethereum transaction hash after sync |

### Model Loading Strategy

The model is loaded once at startup using `@st.cache_resource`. This decorator tells Streamlit to cache the result across all user sessions, so the model is not reloaded every time someone uploads a new image. This is critical because loading TensorFlow model weights is slow.

---

## 8. Known Limitations

| Limitation | Detail |
| :--- | :--- |
| Python version | TensorFlow requires Python 3.9, 3.10, or 3.11. Python 3.12+ is not fully supported. |
| Local-only blockchain | Ganache is a development-only tool. Real deployment would require a public Ethereum testnet (e.g., Sepolia) or mainnet using a service like Alchemy or Infura. |
| Local-only IPFS | The current implementation uses a locally running IPFS node. For cloud deployment, a pinning service such as Pinata or Web3.Storage would be required. |
| Single-image diagnosis | The model classifies one image at a time. Batch processing is not implemented in the current UI. |
| No patient ID system | The current smart contract does not associate records with a patient ID. This would be required for a production medical system. |
| No authentication | The Streamlit interface has no login or access control. Any user can upload an image and trigger a blockchain write. |

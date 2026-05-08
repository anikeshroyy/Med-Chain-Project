import streamlit as st
from PIL import Image
import numpy as np
import os
import requests
from web3 import Web3
import hashlib
import json

# --- Import from ml/ package ---
from ml.predict import load_model, preprocess_image, predict
from ml.gradcam import make_gradcam_heatmap, overlay_heatmap

# --- Page Config ---
st.set_page_config(page_title="Med-Chain AI Node", layout="wide")
st.title("🏥 Med-Chain: Federated Medical AI & Blockchain")
st.write("Secure X-Ray Analysis with Decentralized Storage")

# --- Load AI Model ---
@st.cache_resource
def load_my_model():
    return load_model(weights_path="ml/model/local_model_weights.h5")

model = load_my_model()

# --- Sidebar: Connection Status ---
st.sidebar.header("Connection Status")
ipfs_status = st.sidebar.empty()
eth_status = st.sidebar.empty()

# Check IPFS
try:
    requests.get("http://127.0.0.1:5001/api/v0/version", timeout=2)
    ipfs_status.success("IPFS: Connected")
except:
    ipfs_status.error("IPFS: Disconnected")

# Check Blockchain (Ganache)
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
if w3.is_connected():
    eth_status.success("Blockchain: Connected")
else:
    eth_status.error("Blockchain: Disconnected")

# --- Main UI ---
uploaded_file = st.file_uploader("Upload Chest X-Ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded X-Ray", use_container_width=True)

    with col2:
        st.subheader("AI Diagnostic Results")
        try:
            # 1. Preprocess
            img_array, img_resized = preprocess_image(img)

            # 2. Predict
            with st.spinner("AI is analyzing..."):
                label, confidence = predict(model, img_array)

            # 3. Grad-CAM Heatmap
            st.subheader("Explainable AI (XAI) - Grad-CAM")
            with st.spinner("Generating Heatmap..."):
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='Conv_1')
                superimposed_rgb = overlay_heatmap(heatmap, img_resized)
                st.image(superimposed_rgb, caption="Heatmap: Why AI made this decision?", use_container_width=True)
                st.write("🔴 Red areas show where the AI detected abnormalities.")

            # 4. Display Results
            st.success("Analysis Complete!")
            st.info(f"Diagnosis: **{label}**")
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"AI Error: {e}")
            print(f"ERROR: {e}")

    # --- Security & Sync Section ---
    st.divider()
    if st.button("Secure & Sync to Blockchain"):
        with st.spinner("Encrypting and Uploading..."):
            try:
                # 1. Generate Hash
                file_bytes = uploaded_file.getvalue()
                img_hash = hashlib.sha256(file_bytes).hexdigest()

                # 2. Upload to IPFS
                files = {'file': file_bytes}
                ipfs_res = requests.post("http://127.0.0.1:5001/api/v0/add", files=files).json()
                cid = ipfs_res['Hash']

                # 3. Record on Blockchain
                contract_address = "0x7732384A96eBD07974515DD3A9ED3Fd9287697c8"
                with open('blockchain/blockchain_layer/build/contracts/MedicalRecords.json') as f:
                    abi = json.load(f)['abi']

                contract = w3.eth.contract(address=contract_address, abi=abi)
                account = w3.eth.accounts[0]
                tx_hash = contract.functions.addRecord(cid, img_hash).transact({'from': account})

                st.success("Successfully Synced!")
                st.code(f"IPFS CID: {cid}\nTx Hash: {tx_hash.hex()}")

            except Exception as e:
                st.error(f"Blockchain/IPFS Error: {e}")
                st.info("Make sure IPFS Desktop and Ganache are running.")
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import os

def generate_image_hash(image_path):
    """Creates a SHA-256 fingerprint of the image."""
    with open(image_path, "rb") as f:
        bytes = f.read()
        readable_hash = hashlib.sha256(bytes).hexdigest()
    return readable_hash

def encrypt_image(image_path, key):
    """Encrypts the image using AES-256."""
    cipher = AES.new(key, AES.MODE_EAX)
    with open(image_path, "rb") as f:
        data = f.read()
    
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    
    # Save the encrypted file
    with open("encrypted_xray.bin", "wb") as f:
        [f.write(x) for x in (nonce, tag, ciphertext)]
    print("--- Image Encrypted Successfully ---")

# --- Test it ---
# Pick any image from your 'Normal' folder
test_image = r'C:\Users\prave\Desktop\project 8\chest_xray\chest_xray\train\NORMAL\IM-0115-0001.jpeg'
key = get_random_bytes(32) # 256-bit key

img_hash = generate_image_hash(test_image)
print(f"Digital Fingerprint (Hash): {img_hash}")
encrypt_image(test_image, key)
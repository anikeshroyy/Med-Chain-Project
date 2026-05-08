import requests

def upload_to_ipfs(filename):
    # The default IPFS API port is 5001
    url = "http://127.0.0.1:5001/api/v0/add"
    
    try:
        with open(filename, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                data = response.json()
                cid = data['Hash']
                print("\n--- Success! File Uploaded to IPFS ---")
                print(f"IPFS CID: {cid}")
                print(f"Public Link: https://ipfs.io/ipfs/{cid}")
                return cid
            else:
                print(f"Error: IPFS returned status {response.status_code}")
    except Exception as e:
        print(f"Failed to connect to IPFS: {e}")
        print("Make sure IPFS Desktop is open and running!")

# Run the upload
upload_to_ipfs('encrypted_xray.bin')
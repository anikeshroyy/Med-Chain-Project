from web3 import Web3
import json

# 1. Connect to Ganache
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

# 2. Contract Details (Copy these from your Truffle output)
contract_address = "0x9dA1982739cba28e609bD0cB8A1323A1841BBfDA" # Your address from the screenshot
# The ABI is the 'menu' of your contract. Truffle saved it in blockchain_layer/build/contracts/MedicalRecords.json
with open('blockchain_layer/build/contracts/MedicalRecords.json') as f:
    info_json = json.load(f)
    abi = info_json['abi']

contract = web3.eth.contract(address=contract_address, abi=abi)

# 3. Data to store (The CID and Hash you generated earlier)
ipfs_cid = "QmTaJoVMiVeEvMQZdpUX1Jv5jCqG94RHgQ9g4ZW58BwuqN"
img_hash = "de2079fb45d4bb0e4c6a48481e7b6ee31ab7601e83b43cd07820168b93de1d13"

# 4. Push to Blockchain
# We use the first Ganache account to pay for this
account = web3.eth.accounts[0]
tx_hash = contract.functions.addRecord(ipfs_cid, img_hash).transact({'from': account})

print(f"\n--- Success! Data Recorded on Blockchain ---")
print(f"Transaction Hash: {tx_hash.hex()}")

# Verify it worked
record = contract.functions.getRecord(1).call()
print(f"Verified Record from Chain: {record}")
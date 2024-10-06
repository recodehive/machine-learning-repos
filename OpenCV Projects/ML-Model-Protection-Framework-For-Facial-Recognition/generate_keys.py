from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import base64
import os

# Generate a 256-bit (32-byte) AES key
key = os.urandom(32)  # AES-256 requires a 32-byte key
iv = os.urandom(16)   # AES block size is 16 bytes

# Encode key and IV in Base64 for easy storage in .env
key_b64 = base64.b64encode(key).decode('utf-8')
iv_b64 = base64.b64encode(iv).decode('utf-8')

# Print the results
print(f'AES_KEY={key_b64}')
print(f'AES_IV={iv_b64}')

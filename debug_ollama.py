import os
import requests

ollama_host = os.getenv('OLLAMA_HOST')
print(f"OLLAMA_HOST env var: '{ollama_host}'")

default_url = "http://localhost:11434"
target_url = ollama_host if ollama_host else default_url
print(f"Target URL: '{target_url}'")

try:
    print(f"Attempting GET {target_url} ...")
    resp = requests.get(target_url, timeout=5)
    print(f"Response: {resp.status_code}")
    print(resp.text[:100])
except Exception as e:
    print(f"Error: {e}")

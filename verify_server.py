import requests
import json
import time

def test_chat():
    url = "http://localhost:8001/chat"
    payload = {"message": "hello context test"}
    try:
        with requests.post(url, json=payload, stream=True, timeout=10) as r:
            if r.status_code == 200:
                print("Server OK. Response:")
                for chunk in r.iter_content(decode_unicode=True):
                    if chunk: print(chunk, end="", flush=True)
                print()
            else:
                print(f"Error: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    time.sleep(5) # Wait for server startup
    test_chat()

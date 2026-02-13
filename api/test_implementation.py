import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoints():
    print("Testing Backend Integration...")
    
    # 1. Product Search
    print("\n1. Testing Product Search...")
    try:
        res = requests.post(f"{BASE_URL}/api/products/search", json={"query": "steak"})
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
    except Exception as e:
        print(f"Product Search Failed: {e}")

    # 2. Chat with Context
    print("\n2. Testing Chat (Group of 4)...")
    payload = {
        "message": "we are a group of four people and we want to have dinner what options do you suggest",
        "user_id": "test_script_user",
        "context": "", 
        "user_metadata": {"name": "TestUser"}
    }
    
    try:
        # Note: Chat endpoint returns a stream
        res = requests.post(f"{BASE_URL}/chat", json=payload, stream=True)
        print(f"Status: {res.status_code}")
        print("Response Stream:")
        for chunk in res.iter_content(chunk_size=128):
            if chunk:
                print(chunk.decode(), end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Chat Request Failed: {e}")

if __name__ == "__main__":
    test_endpoints()

import requests
import json

url = "http://localhost:8000/chat"
payload = {
    "message": "I am hungry",
    "user_id": "test_user_123"
}

print(f"Sending request to {url}...")
try:
    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Request Failed: {e}")

import requests
import json

url = "http://localhost:8001/api/chat"
payload = {
    "message": "hello",
    "user_id": "test_user"
}
headers = {'Content-Type': 'application/json'}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload, headers=headers, stream=True)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("Response Content:")
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                print(chunk.decode('utf-8'), end='', flush=True)
        print("\n\n[End of Response]")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Connection Failed: {e}")

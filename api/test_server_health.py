import requests
import json

try:
    print("Testing /welcome endpoint...")
    res = requests.get("http://localhost:8000/welcome?user_id=test@example.com&name=TestUser")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    
    if res.status_code == 200:
        print("✅ Backend is responsive.")
    else:
        print("❌ Backend returned error.")
except Exception as e:
    print(f"❌ Connection Failed: {e}")

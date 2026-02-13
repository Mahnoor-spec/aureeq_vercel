import requests
import json

def test_chat(message):
    url = "http://localhost:8001/chat"
    payload = {"message": message}
    headers = {"Content-Type": "application/json"}
    
    print(f"\nTesting: '{message}'")
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        if response.status_code == 200:
            full_text = ""
            for chunk in response.iter_content(decode_unicode=True):
                if chunk:
                    full_text += chunk
            print(f"Response: {full_text}")
            print(f"Word Count: {len(full_text.split())}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_chat("i want pizza")
    test_chat("hi")
    test_chat("what do you have for dinner?")
    test_chat("i am in a rush, suggest something sweet")

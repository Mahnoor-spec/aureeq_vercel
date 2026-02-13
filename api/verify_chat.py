import requests
import json
import time

def verify_response():
    url = "http://localhost:8001/chat"
    payload = {
        "message": "I am hungry",
        "user_id": "test_user_123",
        "user_metadata": {
            "name": "Mahnoor",
            "email": "test@example.com"
        }
    }
    
    print("Waiting for server...")
    max_retries = 15
    for i in range(max_retries):
        try:
            print(f"Attempt {i+1}/{max_retries}...")
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            full_response = ""
            print("\n--- Response Stream ---")
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    text = chunk.decode("utf-8")
                    print(text, end="", flush=True)
                    full_response += text
            print("\n-----------------------")
            
            # Checks
            if "Agent:" in full_response:
                print("\n[FAIL] Response contains 'Agent:' prefix.")
            else:
                print("\n[PASS] Response does NOT contain 'Agent:' prefix.")
                
            if "Mahnoor" in full_response:
                print("[PASS] Response addresses user by name 'Mahnoor'.")
            else:
                print("[FAIL] Response does NOT address user by name 'Mahnoor'.")
            
            return # Success
            
        except requests.context_error as e: # Catch connection errors specifically if possible, but requests usually raises RequestException
             pass
        except Exception as e:
            if "refused" in str(e) or "establish a new connection" in str(e):
                 time.sleep(2)
                 continue
            print(f"\n[ERROR] Request failed: {e}")
            return

    print("\n[ERROR] Could not connect to server after multiple retries.")

if __name__ == "__main__":
    verify_response()

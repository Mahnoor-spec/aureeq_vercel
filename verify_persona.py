import requests
import time
import json

URL = "http://localhost:8001/chat"

def test_query(msg, expected_intent, expected_fragment=None):
    print(f"\n--- Testing: '{msg}' ---")
    start = time.time()
    try:
        accumulated = ""
        with requests.post(URL, json={"message": msg}, stream=True, timeout=20) as r:
            if r.status_code != 200:
                print(f"FAILED: {r.status_code}")
                return
            for chunk in r.iter_content(decode_unicode=True):
                if chunk:
                    print(chunk, end="", flush=True)
                    accumulated += chunk
        
        latency = time.time() - start
        print(f"\nLatency: {latency:.2f}s")
        
        if expected_fragment:
            if expected_fragment.lower() in accumulated.lower():
                print(f"PASS: Found expected fragment '{expected_fragment}'")
            else:
                print(f"FAIL: Expected '{expected_fragment}' not found.")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    print("Waiting for server startup...")
    time.sleep(5)
    
    # 1. Identity Check
    test_query("what is your restaurant name?", "restaurant_query", "IYI Dining")
    
    # 2. Who are you
    test_query("who are you?", "restaurant_query", "Aureeq")

    # 3. Non-Food Refusal (Strict)
    test_query("tell me about the weather", "non_food", "only able to help you with your food selections")
    
    # 4. Recommendation (Details)
    test_query("suggest a starter", "recommendation", "price")


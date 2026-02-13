import requests
import time
import json

URL = "http://localhost:8001/chat"

def test_query(msg, expected_intent, expected_fragment=None, unexpected_fragment=None):
    print(f"\n--- Testing: '{msg}' ---")
    start = time.time()
    try:
        accumulated = ""
        with requests.post(URL, json={"message": msg}, stream=True, timeout=30) as r:
            if r.status_code != 200:
                print(f"FAILED: {r.status_code}")
                return
            for chunk in r.iter_content(decode_unicode=True):
                if chunk:
                    print(chunk, end="", flush=True)
                    accumulated += chunk
        
        latency = time.time() - start
        print(f"\nLatency: {latency:.2f}s")
        print(f"Response: {accumulated}")
        
        if expected_fragment:
            if expected_fragment.lower() in accumulated.lower():
                print(f"PASS: Found expected fragment '{expected_fragment}'")
            else:
                print(f"FAIL: Expected '{expected_fragment}' not found.")
        
        if unexpected_fragment:
            if unexpected_fragment in accumulated:
                print(f"FAIL: Found unexpected fragment '{unexpected_fragment}'")
            else:
                print(f"PASS: Unexpected fragment '{unexpected_fragment}' absent.")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    print("Waiting for server startup...")
    time.sleep(10) # Give extra time for RAG load
    
    # 1. Out of Menu Item (Pizza) -> Should apologize but recommend something relevant
    test_query("I want to buy pizza", "food_interest", "sorry", "add to cart")

    # 2. Strict Non-Food (Helmet) -> Should refuse
    test_query("I want to buy a helmet", "food_interest", "only able to help you", "sorry, we don't offer")

    # 3. Valid Add to Cart -> Should work
    test_query("add ranjha gosht to cart", "add_to_cart", "[ORDER: Ranjha gosht |")

    # 4. Style Check (Soup) -> Should align with examples
    test_query("I want lentil soup", "food_interest", "recommend", "*") 


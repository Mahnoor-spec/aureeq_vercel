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
            if expected_fragment in accumulated:
                print(f"PASS: Found expected fragment '{expected_fragment}'")
            else:
                print(f"FAIL: Expected '{expected_fragment}' not found.")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    print("Waiting for server startup...")
    time.sleep(5)
    
    # Path A: Greeting (Deterministic)
    test_query("hello aureeq", "greeting", "Hello I am AUREEQ")
    
    # Path A: Non-Food (Deterministic)
    test_query("tell me about the weather", "non_food", "trained to help you with your food")
    
    # Path A: Add to Cart (Deterministic)
    test_query("add ranjha gosht to my cart", "add_to_cart", "||ADD_TO_CART:ranjha_gosht||")
    
    # Path B: Recommendation (OpenAI)
    test_query("I am hungry for something spicy", "food_interest", "spicy")
    
    # Path B: Recommendation (OpenAI)
    test_query("suggest a dessert", "recommendation", "Baklava")

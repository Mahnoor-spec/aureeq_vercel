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
    time.sleep(5)
    
    # 1. Lenient Spelling (The user's specific case)
    test_query("baklava is mentioned in your meu", "dish_query", "baklava", "only able to help you")

    # 2. Typos in food names
    test_query("describe chken wings", "dish_query", "chicken wings")

    # 3. Clean Menu List
    test_query("show me menu", "menu_query", "STARTER", "*") # * should be absent

    # 4. Upselling (Path B)
    test_query("suggest a dessert", "recommendation", "suggest", "sorry") # expecting a suggestion, maybe lenient check

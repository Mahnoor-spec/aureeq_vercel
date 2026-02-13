import requests
import json
import time

API_URL = "http://localhost:8000/chat"

test_cases = [
    {
        "name": "General Hunger",
        "message": "i am hungry",
        "expected": "Greeting + Hunger Acknowledgement + Suggestions"
    },
    {
        "name": "Missing Item (Pizza)",
        "message": "i want to order pizza",
        "expected": "Greeting + Pizza Recognition + Meat Specialization + Lahmacun Suggestion"
    },
    {
        "name": "Budget (Couple, 30 GBP)",
        "message": "we are a couple and want to have a dinner within the budget of 30",
        "expected": "Greeting + Budget Recognition + Combo Suggestion around 30"
    },
    {
        "name": "Unavailable Category (Breakfast)",
        "message": "i want to order something for breakfast",
        "expected": "Greeting + Breakfast Recognition + Unavailability Explanation + Anytime Specials"
    },
    {
        "name": "Missing Item (Sushi)",
        "message": "do you have sushi?",
        "expected": "Greeting + Sushi Recognition + Specialized in Meat + Lamb Chops Suggestion"
    }
]

def run_tests():
    print(f"Starting Rigorous Testing on {API_URL}...\n")
    
    for i, case in enumerate(test_cases):
        print(f"--- Test Case {i+1}: {case['name']} ---")
        print(f"Input: {case['message']}")
        
        payload = {
            "message": case["message"],
            "user_id": "test_user_rigorous",
            "user_metadata": {"name": "Mahnoor", "email": "mahnoor@example.com"}
        }
        
        try:
            # Note: /chat returns a StreamingResponse, so we accumulate the chunks
            response = requests.post(API_URL, json=payload, stream=True)
            
            if response.status_code != 200:
                print(f"FAILED: Status {response.status_code}")
                continue
            
            full_response = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    full_response += chunk.decode('utf-8')
            
            print(f"AI Response:\n{full_response.strip()}")
            print(f"Verification Note: Check if structure matches Greeting -> Context -> Suggestion -> CTA.")
            
        except Exception as e:
            print(f"ERROR: {e}")
            print("Make sure server.py is running on port 8000.")
            break
            
        print("-" * 40 + "\n")
        time.sleep(1) # Small delay between tests

if __name__ == "__main__":
    run_tests()

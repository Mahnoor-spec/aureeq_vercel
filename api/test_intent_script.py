
import sys
import os
# Adjust path to import server
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from server import classify_intent, MENU_DATA

# Mock MENU_DATA if needed, but imported one should be populated if it runs at module level?
# Actually MENU_DATA is populated in startup_event which is async.
# So I need to populate it manually for test.
import json
with open(os.path.join(current_dir, "../data/menu.json"), "r", encoding="utf-8") as f:
    data = json.load(f)
    for item in data:
        MENU_DATA.append(item)

test_cases = [
    "add to cart ranjha",
    "i want to buy pizza",
    "tell me about ranjha",
    "add to cart helmet",
    "hello",
    "restaurant name"
]

for msg in test_cases:
    intent = classify_intent(msg)
    print(f"'{msg}' -> {intent}")

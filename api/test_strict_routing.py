from server import classify_intent, route_query

test_cases = [
    ("Hello there!", "greeting", "fast"),
    ("I'm so hungry right now", "food_interest", "fast"),
    ("What's on the menu?", "menu_query", "fast"),
    ("Tell me about the Ribeye Steak", "dish_query", "fast"),
    ("Add a salad to my cart", "add_to_cart", "fast"),
    ("Recommend a good dessert", "recommendation", "reasoning"),
    ("Suggest something spicy", "recommendation", "reasoning"),
    ("Something light for lunch", "recommendation", "reasoning"),
    ("What's the weather like?", "non_food", "reject"),
    ("Can you help me with coding?", "non_food", "reject"),
]

print(f"{'Input':<30} | {'Expected Intent':<15} | {'Actual Intent':<15} | {'Route':<10} | {'Status'}")
print("-" * 90)

for text, exp_intent, exp_route in test_cases:
    actual_route, actual_intent = route_query(text)
    status = "✅" if actual_intent == exp_intent and actual_route == exp_route else "❌"
    print(f"{text:<30} | {exp_intent:<15} | {actual_intent:<15} | {actual_route:<10} | {status}")

import re

def classify_intent(text):
    text = text.lower()
    
    def has_word(word, text):
        return re.search(r'\b' + re.escape(word) + r'\b', text) is not None

    # 7. Non-Food - Highest priority for rejection
    non_food_keywords = ["weather", "news", "coding", "programming", "politics", "joke", "sports", "movie", "technology", "stock", "finance"]
    if any(has_word(w, text) for w in non_food_keywords):
        return "non_food"

    # 1. Greeting
    if any(has_word(w, text) for w in ["hello", "hi", "hey", "greetings", "morning", "evening"]):
        return "greeting"
    
    # 2. Food Interest (Critical Rule)
    food_interest_keywords = ["hungry", "starving", "craving", "want to eat", "need food", "feel like eating", "something to eat", "what should i eat"]
    if any(w in text for w in food_interest_keywords): # Generic phrases ok for substring
        return "food_interest"
    
    # 5. Add to Cart
    if any(w in text for w in ["add", "cart", "buy", "order"]) and any(w in text for w in ["cart", "order", "this"]):
        if "add" in text or "buy" in text or "order" in text:
            return "add_to_cart"
    
    # 6. Recommendation
    recommend_keywords = ["recommend", "suggest", "best", "try", "sweet", "light", "spicy", "dessert", "main", "drink"]
    if any(has_word(w, text) for w in recommend_keywords):
        return "recommendation"

    # 4. Dish Query / 3. Menu Query
    if has_word("menu", text) or has_word("options", text) or has_word("offer", text):
        return "menu_query"
    
    # Default fallback
    return "dish_query"

def route_query(text):
    intent = classify_intent(text)
    
    if intent == "non_food":
        return "reject", "non_food"
    
    if intent == "recommendation":
        return "reasoning", "recommendation"
    
    return "fast", intent

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
    status = "PASS" if actual_intent == exp_intent and actual_route == exp_route else "FAIL"
    print(f"{text:<30} | {exp_intent:<15} | {actual_intent:<15} | {actual_route:<10} | {status}")

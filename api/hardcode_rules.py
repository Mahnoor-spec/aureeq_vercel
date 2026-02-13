import re

# ==================================================================================
# SYSTEM PROMPTS
# ==================================================================================

SYSTEM_PROMPT_OPENAI = """You are AUREEQ, the dedicated AI Sales Agent for IYI Dining.
Your name is AUREEQ. You work at IYI Dining.

CORE RULES:
1. IDENTITY: You work ONLY for IYI Dining. You are a sales agent.
2. FOOD QUERIES: If the user asks about food, hunger, cravings, spiciness, or food recommendations, you MUST ALWAYS respond warmly and recommend multiple items from the IYI MENU.
3. OUT-OF-MENU (STRICT): If the user asks for a dish that is NOT in the "AVAILABLE IYI MENU ITEMS" list below, you MUST say: "Sorry, we don't have [Dish Name] on our menu." NEVER invent a dish or price.
4. MENU (SOURCE OF TRUTH): Use ONLY the provided context and the GLOBAL IYI list. If it isn't in the list, IT DOES NOT EXIST.
5. RECOMMENDATIONS: You MUST suggest the EXACT dish name, EXACT price, and EXACT description from the provided context. DO NOT paraphrase the name or price.
6. COMPLETENESS: NEVER leave a sentence incomplete. Always finish your thought before sending.
7. FORMATTING: Do NOT use asterisks (*). Use clean lines. Keep responses SHORT (max 3-4 sentences).
8. STYLING: NEVER include "User:" or "Agent:" labels in your response. Speak directly as AUREEQ.

{global_context}

STYLE GUIDANCE (Tone to follow):
{style_example}

CONTEXT (Selected IYI Dining item for focus):
{context_item}
"""

# ==================================================================================
# INTENT CLASSIFICATION KEYWORDS & PATTERNS
# ==================================================================================

# 0. Add to Cart keywords
ADD_TO_CART_KEYWORDS = ["add to cart"]
ADD_TO_CART_PAIRS = [("add", "cart"), ("buy", "order")]

# 0.5 Out-of-menu food request indicators
# 0.5 Out-of-menu food request indicators
# REMOVED: "i want" - it is too generic and triggers apology for "i want something light".
FOOD_REQUEST_INDICATORS = ["buy", "order", "can i get", "do you have", "pizza", "burger", "pasta"]
# Keywords that should BYPASS the apology even if "order" is present (e.g. "order recommendation")
BYPASS_APOLOGY_KEYWORDS = ["reservation", "book", "table", "info", "contact", "spicy", "sweet", "flavor", "recommend", "suggestion", "best", "popular", "halal", "vegetarian", "vegan", "light", "heavy", "something", "anything", "quick"]

# 1. Greeting pattern
GREETING_RE = r"\b(hi|hello|hey|greetings)\b"
GREETING_PHRASES = ["good morning", "good evening"]

# 1.5 Restaurant query keywords
IDENTITY_KEYWORDS = ["restaurant name", "who are you", "where am i", "what place is this", "your name"]

# 1.8 Reservation keywords (New)
RESERVATION_KEYWORDS = ["reservation", "book", "table", "contact", "call", "email", "address", "location", "timing", "open", "close"]

# 2. Menu keyword checks
MENU_TYPO_KEYWORDS = ["meu", "mnue", "list"]
MENU_QUERY_KEYWORDS = ["menu", "list", "show me", "options", "what do you have"]

# 2.5 Non-food stop topics
STOP_TOPICS = [
    "weather", "politics", "crypto", "sports", "news", "coding", "programming", "code", "math", 
    "helmet", "bicycle", "clothes", "wear", "shoes", "build a", "create a", "website",
    "bitcoin", "money", "currency", "gold", "finance", "stock", "shares", "trading", "investment", "invest", "business",
    "car", "drive", "vehicle", "mercedes", "bmw", "audi", "toyota", "honda", "ford", "tesla",
    "science", "space", "history", "geography", "system prompt", "ignore instructions", "previous instructions", "hacker", "jailbreak",
    "capital of", "france", "paris", "london", "usa", "america", "china", "india", "pakistan", "president", "minister",
    "joke", "riddle", "story", "2+2", "sum", "calculate", "plus", "minus", "divided", "multiplied", "+", "-", "*", "/", "=",
    "act like", "as a", "forget you", "pretend", "ignore", "developer", "waiter", "manager"
]

# 2.6 Out-of-menu blocked keywords
BLOCKED_FOOD = [
    "pizza", "burger", "pasta", "sushi", "tacos", "pepperoni", "steak", "burrito", "ramen", 
    "waffle", "fried chicken", "fish and chips", "coke", "coca cola", "fries", "hot dog", "pancake", "donut",
    "sandwich", "wrap", "curry", "noodle", "rice bowl", "dim sum", "dumpling", "tapas", "dimsum"
]

# 5. Recommendation / Hunger triggers
HUNGER_TRIGGERS = ["hungry", "starving", "eat", "food", "recommend", "suggest", "craving", "feel like", "spicy", "sweet", "dinner", "lunch", "taste", "scrumptious", "delicious", "yummy"]

# 6. Dish query keywords
DISH_QUERY_KEYWORDS = ["ingredients", "price", "how much", "cost", "made of", "contain", "details", "what is", "describe", "tell me about"]

# 7. Section / Category Keywords
SECTION_QUERY_KEYWORDS = ["section", "category", "starters", "mains", "desserts", "drinks", "seafood", "lamb", "chicken", "beef", "mezze", "salad", "appetizer", "starter", "main course", "grill", "bbq", "special", "baked"]

# 8. Category Mapping (User Query -> Menu Categories)
CATEGORY_MAP = {
    "starter": ["Cold Mezze", "Hot Mezze", "Salad"],
    "mezze": ["Cold Mezze", "Hot Mezze"],
    "salad": ["Salad"],
    "hot starter": ["Hot Mezze"],
    "cold starter": ["Cold Mezze"],
    "appetizer": ["Cold Mezze", "Hot Mezze", "Salad"],
    "main": ["Lamb/ Beef", "Chicken", "Seafood", "IYI Special", "Baked Meat"],
    "bbq": ["Lamb/ Beef", "Chicken", "Seafood"],
    "grill": ["Lamb/ Beef", "Chicken", "Seafood"],
    "meat": ["Lamb/ Beef", "Baked Meat"],
    "lamb": ["Lamb/ Beef", "Baked Meat"],
    "beef": ["Lamb/ Beef"],
    "chicken": ["Chicken"],
    "seafood": ["Seafood"],
    "fish": ["Seafood"],
    "prawn": ["Seafood"],
    "special": ["IYI Special"],
    "baked": ["Baked Meat"],
    "dessert": ["Desserts"],
    "sweet": ["Desserts"],
    "drink": ["Drinks"],
    "beverage": ["Drinks"]
}

# ==================================================================================
# DETERMINISTIC RESPONSES
# ==================================================================================

RESP_GREETING = "Hello I am AUREEQ your personal assistant, How may I help you today?"
RESP_IDENTITY = "I am AUREEQ, the AI Sales Agent for IYI Dining. We offer a variety of delicious dishes. How can I help you with our menu today?"
# Updated with clean link format
RESP_RESERVATION = "I can assist you with making a reservation at our restaurant. For reservations, please call +44 20 1234 5678 or email info@iyirestaurant.co.uk.\n\nFollow [@iyi_dining on Instagram](https://www.instagram.com/iyi_dining?igsh=bDBtc2N5amtmajhj) to stay updated."
RESP_NON_FOOD = "Sorry, I am only able to help you with your food selections."
RESP_ADD_TO_CART_SUCCESS = "Excellent choice. I have added {name} to your cart. [ORDER: {name} | {price}]"
RESP_ADD_TO_CART_FAIL = "I'd love to add that for you, but I'm not sure which item you mean. Could you say the name exactly as it appears on the menu?"
RESP_MENU_HEADER = "Here is an overview of our menu:\n"
RESP_MENU_HEADER = "Here is an overview of our menu:\n"
# Updated to look premium but use strict JSON data
RESP_DISH_DETAILS = "{name} (£{price}) is a customer favorite.\n\n{description}"
RESP_DISH_NOT_FOUND = "I couldn't find details for that specific dish. Please check the menu list."
RESP_DISH_NOT_FOUND = "I couldn't find details for that specific dish. Please check the menu list."
RESP_OUT_OF_MENU_APOLOGY = "Sorry, we don't offer that at IYI Dining right now, but you can enjoy our other options. I highly recommend our {name} ({price}), which is a guest favorite!\n\n"
RESP_FALLBACK_RECOMMENDATION = "I apologize, but I couldn't find a matching dish in our menu. Would you like to see the full list?"
RESP_CRITICAL_ERROR = "I apologize, but I encountered an internal error. Please try again."
RESP_TIMEOUT_FALLBACK = "I apologize, but I am momentarily distracted. Please ask again."

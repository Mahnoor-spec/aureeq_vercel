import os
import json
import sys
import re
import asyncio
import traceback
import sqlite3
import uuid
import difflib
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import numpy as np
import edge_tts
from simple_rag import SimpleExampleRAG
import hardcode_rules as rules

# ==================================================================================
# CONFIGURATION
# ==================================================================================

# Helper to get absolute path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_HOST_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_EMBED = "nomic-embed-text"
SQLITE_PATH = os.path.join(BASE_DIR, "aureeq.db")
MENU_JSON_PATH = os.path.join(DATA_DIR, "menu.json")
EXAMPLES_TXT_PATH = os.path.join(DATA_DIR, "sales_examples_new.txt")

# Audio output for Vercel
IS_VERCEL = "VERCEL" in os.environ
AUDIO_OUTPUT_DIR = "/tmp" if IS_VERCEL else DATA_DIR


# Global State
MENU_DATA: List[Dict] = []
MENU_VECTORS: Optional[np.ndarray] = None
HTTP_CLIENT: Optional[httpx.AsyncClient] = None
EXAMPLE_RAG: Optional[SimpleExampleRAG] = None
INIT_LOCK = asyncio.Lock()
IS_INITIALIZED = False
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Prompts are now loaded from hardcode_rules.py
SYSTEM_PROMPT_OPENAI = rules.SYSTEM_PROMPT_OPENAI

# ==================================================================================
# UTILITIES & DATABASE
# ==================================================================================

def log(msg):
    try:
        print(msg.encode('ascii', 'replace').decode('ascii'))
    except:
        print(msg)
    sys.stdout.flush()

def get_db_connection():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            name TEXT,
            preferences TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            items TEXT,
            total_price REAL,
            status TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_order(user_id: str, items: list, total_price: float):
    conn = get_db_connection()
    cursor = conn.cursor()
    order_id = str(uuid.uuid4())[:8]
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO orders (id, user_id, items, total_price, status, created_at) VALUES (?, ?, ?, ?, ?, ?)", (order_id, user_id, json.dumps(items), total_price, "Completed", timestamp))
        conn.commit()
        return order_id
    except Exception as e:
        log(f"DB Error (save_order): {e}")
        return None
    finally: conn.close()

# ==================================================================================
# DATA LOADING & VECTOR SEARCH (RAG)
# ==================================================================================

async def get_ollama_embedding(text: str):
    try:
        resp = await HTTP_CLIENT.post(
            f"{OLLAMA_HOST_URL}/api/embeddings",
            json={"model": MODEL_EMBED, "prompt": text},
            timeout=5.0
        )
        if resp.status_code == 200:
            return resp.json().get("embedding")
    except Exception as e:
        log(f"Embedding ID Error: {e}")
    return None

async def init_data():
    global MENU_DATA, MENU_VECTORS, IS_INITIALIZED, HTTP_CLIENT
    if IS_INITIALIZED:
        return
    
    async with INIT_LOCK:
        if IS_INITIALIZED:
            return
            
        if HTTP_CLIENT is None:
            HTTP_CLIENT = httpx.AsyncClient(timeout=60.0)

        if not os.path.exists(MENU_JSON_PATH):
            log("ERROR: menu.json not found!")
            return

        try:
            with open(MENU_JSON_PATH, "r", encoding="utf-8") as f:
                MENU_DATA = json.load(f)
            log(f"Loaded {len(MENU_DATA)} menu items.")
            
            # Pre-compute vectors if possible, or computing on demand (batching preferred)
            # For simplicity/speed in this restart, we'll embed on demand or do a quick batch if small
            # Actually, let's do sequential async init for reliability
            vectors = []
            log("Initializing Embeddings...")
            for item in MENU_DATA:
                text = f"{item['name']} {item['description']} {item['category']}"
                vec = await get_ollama_embedding(text)
                if vec:
                    vectors.append(vec)
                else:
                    vectors.append([0.0]*768) # Fallback
            
            from sklearn.metrics.pairwise import cosine_similarity
            MENU_VECTORS = np.array(vectors)
            log("Embeddings Initialized.")
            
        except Exception as e:
            log(f"Data Init Error: {e}")

        # Initialize Example RAG
        global EXAMPLE_RAG
        EXAMPLE_RAG = SimpleExampleRAG(EXAMPLES_TXT_PATH, get_ollama_embedding)
        await EXAMPLE_RAG.load_examples()
        IS_INITIALIZED = True

async def get_nearest_item(query: str):
    if MENU_VECTORS is None or not MENU_DATA:
        return None
    
    query_vec = await get_ollama_embedding(query)
    if not query_vec:
        return None
        
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity([query_vec], MENU_VECTORS)[0]
    best_idx = np.argmax(scores)
    
    # Threshold check? strictness requires relevance.
    # returning top 1 unconditionally as per strict interaction rules for Path B
    return MENU_DATA[best_idx]

# ==================================================================================
# INTENT ROUTER & PATH LOGIC
# ==================================================================================

def classify_intent(msg: str) -> str:
    msg = msg.lower()
    
    # 0. STRICT ADD TO CART CHECK (Priority)
    if any(k in msg for k in rules.ADD_TO_CART_KEYWORDS) or any(all(w in msg for w in p) for p in rules.ADD_TO_CART_PAIRS):
        item = find_item_lenient(msg)
        if item:
            return "add_to_cart"

    # 1. GREETING
    if re.search(rules.GREETING_RE, msg) or any(p in msg for p in rules.GREETING_PHRASES):
        return "greeting"

    # 1.5 RESTAURANT QUERY (Identity)
    if any(x in msg for x in rules.IDENTITY_KEYWORDS):
        return "restaurant_query"
        
    # 1.8 RESERVATION QUERY
    if any(x in msg for x in rules.RESERVATION_KEYWORDS):
        return "reservation_query"
        
    # --- HIGH PRIORITY MATCHES ---
    
    # 0.4 ADVERSARIAL / ROLEPLAY / SYSTEM CHECK (Strict)
    if any(x in msg for x in ["ignore", "forget", "pretend", "act like", "as a ", "system prompt", "internal rules"]):
        return "non_food"

    # 0.5 OUT-OF-MENU CHECK (High Priority)
    if any(x in msg for x in rules.BLOCKED_FOOD):
        # Allow if it's a specific question about an EXISTING item (e.g., "Tell me about Chicken Wings" even if 'chicken' is blocked - wait, don't block 'chicken')
        # We only block specific missing dishes.
        has_real_item = any(i['name'].lower() in msg for i in MENU_DATA)
        if not has_real_item:
            return "out_of_menu"
    
    # Check for Specific Dish match first (High Priority)
    # If the user names a specific dish (e.g. "Lamb Chops"), we treat as food_interest for Path B.
    potential_item = find_item_lenient(msg)
    if potential_item:
        # Check if it's a specific question about the item
        if any(x in msg for x in rules.DISH_QUERY_KEYWORDS):
            return "dish_query"
        return "food_interest"

    # SECTION/CATEGORY QUERY (e.g. "Show me seafood")
    # Check strict mapping first
    for cat_key in rules.CATEGORY_MAP.keys():
        if re.search(r'\b' + re.escape(cat_key) + r's?\b', msg):
            return "section_query"
            
    if any(x in msg for x in rules.SECTION_QUERY_KEYWORDS):
        return "section_query"

    # MENU QUERY (List)
    if any(x in msg for x in rules.MENU_QUERY_KEYWORDS) or any(x in msg for x in rules.MENU_TYPO_KEYWORDS):
        return "menu_query"

    # RECOMMENDATION / HUNGER / CRAVING / TASTE
    if any(x in msg for x in rules.HUNGER_TRIGGERS):
        return "food_interest"
    
    # 2.5 NON-FOOD (Strict)
    if any(x in msg for x in rules.STOP_TOPICS):
        return "non_food"

    # Default fallback
    return "food_interest"

def find_item_fuzzy(name_query: str):
    # simple substring match
    name_query = name_query.lower()
    for item in MENU_DATA:
        if item['name'].lower() in name_query:
            return item
    return None

def find_item_lenient(query: str):
    # 1. Exact/Substring match first (e.g. "I want Baklava")
    exact = find_item_fuzzy(query)
    if exact: return exact
    
    query_lower = query.lower()
    keywords = query_lower.split()
    all_names = [i['name'].lower() for i in MENU_DATA]
    
    # 2. DISABLED: Single word match is too loose (e.g. "Gosht" matches "Ranjha Gosht" even for "Achar Gosht")
    # Instead, we rely on fuzzy matching of the WHOLE PHRASE first.
    pass

    # Remove Common Question/Order Words for better fuzzy match
    STOP_WORDS = ["what", "is", "describe", "tell", "me", "about", "i", "want", "to", "order", "get", "buy", "a", "an", "the", "price", "how", "much", "ingredients"]
    query_clean = query_lower
    for w in STOP_WORDS:
        query_clean = query_clean.replace(w, "").strip() # naive replace but works for whole words usually if padded
        # Better: split and filter
    
    query_words = [w for w in query_lower.split() if w not in STOP_WORDS and len(w) > 2]
    query_clean_str = " ".join(query_words)

    # 3. Difflib fuzzy match on full CLEANED query
    # E.g. Query="what is prawn tikka" -> Clean="prawn tikka" -> Fuzzy Match="Prawns Tikka" (Success)
    if query_clean_str:
        matches = difflib.get_close_matches(query_clean_str, all_names, n=1, cutoff=0.75) # slightly strict but allows plural diffs
        if matches:
            return next((i for i in MENU_DATA if i['name'].lower() == matches[0]), None)

    # 4. Fallback on original query if cleaned failed
    matches = difflib.get_close_matches(query_lower, all_names, n=1, cutoff=0.8)

    # 4. Difflib on keywords (Typos in partial names)
    # 4. Difflib on keywords (Typos in partial names) - Also increased cutoff
    for word in keywords:
        if len(word) > 4: # increased min length to 4
            matches = difflib.get_close_matches(word, all_names, n=1, cutoff=0.8) # increased from 0.7
            if matches:
                return next((i for i in MENU_DATA if i['name'].lower() == matches[0]), None)
                
    return None

# ==================================================================================
# API ENDPOINTS
# ==================================================================================

class ChatRequest(BaseModel):
    message: str
    user_id: str = None
    user_metadata: dict = None

class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-ChristopherNeural"

@app.on_event("startup")
async def startup_event():
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(timeout=60.0)
    init_db()
    # Removed asyncio.create_task(init_data()) as it's now lazy-loaded

@app.on_event("shutdown")
async def shutdown_event():
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()

@app.post("/chat")
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not IS_INITIALIZED:
        await init_data()
    user_query = request.message
    log(f"--- Chat: {user_query} ---")
    
    async def response_stream():
        try:
            intent = classify_intent(user_query)
            log(f"Intent: {intent}")

            # --- PATH A: DETERMINSTIC (No LLM) ---
            
            if intent == "greeting":
                yield rules.RESP_GREETING
                return
                
            if intent == "restaurant_query":
                yield rules.RESP_IDENTITY
                return
                
            if intent == "reservation_query":
                yield rules.RESP_RESERVATION
                return

            if intent == "non_food":
                yield rules.RESP_NON_FOOD
                return
                
            if intent == "add_to_cart":
                found_item = find_item_lenient(user_query)
                if found_item:
                    yield rules.RESP_ADD_TO_CART_SUCCESS.format(name=found_item['name'], price=found_item['price'])
                else:
                    yield rules.RESP_ADD_TO_CART_FAIL
                return

            if intent == "menu_query":
                # User asked for WHOLE MENU. Show all details? User said "show them the whole menu".
                # To be safe but compliant, we list ALL headings and items.
                output = [rules.RESP_MENU_HEADER]
                categories = sorted(list(set(i['category'] for i in MENU_DATA)))
                for cat in categories:
                    items = [i for i in MENU_DATA if i['category'] == cat]
                    output.append(f"\n--- {cat.upper()} ({len(items)} dishes) ---")
                    # List Name + Price only for full menu to keep it readable, unless user asked for details?
                    # User request: "if asked for the menu show them the whole menu"
                    # We will show Name + Price for brevity but cover everything.
                    for item in items:
                        output.append(f"- {item['name']} (£{item['price']})")
                
                yield "\n".join(output)
                return

            if intent == "section_query":
                # Identify which categories to show
                target_cats = []
                msg_lower = user_query.lower()
                
                # Check mapping
                for key, cats in rules.CATEGORY_MAP.items():
                    if re.search(r'\b' + re.escape(key) + r's?\b', msg_lower):
                        target_cats.extend(cats)
                
                # Uniquify
                target_cats = list(set(target_cats))
                
                if not target_cats:
                   # Fallback: Treat as full menu or error?
                   yield "I'm not sure which section you mean. We have Starters, BBQ, Specials, Desserts, and Drinks."
                   return

                output = []
                for cat in target_cats:
                    items = [i for i in MENU_DATA if i['category'] == cat]
                    if items:
                        output.append(f"\n--- {cat.upper()} ({len(items)} dishes) ---")
                        # User Rule: "show them the all dishes with exact price and description listed their"
                        for item in items:
                            output.append(f"\n* {item['name']} ({item['price']})\n  {item['description']}")
                
                if not output:
                     yield "I couldn't find any items in that section."
                else:
                     yield "\n".join(output)
                return

            if intent == "out_of_menu":
                # Get a high-quality recommendation
                rec = next((i for i in MENU_DATA if "Lamb Chops" in i['name']), MENU_DATA[0])
                yield rules.RESP_OUT_OF_MENU_APOLOGY.format(name=rec['name'], price=rec['price'])
                return

            if intent == "dish_query":
                item = find_item_lenient(user_query)
                if item:
                    yield rules.RESP_DISH_DETAILS.format(name=item['name'], price=item['price'], description=item['description'])
                else:
                    yield rules.RESP_DISH_NOT_FOUND
                return

            # --- DETERMINISTIC OUT-OF-MENU (Consolidated) ---
            # If the user is specifically ordering/asking for a dish we DON'T have
            is_specific_request = any(x in user_query.lower() for x in rules.FOOD_REQUEST_INDICATORS)
            is_bypass = any(x in user_query.lower() for x in rules.BYPASS_APOLOGY_KEYWORDS)
            item_exists = find_item_lenient(user_query)
            
            if is_specific_request and not item_exists and not is_bypass:
                # Get a high-quality recommendation (RAG would be overkill here, use popular items)
                rec = next((i for i in MENU_DATA if "Lamb Chops" in i['name']), MENU_DATA[0])
                yield rules.RESP_OUT_OF_MENU_APOLOGY.format(name=rec['name'], price=rec['price'])
                return

            # --- PATH B: REASONING (OpenAI + RAG) ---
            
            if intent == "food_interest" or intent == "recommendation":
                # 0. STRICT DESSERT/ITEM QUERY Check
                # If the user mentions a specific item we HAVE, serve definition deterministically.
                # This prevents "Prawn Tikka Masala" hallucinations.
                specific_item_match = find_item_lenient(user_query)
                if specific_item_match:
                    yield rules.RESP_DISH_DETAILS.format(name=specific_item_match['name'], price=specific_item_match['price'], description=specific_item_match['description'])
                    yield "\n\nWould you like to add it to your order?"
                    return

                # 1. Try Lenient Search First (Specific Item Focus) -- redundant now but keep for nearest logic
                nearest_item = None # Reset
                
                # 2. If no specific item found, use RAG Retrieval
                nearest_item = await get_nearest_item(user_query)
                
                # 3. Retrieve Style Example
                style_example = ""
                if EXAMPLE_RAG and EXAMPLE_RAG.is_ready:
                    style_example = await EXAMPLE_RAG.retrieve(user_query)

                if not nearest_item:
                     # If truly NOTHING found even via RAG (rare), fallback to list
                    yield rules.RESP_FALLBACK_RECOMMENDATION
                    return
                    
                # 4. OpenAI Generation
                context_str = json.dumps(nearest_item, indent=2)
                
                # 3. Global IYI Context (Dish names) to ensure recommendation
                global_context = "AVAILABLE IYI MENU ITEMS: " + ", ".join([i['name'] for i in MENU_DATA])
                
                # 4. Final System Message Construction
                additional_warning = ""
                nearest_item_context = nearest_item.copy() if nearest_item else {}
                
                # Check: Is the user asking for X but we are giving context for Y?
                # Deterministic Out-of-Menu Checks
                is_specific_request = any(x in user_query.lower() for x in rules.FOOD_REQUEST_INDICATORS)
                is_bypass = any(x in user_query.lower() for x in rules.BYPASS_APOLOGY_KEYWORDS)
                
                # Check: Is the user asking for X but we are giving context for Y?
                # NOTE: We use find_item_lenient(user_query) to check if the specific item exists.
                # If not, but we have a `nearest_item` (from RAG), we MUST warn the AI.
                item_exists_deterministic = find_item_lenient(user_query)
                
                if is_specific_request and not item_exists_deterministic:
                    additional_warning = "" 
                    # STRICT MASKING: Do not even mention the missing item to the LLM. 
                    # The apology handled the "not found" part.
                    # We only want the description of the RECOMMENDATION.
                    system_msg = SYSTEM_PROMPT_OPENAI.format(
                        context_item=json.dumps(nearest_item_context, indent=2),
                        style_example=style_example,
                        global_context=global_context
                    )
                    # Rewrite the User Message for the LLM context to focus ONLY on the available item
                    user_query_for_llm = f"Describe '{nearest_item['name']}' ({nearest_item['price']}) enthusiastically. Do not mention availability."
                else:
                    system_msg = SYSTEM_PROMPT_OPENAI.format(
                        context_item=json.dumps(nearest_item_context, indent=2),
                        style_example=style_example,
                        global_context=global_context
                    )
                    user_query_for_llm = user_query
                
                if is_specific_request and not item_exists_deterministic and not is_bypass:
                    # The consolidated apology was already yielded before Path B.
                    # We only proceed to Path B to provide the enthusiastic description of the recommendation.
                    pass

                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_query_for_llm}
                ]
                
                try:
                    async with HTTP_CLIENT.stream(
                        "POST", 
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                        json={
                            "model": "gpt-4o-mini", 
                            "messages": messages,
                            "stream": True,
                            "temperature": 0.7,
                            "max_tokens": 600 # Increased further to ensure completion (User Request)
                        }, timeout=20.0
                    ) as response:
                        # ... processing loop ...
                        if response.status_code != 200:
                            # ... exception ...
                            raw_err = await response.aread()
                            err = raw_err.decode('utf-8', errors='replace')
                            raise Exception(f"OpenAI API Error: {response.status_code} - {err}")

                        async for line in response.aiter_lines():
                            if not line.strip(): continue
                            if line.startswith("data: [DONE]"): break
                            if line.startswith("data: "):
                                try:
                                    chunk = json.loads(line[6:])
                                    content = chunk["choices"][0]["delta"].get("content", "")
                                    if content: yield content
                                except: pass

                except Exception as e:
                    log(f"OpenAI Stream Error: {e}")
                    # Fallback to Local Llama-3.2:1b
                    log("Falling back to local Llama-3.2:1b...")
                    # ... fallback logic similar update ...
                    try:
                        # Inject simplified instructions for local model
                        async with HTTP_CLIENT.stream("POST", f"{OLLAMA_HOST_URL}/api/chat", json={
                            "model": "llama3.2:1b",
                            "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_query_for_llm}],
                            "stream": True,
                            "options": {"temperature": 0.3, "num_predict": 600} # Increased to ensure completion
                        }, timeout=30.0) as resp:
                             async for line in resp.aiter_lines():
                                if not line: continue
                                chunk = json.loads(line)
                                content = chunk.get("message", {}).get("content", "")
                                if content: yield content
                    except Exception as local_e:
                        log(f"Local Fallback Error: {local_e}")
                        yield rules.RESP_TIMEOUT_FALLBACK
                return
        except Exception as e:
            log(f"CRITICAL STREAM ERROR: {e}")
            log(traceback.format_exc())
            yield rules.RESP_CRITICAL_ERROR

    return StreamingResponse(response_stream(), media_type="text/plain")

# ==================================================================================
# TTS / AUDIO ENDPOINTS
# ==================================================================================

@app.post("/tts")
@app.post("/api/tts")
async def tts_endpoint(request: TTSRequest):
    if not IS_INITIALIZED:
        await init_data()
    try:
        text = request.text
        voice = request.voice
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        # Generate unique filename based on text hash or uuid
        # For simplicity, use uuid
        filename = f"tts_{uuid.uuid4()}.mp3"
        filepath = os.path.join(AUDIO_OUTPUT_DIR, filename)
        
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filepath)
        
        return {"audio_url": f"/api/audio/{filename}"}
    except Exception as e:
        log(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/welcome")
@app.get("/api/welcome")
async def welcome_endpoint(name: str = "Guest", user_id: str = None):
    if not IS_INITIALIZED:
        await init_data()
    text = f"Hello {name}, I am AUREEQ your personal assistant. How may I help you today?"
    try:
        # Check for cached welcome message? 
        # For dynamic names, we generate fresh.
        filename = f"welcome_{uuid.uuid4()}.mp3"
        filepath = os.path.join(AUDIO_OUTPUT_DIR, filename)
        
        communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
        await communicate.save(filepath)
        
        return {"response": text, "audio_url": f"/api/audio/{filename}"}
    except Exception as e:
        log(f"Welcome TTS Error: {e}")
        return {"response": text, "audio_url": None}

app.mount("/api/audio", StaticFiles(directory=AUDIO_OUTPUT_DIR), name="audio_api")
app.mount("/audio", StaticFiles(directory=AUDIO_OUTPUT_DIR), name="audio_root")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

import httpx
import time
import asyncio

async def test_chat(query):
    url = "http://127.0.0.1:8001/chat"
    payload = {
        "message": query,
        "user_metadata": {"name": "Test User", "email": "test@example.com"}
    }
    
    start_time = time.time()
    print(f"\nTESTING: '{query}'")
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            first_token_time = None
            async with client.stream("POST", url, json=payload) as response:
                async for chunk in response.aiter_text():
                    if not first_token_time:
                        first_token_time = time.time()
                        print(f"[Latency to first token: {first_token_time - start_time:.2f}s]")
                    print(chunk, end="", flush=True)
                print()
    except Exception as e:
        print(f"Error: {e}")

async def main():
    queries = [
        "hi aureeq",
        "how is Ranjha Gosht made?",
        "what do you have for dinner?"
    ]
    for q in queries:
        await test_chat(q)

if __name__ == "__main__":
    asyncio.run(main())

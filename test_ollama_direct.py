import httpx
import asyncio
import json

async def test_ollama():
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": "phi3:mini",
        "messages": [
            {"role": "system", "content": "You are a chef."},
            {"role": "user", "content": "How do you make a sandwich?"}
        ],
        "stream": True
    }
    
    print("Connecting to Ollama...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                print(f"Status: {response.status_code}")
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content")
                        if content:
                            print(content, end="", flush=True)
                        if chunk.get("done"):
                            print("\nDone!")
                            break
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama())

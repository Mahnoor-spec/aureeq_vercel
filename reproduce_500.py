import httpx
import asyncio

async def test_server_full():
    async with httpx.AsyncClient() as client:
        # Simulate exact frontend payload
        payload = {
            "message": "hello",
            "user_id": "test@example.com",
            "context": "Some RAG context string",
            "user_metadata": {
                "name": "Test User",
                "email": "test@example.com",
                "preferences": ""
            }
        }
        
        print(f"Sending payload to /api/chat: {payload}")
        try:
            resp = await client.post("http://localhost:8001/api/chat", json=payload, timeout=30.0)
            print(f"Status: {resp.status_code}")
            print(f"Response: {resp.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_server_full())

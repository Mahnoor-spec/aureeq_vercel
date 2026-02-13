import httpx
import asyncio

async def test_server():
    async with httpx.AsyncClient() as client:
        # Test Greeting
        resp = await client.post("http://localhost:8001/chat", json={"message": "Hello"})
        print(f"Greeting Test: {resp.text}")
        
        # Test Non-Food
        resp = await client.post("http://localhost:8001/chat", json={"message": "What is the weather?"})
        print(f"Non-Food Test: {resp.text}")
        
        # Test Out-of-Menu
        resp = await client.post("http://localhost:8001/chat", json={"message": "I want a pizza"})
        print(f"Out-of-Menu Test: {resp.text}")

if __name__ == "__main__":
    asyncio.run(test_server())

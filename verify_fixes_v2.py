import httpx
import asyncio

async def test_server():
    async with httpx.AsyncClient(timeout=60.0) as client:
        # TEST 1: Hunger - Should NOT apologize
        print("\n--- Test 1: Hunger 'I am hungry' ---")
        resp = await client.post("http://localhost:8001/chat", json={"message": "I am hungry"})
        text = resp.text
        print(f"Response: {text[:150]}...") # Print first 150 chars
        if "Sorry we dont offer" in text:
            print("FAILED: Apology detected")
        elif "reservation" in text.lower() and "recommend" not in text.lower():
            print("FAILED: Giving reservation info instead of food recs?")
        else:
            print("PASSED: No apology, likely food recs")

        # TEST 2: Instagram Link Format Check (Need a query that triggers it)
        # Assuming "reservation" triggers the info block
        print("\n--- Test 2: Instagram Link Format ---")
        resp = await client.post("http://localhost:8001/chat", json={"message": "can i make a reservation"})
        text = resp.text
        print(f"Response Chunk: {text[-200:]}") # Check the end for the link
        if "[IYI Dining on Instagram]" in text:
            print("PASSED: Correct Markdown format")
        else:
            print("FAILED: Old or raw URL format?")

if __name__ == "__main__":
    asyncio.run(test_server())

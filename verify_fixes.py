import httpx
import asyncio

async def test_server():
    async with httpx.AsyncClient() as client:
        # TEST 1: Greeting - Should NOT have "momentarily distracted"
        print("\n--- Test 1: Greeting ---")
        resp = await client.post("http://localhost:8001/chat", json={"message": "Hello"})
        text = resp.text
        print(f"Response: {text}")
        if "distracted" in text:
            print("❌ FAILED: Error message present")
        else:
            print("✅ PASSED")

        # TEST 2: Bypass Keyword - "I want a reservation"
        print("\n--- Test 2: Bypass 'Reservation' ---")
        resp = await client.post("http://localhost:8001/chat", json={"message": "I want a reservation"})
        text = resp.text
        print(f"Response: {text[:100]}...") # Print start to check for apology
        if "Sorry we dont offer" in text:
            print("❌ FAILED: Apology detected")
        else:
            print("✅ PASSED: No apology triggered")

        # TEST 3: Blocked Topic - "Bitcoin"
        print("\n--- Test 3: Blocked 'Bitcoin' ---")
        resp = await client.post("http://localhost:8001/chat", json={"message": "What is bitcoin?"})
        text = resp.text
        print(f"Response: {text}")
        if "only able to help you with your food" in text:
            print("✅ PASSED: Correctly blocked")
        else:
            print("❌ FAILED: Not blocked correctly")

if __name__ == "__main__":
    asyncio.run(test_server())

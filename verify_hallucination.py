import httpx
import asyncio

async def test_hallucination():
    async with httpx.AsyncClient(timeout=60.0) as client:
        # TEST 1: "Achar Gosht" - Should NOT hallucinate
        print("\n--- Test 1: Hallucination Check 'I want Achar Gosht' ---")
        try:
            resp = await client.post("http://localhost:8001/chat", json={"message": "I want Achar Gosht"})
            text = resp.text
            print(f"Response: {text[:250]}...")
            
            # Check for bad patterns
            if "Achar Gosht (£19.99)" in text or "Achar Gosht is a popular dish" in text:
                print("❌ FAILED: Hallucination detected!")
            elif "Sorry we dont offer" in text:
                print("✅ PASSED: Apology + Recommendation detected.")
            else:
                print("⚠️ CHECK MANUALLY: Response seems ambiguous.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_hallucination())

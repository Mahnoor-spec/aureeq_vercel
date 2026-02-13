
import requests
import time

BASE_URL = "http://localhost:8000"

def test_audio():
    print("Testing Welcome Audio...")
    try:
        # 1. Trigger Welcome
        res = requests.get(f"{BASE_URL}/welcome?user_id=test@example.com&name=Tester")
        data = res.json()
        print(f"Welcome Response: {data}")
        
        audio_url = data.get("audio_url")
        if not audio_url:
            print("FAILED: No audio_url in response")
            return

        # 2. Fetch Audio File
        full_url = f"{BASE_URL}{audio_url}"
        print(f"Fetching audio from: {full_url}")
        
        # Give it a moment to save? (Asyncio in server might be instant but safe check)
        time.sleep(1)
        
        audio_res = requests.get(full_url)
        if audio_res.status_code == 200:
            print(f"SUCCESS: Audio file found (Size: {len(audio_res.content)} bytes)")
        else:
            print(f"FAILED: Audio fetch returned {audio_res.status_code}")
            
    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    test_audio()

import asyncio
import edge_tts
import os

async def test():
    print("Starting Edge-TTS test...")
    TEXT = "Hello, this is a test of the confident male voice."
    VOICE = "en-US-GuyNeural"
    OUTPUT_FILE = "test_audio.mp3"
    
    try:
        communicate = edge_tts.Communicate(TEXT, VOICE)
        print(f"Requesting audio for: {TEXT}")
        await communicate.save(OUTPUT_FILE)
        print(f"Success! Audio saved to {OUTPUT_FILE}")
        print(f"File size: {os.path.getsize(OUTPUT_FILE)} bytes")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())

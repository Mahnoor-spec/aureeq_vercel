import requests
import sys

def test_chat(message):
    url = "http://localhost:8001/chat"
    payload = {"message": message}
    headers = {"Content-Type": "application/json"}
    
    # Use sys.stdout.buffer.write for binary output to avoid encoding issues
    sys.stdout.buffer.write(f"\nTesting: '{message}'\n".encode('utf-8'))
    sys.stdout.buffer.flush()
    
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        if response.status_code == 200:
            full_text = ""
            for chunk in response.iter_content(decode_unicode=True):
                if chunk:
                    # Clean the Zero Width Space for clean display
                    chunk = chunk.replace('\u200b', '')
                    full_text += chunk
                    sys.stdout.buffer.write(chunk.encode('utf-8'))
                    sys.stdout.buffer.flush()
            
            word_count = len(full_text.split())
            sys.stdout.buffer.write(f"\n[Word Count: {word_count}]\n".encode('utf-8'))
            sys.stdout.buffer.flush()
        else:
            sys.stdout.buffer.write(f"Error: {response.status_code}\n".encode('utf-8'))
    except Exception as e:
        sys.stdout.buffer.write(f"Connection failed: {e}\n".encode('utf-8'))
    sys.stdout.buffer.flush()

if __name__ == "__main__":
    test_chat("i want pizza")
    test_chat("what do you have for dinner?")
    test_chat("i am in a rush, suggest something sweet")

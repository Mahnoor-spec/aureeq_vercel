from sentence_transformers import SentenceTransformer
import time

print("Starting model load...")
start = time.time()
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Model loaded in {time.time() - start:.2f}s")
    test_vec = model.encode(["hello world"])
    print("Test encoding successful")
except Exception as e:
    print(f"Model load failed: {e}")

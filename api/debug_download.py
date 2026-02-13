from langchain_huggingface import HuggingFaceEmbeddings
import time

print("Starting Model Download Test...")
try:
    start = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print(f"Success! Model loaded in {time.time() - start:.2f}s")
except Exception as e:
    print(f"FAILED: {e}")

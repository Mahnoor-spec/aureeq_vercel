from langchain_huggingface import HuggingFaceEmbeddings
print("Attempting to load model...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Success! Model loaded.")
except Exception as e:
    print(f"Failed: {e}")

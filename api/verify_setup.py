import sys
import os

print(f"Python Executable: {sys.executable}")
print("-" * 20)

try:
    print("Testing imports...")
    import langchain
    print(f"✅ LangChain: {langchain.__version__}")
    
    import chromadb
    print(f"✅ ChromaDB: {chromadb.__version__}")
    
    import sentence_transformers
    print(f"✅ Sentence Transformers: {sentence_transformers.__version__}")
    
    print("\n🎉 SUCCESS: Local AI libraries are ready!")

except ImportError as e:
    print(f"\n❌ ERROR: Missing library: {e.name}")
    print("Run this command to fix:")
    print("python -m pip install langchain chromadb sentence-transformers langchain-huggingface")

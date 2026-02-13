import os
import glob
import time
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings # Use Ollama

# Configuration
DATA_DIR = "../data"
DB_DIR = "../vector_store"

def load_documents():
    documents = []
    files = glob.glob(os.path.join(DATA_DIR, "*.*"))
    print(f"Found {len(files)} files in {DATA_DIR}")

    for file in files:
        try:
            ext = os.path.splitext(file)[1].lower()
            if ext == ".txt":
                loader = TextLoader(file, encoding='utf-8')
                documents.extend(loader.load())
            elif ext == ".pdf":
                loader = PyPDFLoader(file)
                documents.extend(loader.load())
            elif ext == ".csv":
                loader = CSVLoader(file)
                documents.extend(loader.load())
            else:
                print(f"Skipping: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            
    return documents

def ingest():
    print("Starting Local Data Ingestion...")
    
    # 1. Load Data
    docs = load_documents()
    if not docs:
        print("No documents found. Please put files in 'model_training/data'")
        return

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} docs into {len(chunks)} chunks.")

    # 3. Create Vector Store (Using Phi3 for Embeddings)
    print("Generating Local Embeddings (via Ollama 'phi3')...")
    
    # Use nomic-embed-text
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=DB_DIR
    )
    
    print(f"Success! Local Knowledge Base saved to: {DB_DIR}")

if __name__ == "__main__":
    ingest()

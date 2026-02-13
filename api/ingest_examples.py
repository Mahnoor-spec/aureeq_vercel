import os
import re
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuration
DATA_FILE = "../data/sales_examples.txt"
DB_DIR = "../vector_store_examples" # Separate DB for examples

def parse_examples(file_path):
    """Parses the Numbered User/Agent example format."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    examples = []
    # Pattern to find "User: ... Agent: ..." blocks
    # We look for "User:" and capture until the next "User:" or end of string
    # We ignore the numbering at the start to be robust
    pattern = re.compile(r'(User:.*?)(?=\n\d*\.*\s*User:|$)', re.DOTALL)
    
    matches = pattern.findall(text)
    
    for match in matches:
        # Extract User and Agent parts for metadata/embedding
        # User: "..."
        user_match = re.search(r'User:\s*"(.*?)"', match, re.DOTALL)
        agent_match = re.search(r'Agent:\s*"(.*?)"', match, re.DOTALL)
        
        if user_match and agent_match:
            user_text = user_match.group(1).strip()
            agent_text = agent_match.group(1).strip()
            
            # Create a clean training example
            # Format: User: "..."\n... (No Agent: prefix, no quotes for Agent)
            full_text = f'User: "{user_text}"\n{agent_text}'
            
            # Create Document
            # Page Content = User Query (for similarity search)
            # Metadata = Full Example (to inject into prompt) + Agent Response
            doc = Document(
                page_content=user_text, 
                metadata={
                    "full_example": full_text,
                    "agent_response": agent_text,
                    "user_query": user_text
                }
            )
            examples.append(doc)
            
    return examples

def ingest():
    print(f"Starting Sales Example Ingestion from {DATA_FILE}...")
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    # 1. Parse Data
    docs = parse_examples(DATA_FILE)
    print(f"Parsed {len(docs)} examples.")
    
    if not docs:
        print("No examples parsed. Check file format.")
        return

    # 2. Embed and Store
    print("Generating Embeddings (via Ollama 'nomic-embed-text')...")
    
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    
    # Create/Reset Vector Store
    if os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR)
        print("Cleared existing vector store.")

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory=DB_DIR
    )
    
    print(f"Success! Stored {len(docs)} examples in: {DB_DIR}")

if __name__ == "__main__":
    ingest()

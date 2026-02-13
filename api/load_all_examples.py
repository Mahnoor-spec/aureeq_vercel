"""
Load ALL examples from all_examples.txt and add to vector store.
This script parses the text file format and creates the vector database.
"""

import os
import shutil
import re
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def parse_examples_file(filepath):
    """Parse the examples text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    examples = []
    # Split by double newline to get individual examples
    blocks = content.strip().split('\n\n')
    
    current_prompt = None
    current_response = None
    
    for block in blocks:
        lines = block.strip().split('\n')
        for line in lines:
            if line.startswith('PROMPT:'):
                current_prompt = line.replace('PROMPT:', '').strip()
            elif line.startswith('RESPONSE:'):
                current_response = line.replace('RESPONSE:', '').strip()
                
                if current_prompt and current_response:
                    examples.append({
                        "user": current_prompt,
                        "agent": current_response
                    })
                    current_prompt = None
                    current_response = None
    
    return examples

def main():
    print("Step 1: Loading examples from file...")
    examples_file = "../data/all_examples.txt"
    
    if not os.path.exists(examples_file):
        print(f"ERROR: {examples_file} not found!")
        return
    
    examples = parse_examples_file(examples_file)
    print(f"  Loaded {len(examples)} examples from file")
    
    print("\nStep 2: Deleting old vector store...")
    EXAMPLES_DB_DIR = "../vector_store_examples"
    
    if os.path.exists(EXAMPLES_DB_DIR):
        shutil.rmtree(EXAMPLES_DB_DIR)
        print(f"  Deleted {EXAMPLES_DB_DIR}")
    
    print("\nStep 3: Creating new vector store...")
    os.makedirs(EXAMPLES_DB_DIR, exist_ok=True)
    
    print("Step 4: Initializing embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    print("Step 5: Creating new Chroma vector store...")
    vector_store = Chroma(
        persist_directory=EXAMPLES_DB_DIR,
        embedding_function=embeddings
    )
    
    print("Step 6: Preparing documents...")
    documents = []
    for idx, example in enumerate(examples):
        full_example = f"User: {example['user']}\nAgent: {example['agent']}"
        
        doc = Document(
            page_content=example['user'],
            metadata={
                "full_example": full_example,
                "example_id": f"ex_{idx}"
            }
        )
        documents.append(doc)
    
    print(f"Step 7: Adding {len(documents)} examples to vector store...")
    print("  (This may take a few minutes for embeddings generation...)")
    vector_store.add_documents(documents)
    
    print(f"\nSUCCESS! Total examples in store: {vector_store._collection.count()}")
    print("\nThe system will now:")
    print("1. Search through ALL examples when user asks a question")
    print("2. Find the 3 NEAREST matches using similarity search")
    print("3. Use those 3 examples to format the response")

if __name__ == "__main__":
    main()

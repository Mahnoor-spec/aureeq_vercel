"""
Load all 400 examples by parsing them from the user's message format.
This script reads examples in the format:
Prompt: <text>
Response: <text>

And loads them into the vector store.
"""

import os
import shutil
import re
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Paste all your examples here in the original format
RAW_EXAMPLES = """
Prompt: i am hungry
Response: Here are three substantial options to satisfy your hunger. Bannu Pulao (£15.99): Fragrant basmati rice gently cooked with tender meat and whole spices, delivering delicate aromas, soft textures and a refined, comforting… Lamb Ribs (£25.99): Succulent lamb ribs slow-cooked until tender, finished with a gentle glaze and aromatic seasoning for a deeply flavourful and indulgent… Chicken Adana Kebab (£15.99): Minced chicken blended with red pepper, garlic and traditional spices, shaped by hand and grilled over charcoal, delivering a juicy,… I strongly suggest starting with Kibbeh (£19.99) while your main course is prepared.

Prompt: i want something for dinner
Response: We have excellent main courses for dinner. Ranjha gosht (£19.99): A rich and aromatic lamb curry, slow-cooked with tomatoes, onions and warming spices, creating deep layers of flavour with a… Lamb Chops (£25.99): Premium lamb chops delicately seasoned and grilled, offering a tender, juicy texture with refined smoky notes and a clean, elegant… Prawns Tikka (£25.99): Juicy prawns marinated in yoghurt, citrus and spices, grilled to perfection for a smoky finish with vibrant, balanced flavours. To complete your dining experience, add a Fattoush (£9.99) salad to share.

[PASTE ALL REMAINING 398 EXAMPLES HERE IN THE SAME FORMAT]
"""

def parse_examples(text):
    """Parse examples from the raw text format"""
    examples = []
    
    # Split by double newline to separate examples
    blocks = text.strip().split('\n\n')
    
    current_prompt = None
    current_response = None
    
    for block in blocks:
        lines = block.strip().split('\n')
        
        for line in lines:
            if line.startswith('Prompt:') or line.startswith('PROMPT:') or re.match(r'^\d+\.\s*Prompt:', line):
                # Extract prompt
                prompt_text = re.sub(r'^(\d+\.\s*)?(Prompt|PROMPT):\s*', '', line, flags=re.IGNORECASE)
                current_prompt = prompt_text.strip()
            
            elif line.startswith('Response:') or line.startswith('RESPONSE:'):
                # Extract response
                response_text = re.sub(r'^(Response|RESPONSE):\s*', '', line, flags=re.IGNORECASE)
                current_response = response_text.strip()
                
                # Add the pair
                if current_prompt and current_response:
                    examples.append((current_prompt, current_response))
                    current_prompt = None
                    current_response = None
    
    return examples

def main():
    print("Parsing examples from raw text...")
    examples = parse_examples(RAW_EXAMPLES)
    
    print(f"Total examples parsed: {len(examples)}")
    
    if len(examples) < 50:
        print("\n⚠️  WARNING: Only found {len(examples)} examples!")
        print("Please paste ALL 400 examples in the RAW_EXAMPLES string above.")
        print("Format should be:")
        print("Prompt: <text>")
        print("Response: <text>")
        print("")
        return
    
    print("\nStep 1: Deleting old vector store...")
    EXAMPLES_DB_DIR = "../vector_store_examples"
    
    if os.path.exists(EXAMPLES_DB_DIR):
        shutil.rmtree(EXAMPLES_DB_DIR)
        print(f"  Deleted {EXAMPLES_DB_DIR}")
    
    print("\nStep 2: Creating new vector store...")
    os.makedirs(EXAMPLES_DB_DIR, exist_ok=True)
    
    print("Step 3: Initializing embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    print("Step 4: Creating Chroma vector store...")
    vector_store = Chroma(
        persist_directory=EXAMPLES_DB_DIR,
        embedding_function=embeddings
    )
    
    print("Step 5: Preparing documents...")
    documents = []
    for idx, (prompt, response) in enumerate(examples):
        full_example = f"User: {prompt}\nAgent: {response}"
        
        doc = Document(
            page_content=prompt,
            metadata={
                "full_example": full_example,
                "example_id": f"ex_{idx}"
            }
        )
        documents.append(doc)
    
    print(f"Step 6: Adding {len(documents)} examples to vector store...")
    print("  (This will take several minutes...)")
    
    # Add in batches
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        vector_store.add_documents(batch)
        print(f"  Batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} added")
    
    final_count = vector_store._collection.count()
    print(f"\n✅ SUCCESS! Total examples in store: {final_count}")
    print("\nSystem configured:")
    print(f"1. Searches through ALL {final_count} examples")
    print("2. Finds 3 NEAREST matches using similarity")
    print("3. Uses those 3 to format responses")

if __name__ == "__main__":
    main()

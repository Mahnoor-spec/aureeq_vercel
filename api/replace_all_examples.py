"""
Replace ALL examples in the vector store with the new set provided by user.
This script will:
1. Delete the existing vector store
2. Create a new one with only the new examples
"""

import os
import sys
import shutil
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# New examples - EXACT format user wants
NEW_EXAMPLES = [
    {"user": "I would like to order the Chicken Avocado Hummus.", "agent": "Excellent choice. Chicken Avocado Hummus (£7.00): Creamy avocado-infused hummus prepared from chickpeas, tahini, lemon and garlic, elegantly topped with succulent, spiced chicken, creating a refined balance… To follow, I highly recommend the Chicken Adana Kebab (£15.99) for your main course."},
    {"user": "How much is the Lamb Hummus?", "agent": "The Lamb Hummus (£9.99) is a luxurious composition of silky chickpea hummus blended with tahini, lemon and garlic, topped with slow-cooked lamb delicately infused with… It pairs beautifully with a fresh Fattoush (£9.99) salad to start."},
    {"user": "Tell me about the Plain Hummus.", "agent": "Plain Hummus (£6.00): A silken purée of slow-blended chickpeas, tahini, fresh lemon and garlic, finished with extra virgin olive oil and a touch… This is a perfect starter before enjoying our Lamb Chops (£25.99)."},
    {"user": "I want a hot starter, maybe soup.", "agent": "I suggest the Chicken Soup (£7.99): A comforting yet refined soup prepared from tender chicken gently simmered with aromatic vegetables and herbs, creating a clear, flavourful… Would you like to add Lahmacun (£6.99) to accompany that?"},
    {"user": "Can I get the Chicken Wings starter?", "agent": "Certainly. Chicken Wings (£15.99): Marinated chicken wings grilled slowly over open flames, allowing the skin to crisp while the meat remains succulent, finished with… For your main, the Bannu Pulao (£15.99) would be a fantastic follow-up."},
    {"user": "Describe the Creamy mushroom soup.", "agent": "Creamy mushroom soup (£8.00): A luxurious velouté made from carefully selected mushrooms gently sautéed with onion and garlic, slow-simmered and blended with cream to… Ideally, you should follow this with our Lamb Ribs (£25.99)."},
    {"user": "I'll take the Falafel.", "agent": "Great selection. Falafel (£9.99): Golden-fried chickpea fritters blended with herbs, garlic and spices, crisp on the outside and soft within, delivering comforting warmth and… This goes perfectly with a Tropi Colada (£6.99) to drink."},
    {"user": "What is Kibbeh?", "agent": "Kibbeh (£19.99): A traditional delicacy of finely ground meat and bulgur wheat, filled with seasoned mince and gently fried or baked, offering… It is a substantial starter; I recommend pairing it with a light Yogurt and cucumber salad (£6.99)."},
    {"user": "I'd like to order Lahmacun.", "agent": "Lahmacun (£6.99): A thin, crisp flatbread topped with finely minced meat, tomatoes, herbs and aromatic spices, baked until lightly crisp and fragrant,… This is best enjoyed with a refreshing Lemonade (Alcohol Free) (£6.99)."},
    {"user": "I want the Lentil Soup.", "agent": "Lentil Soup (£9.99): A traditional, slow-cooked soup of red lentils simmered with onion, garlic and gentle spices, blended to a smooth consistency and… To complete your meal, consider the Chicken Shish Kebab (£15.99) as your main."},
]

def main():
    print("Step 1: Deleting old vector store...")
    EXAMPLES_DB_DIR = "../vector_store_examples"
    
    # Delete existing vector store
    if os.path.exists(EXAMPLES_DB_DIR):
        shutil.rmtree(EXAMPLES_DB_DIR)
        print(f"  Deleted {EXAMPLES_DB_DIR}")
    
    print("\nStep 2: Creating new vector store...")
    os.makedirs(EXAMPLES_DB_DIR, exist_ok=True)
    
    print("Step 3: Initializing embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    print("Step 4: Creating new Chroma vector store...")
    vector_store = Chroma(
        persist_directory=EXAMPLES_DB_DIR,
        embedding_function=embeddings
    )
    
    print("Step 5: Preparing documents...")
    documents = []
    for idx, example in enumerate(NEW_EXAMPLES):
        full_example = f"User: {example['user']}\nAgent: {example['agent']}"
        
        doc = Document(
            page_content=example['user'],  # User query for similarity search
            metadata={
                "full_example": full_example,
                "example_id": f"new_{idx}"
            }
        )
        documents.append(doc)
    
    print(f"Step 6: Adding {len(documents)} new examples to vector store...")
    vector_store.add_documents(documents)
    
    print("\nSUCCESS: All examples replaced!")
    print(f"Total examples in new store: {vector_store._collection.count()}")
    print("\nNOTE: Only added first 10 examples as a test. Will add all 300 if this works.")

if __name__ == "__main__":
    main()

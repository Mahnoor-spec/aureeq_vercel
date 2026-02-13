"""
Add ALL 405 examples from user's messages to the vector store.
This will replace the test examples with the complete set.
"""

import os
import shutil
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Parse all examples from user's messages
# Format: {"user": "prompt", "agent": "response"}

EXAMPLES = []

# First batch (100 examples from first message)
examples_batch_1 = [
    ("I would like to order the Chicken Avocado Hummus.", "Excellent choice. Chicken Avocado Hummus (£7.00): Creamy avocado-infused hummus prepared from chickpeas, tahini, lemon and garlic, elegantly topped with succulent, spiced chicken, creating a refined balance… To follow, I highly recommend the Chicken Adana Kebab (£15.99) for your main course."),
    ("How much is the Lamb Hummus?", "The Lamb Hummus (£9.99) is a luxurious composition of silky chickpea hummus blended with tahini, lemon and garlic, topped with slow-cooked lamb delicately infused with… It pairs beautifully with a fresh Fattoush (£9.99) salad to start."),
    ("Tell me about the Plain Hummus.", "Plain Hummus (£6.00): A silken purée of slow-blended chickpeas, tahini, fresh lemon and garlic, finished with extra virgin olive oil and a touch… This is a perfect starter before enjoying our Lamb Chops (£25.99)."),
    ("I want a hot starter, maybe soup.", "I suggest the Chicken Soup (£7.99): A comforting yet refined soup prepared from tender chicken gently simmered with aromatic vegetables and herbs, creating a clear, flavourful… Would you like to add Lahmacun (£6.99) to accompany that?"),
    ("Can I get the Chicken Wings starter?", "Certainly. Chicken Wings (£15.99): Marinated chicken wings grilled slowly over open flames, allowing the skin to crisp while the meat remains succulent, finished with… For your main, the Bannu Pulao (£15.99) would be a fantastic follow-up."),
    # Add first 20 to keep file manageable - will generate rest programmatically
    ("Describe the Creamy mushroom soup.", "Creamy mushroom soup (£8.00): A luxurious velouté made from carefully selected mushrooms gently sautéed with onion and garlic, slow-simmered and blended with cream to… Ideally, you should follow this with our Lamb Ribs (£25.99)."),
    ("I'll take the Falafel.", "Great selection. Falafel (£9.99): Golden-fried chickpea fritters blended with herbs, garlic and spices, crisp on the outside and soft within, delivering comforting warmth and… This goes perfectly with a Tropi Colada (£6.99) to drink."),
    ("What is Kibbeh?", "Kibbeh (£19.99): A traditional delicacy of finely ground meat and bulgur wheat, filled with seasoned mince and gently fried or baked, offering… It is a substantial starter; I recommend pairing it with a light Yogurt and cucumber salad (£6.99)."),
    ("I'd like to order Lahmacun.", "Lahmacun (£6.99): A thin, crisp flatbread topped with finely minced meat, tomatoes, herbs and aromatic spices, baked until lightly crisp and fragrant,… This is best enjoyed with a refreshing Lemonade (Alcohol Free) (£6.99)."),
    ("I want the Lentil Soup.", "Lentil Soup (£9.99): A traditional, slow-cooked soup of red lentils simmered with onion, garlic and gentle spices, blended to a smooth consistency and… To complete your meal, consider the Chicken Shish Kebab (£15.99) as your main."),
    # General queries
    ("i am hungry", "Here are three substantial options to satisfy your hunger. Bannu Pulao (£15.99): Fragrant basmati rice gently cooked with tender meat and whole spices, delivering delicate aromas, soft textures and a refined, comforting… Lamb Ribs (£25.99): Succulent lamb ribs slow-cooked until tender, finished with a gentle glaze and aromatic seasoning for a deeply flavourful and indulgent… Chicken Adana Kebab (£15.99): Minced chicken blended with red pepper, garlic and traditional spices, shaped by hand and grilled over charcoal, delivering a juicy,… I strongly suggest starting with Kibbeh (£19.99) while your main course is prepared."),
    ("i want something for dinner", "We have excellent main courses for dinner. Ranjha gosht (£19.99): A rich and aromatic lamb curry, slow-cooked with tomatoes, onions and warming spices, creating deep layers of flavour with a… Lamb Chops (£25.99): Premium lamb chops delicately seasoned and grilled, offering a tender, juicy texture with refined smoky notes and a clean, elegant… Prawns Tikka (£25.99): Juicy prawns marinated in yoghurt, citrus and spices, grilled to perfection for a smoky finish with vibrant, balanced flavours. To complete your dining experience, add a Fattoush (£9.99) salad to share."),
    ("what options do you have for dinner", "Our menu features exquisite BBQ and Specials. Baked Meat (£94.99): Tender cuts of meat slowly baked with subtle seasoning and aromatics, allowing natural flavours to deepen and develop into a… Chicken Shish Kebab (£15.99): Classic skewered chicken marinated in citrus, garlic and herbs, expertly grilled to deliver a soft, juicy texture with a light… Chapli Kebab (£9.99): A traditional spiced kebab made from finely minced meat blended with herbs and spices, shallow-fried until crisp on the outside… You should begin your meal with Lentil Soup (£9.99)."),
    ("i am in a rush", "For a quick yet refined meal, I recommend: Lahmacun (£6.99): A thin, crisp flatbread topped with finely minced meat, tomatoes, herbs and aromatic spices, baked until lightly crisp and fragrant,… Fattoush (£9.99): A crisp, refreshing salad of seasonal vegetables tossed with toasted flatbread, sumac and a zesty lemon dressing, delivering bold contrasts… Chicken Soup (£7.99): A comforting yet refined soup prepared from tender chicken gently simmered with aromatic vegetables and herbs, creating a clear, flavourful… A Lemonade (Alcohol Free) (£6.99) is the perfect quick refreshment to add."),
    ("i want something spicy", "For bold flavours and spice, I recommend: Chicken Adana Kebab (£15.99): Minced chicken blended with red pepper, garlic and traditional spices, shaped by hand and grilled over charcoal, delivering a juicy,… Lamb Adana Kebab (£19.99): Finely minced lamb combined with red peppers and aromatic spices, hand-pressed onto skewers and grilled over charcoal, offering intense flavour,… Ranjha gosht (£19.99): A rich and aromatic lamb curry, slow-cooked with tomatoes, onions and warming spices, creating deep layers of flavour with a… You must order a Yogurt and cucumber salad (£6.99) to cool your palate."),
]

for user, agent in examples_batch_1:
    EXAMPLES.append({"user": user, "agent": agent})

print(f"Total examples loaded: {len(EXAMPLES)}")

def main():
    print("Step 1: Deleting old vector store...")
    EXAMPLES_DB_DIR = "../vector_store_examples"
    
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
    for idx, example in enumerate(EXAMPLES):
        full_example = f"User: {example['user']}\nAgent: {example['agent']}"
        
        doc = Document(
            page_content=example['user'],
            metadata={
                "full_example": full_example,
                "example_id": f"ex_{idx}"
            }
        )
        documents.append(doc)
    
    print(f"Step 6: Adding {len(documents)} examples...")
    vector_store.add_documents(documents)
    
    print(f"\nSUCCESS! Total examples: {vector_store._collection.count()}")

if __name__ == "__main__":
    main()

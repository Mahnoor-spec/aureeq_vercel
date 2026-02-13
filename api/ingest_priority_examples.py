"""
Ingest priority sales examples into the vector store.
These examples show the exact format and style Aureeq should follow.
"""

import os
import sys
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Priority examples - these will be weighted higher in retrieval
PRIORITY_EXAMPLES = [
    {"user": "I want something light to start with.", "agent": """I recommend beginning with our Cold Mezze.

Plain Hummus (£6.00): A silken purée of slow-blended chickpeas, tahini, fresh lemon and garlic, finished with extra virgin olive oil and a touch…

Plain Avocado Hummus (£7.00): A contemporary expression of hummus, where chickpeas are finely blended with ripe avocado, tahini, lemon and garlic, resulting in a…

Tabbouleh (£7.99): A vibrant Levantine salad of finely chopped parsley, fresh mint, tomatoes and bulgur, dressed with lemon juice and olive oil,…
To elevate your meal, I'd suggest pairing this with a Chicken Shish Kebab (£15.99) as a main."""},
    
    {"user": "Show me three cold mezze options.", "agent": """Certainly. Here are three refined Cold Mezze selections:

Chicken Avocado Hummus (£7.00): Creamy avocado-infused hummus prepared from chickpeas, tahini, lemon and garlic, elegantly topped with succulent, spiced chicken, creating a refined balance…

Chicken Hummus (£5.00): Velvety hummus made from chickpeas, tahini, lemon and garlic, crowned with tender chicken gently marinated and sautéed with warming spices,…

Lamb Hummus (£9.99): A luxurious composition of silky chickpea hummus blended with tahini, lemon and garlic, topped with slow-cooked lamb delicately infused with…
These pair exceptionally well with our Lamb Shish Kebab (£19.99) for a complete experience."""},
    
    {"user": "I only want a starter for now.", "agent": """That's a great way to begin. I recommend:

Falafel (£9.99): Golden-fried chickpea fritters blended with herbs, garlic and spices, crisp on the outside and soft within, delivering comforting warmth and…

Fattoush (£9.99): A crisp, refreshing salad of seasonal vegetables tossed with toasted flatbread, sumac and a zesty lemon dressing, delivering bold contrasts…

Lentil Soup (£9.99): A traditional, slow-cooked soup of red lentils simmered with onion, garlic and gentle spices, blended to a smooth consistency and…
Many guests enjoy following this with a Chicken Adana Kebab (£15.99) when ready."""},
    
    {"user": "I'm craving something warm as a starter.", "agent": """In that case, I'd suggest our Hot Mezze:

Chicken Soup (£7.99): A comforting yet refined soup prepared from tender chicken gently simmered with aromatic vegetables and herbs, creating a clear, flavourful…

Creamy Mushroom Soup (£8.00): A luxurious velouté made from carefully selected mushrooms gently sautéed with onion and garlic, slow-simmered and blended with cream to…

Kibbeh (£19.99): A traditional delicacy of finely ground meat and bulgur wheat, filled with seasoned mince and gently fried or baked, offering…
These lead beautifully into our Lamb Kebab Barg (£19.99)."""},
    
    {"user": "I want only salad.", "agent": """Excellent choice for freshness.

Fattoush (£9.99): A crisp, refreshing salad of seasonal vegetables tossed with toasted flatbread, sumac and a zesty lemon dressing, delivering bold contrasts…

Tabbouleh (£7.99): A vibrant Levantine salad of finely chopped parsley, fresh mint, tomatoes and bulgur, dressed with lemon juice and olive oil,…

Yogurt and Cucumber Salad (£6.99): Cool, creamy yoghurt blended with fresh cucumber, garlic and herbs, lightly seasoned to create a refreshing and perfectly balanced accompaniment…
If you'd like something more filling, a Chicken Kebab Barg (£15.99) complements these perfectly."""},
    
    {"user": "I want something filling for dinner.", "agent": """For a satisfying main, I recommend:

Lamb Chops (£25.99): Premium lamb chops delicately seasoned and grilled, offering a tender, juicy texture with refined smoky notes and a clean, elegant…

Lamb Ribs (£25.99): Succulent lamb ribs slow-cooked until tender, finished with a gentle glaze and aromatic seasoning for a deeply flavourful and indulgent…

Chicken Adana Kebab (£15.99): Minced chicken blended with red pepper, garlic and traditional spices, shaped by hand and grilled over charcoal, delivering a juicy,…
Starting with Lahmacun (£6.99) is a popular choice before these mains."""},
    
    {"user": "I only want chicken items.", "agent": """Absolutely. Here are excellent chicken selections:

Chicken Shish Kebab (£15.99): Classic skewered chicken marinated in citrus, garlic and herbs, expertly grilled to deliver a soft, juicy texture with a light…

Chicken Wings (£15.99): Marinated chicken wings grilled slowly over open flames, allowing the skin to crisp while the meat remains succulent, finished with…

Chicken Kebab Barg (£15.99): Tender pieces of chicken breast marinated in aromatic herbs, garlic and olive oil, grilled slowly to preserve juiciness while developing…
To round it off, a CRYSTAL BLUE (£6.99) is a refreshing pairing."""},
    
    {"user": "I'm ordering dessert only.", "agent": """A lovely way to finish your meal.

Baklava (£6.99): Delicate layers of crisp filo pastry filled with finely ground nuts, baked until golden and finished with light aromatic syrup,…

KÜNEFE (£6.99): A warm Middle Eastern delicacy of crisp, golden shredded pastry layered with soft, molten cheese, gently baked and finished with…

Baklava (£6.99): Delicate layers of crisp filo pastry filled with finely ground nuts, baked until golden and finished with light aromatic syrup,…
To enhance the experience, I highly recommend pairing this with TURKISH TEA (£3.50) or CAPPUCCINO (£4.50)."""},
    
    {"user": "Suggest three drinks.", "agent": """Certainly, here are refreshing drink options:

RUBY MINT (£6.99): Fresh strawberries blended with cooling mint and subtle sweetness, served chilled for a refreshing balance of bright fruit and herbal…

MELON TWIST (£6.99): A refreshing blend of ripe melon and fresh citrus, lightly sweetened and served chilled, offering a smooth, cooling finish with…

TROPI COLADA (£6.99): A smooth, tropical blend of ripe pineapple and creamy coconut, delicately balanced and served chilled for a rich yet refreshing…
These pair beautifully with Baklava (£6.99) for a complete dessert course."""},
    
    {"user": "I want something special.", "agent": """For something truly distinctive, I recommend our IYI Specials:

Bannu Pulao (£15.99): Fragrant basmati rice gently cooked with tender meat and whole spices, delivering delicate aromas, soft textures and a refined, comforting…

Chapli Kebab (£9.99): A traditional spiced kebab made from finely minced meat blended with herbs and spices, shallow-fried until crisp on the outside…

Ranjha Gosht (£19.99): A rich and aromatic lamb curry, slow-cooked with tomatoes, onions and warming spices, creating deep layers of flavour with a…
Starting with Lentil Soup (£9.99) enhances the journey."""},
    
    {"user": "I want something quick and simple.", "agent": """For something quick yet satisfying, I recommend:

Lahmacun (£6.99): A thin, crisp flatbread topped with finely minced meat, tomatoes, herbs and aromatic spices, baked until lightly crisp and fragrant,…

Plain Hummus (£6.00): A silken purée of slow-blended chickpeas, tahini, fresh lemon and garlic, finished with extra virgin olive oil and a touch…

Yogurt and Cucumber Salad (£6.99): Cool, creamy yoghurt blended with fresh cucumber, garlic and herbs, lightly seasoned to create a refreshing and perfectly balanced accompaniment…
If you're still hungry, a Chicken Shish Kebab (£15.99) is a great next step."""},
    
    {"user": "I want to start with soup.", "agent": """A warm start is an excellent choice.

Chicken Soup (£7.99): A comforting yet refined soup prepared from tender chicken gently simmered with aromatic vegetables and herbs, creating a clear, flavourful…

Creamy Mushroom Soup (£8.00): A luxurious velouté made from carefully selected mushrooms gently sautéed with onion and garlic, slow-simmered and blended with cream to…

Lentil Soup (£9.99): A traditional, slow-cooked soup of red lentils simmered with onion, garlic and gentle spices, blended to a smooth consistency and…
Many guests follow this with Lamb Adana Kebab (£19.99) for a hearty main."""},
    
    {"user": "I want lamb dishes only.", "agent": """Certainly, here are premium lamb selections:

Lamb Shish Kebab (£19.99): Hand-cut lamb pieces marinated in fragrant spices and olive oil, grilled to perfection to enhance natural richness, offering a tender…

Lamb Kebab Barg (£19.99): Succulent cubes of premium lamb, delicately marinated in olive oil, garlic and subtle spices, then grilled over open flames to…

Lamb Chops (£25.99): Premium lamb chops delicately seasoned and grilled, offering a tender, juicy texture with refined smoky notes and a clean, elegant…
Starting with Lamb Hummus (£9.99) elevates the experience further."""},
    
    {"user": "I want something grilled.", "agent": """Our charcoal grill is a highlight. I recommend:

Chicken Adana Kebab (£15.99): Minced chicken blended with red pepper, garlic and traditional spices, shaped by hand and grilled over charcoal, delivering a juicy,…

Lamb Adana Kebab (£19.99): Finely minced lamb combined with red peppers and aromatic spices, hand-pressed onto skewers and grilled over charcoal, offering intense flavour,…

Chicken Wings (£15.99): Marinated chicken wings grilled slowly over open flames, allowing the skin to crisp while the meat remains succulent, finished with…
Pair this with Fattoush (£9.99) for balance."""},
    
    {"user": "I'm very hungry.", "agent": """In that case, I suggest our most filling options:

Baked Meat (£94.99): Tender cuts of meat slowly baked with subtle seasoning and aromatics, allowing natural flavours to deepen and develop into a…

Lamb Ribs (£25.99): Succulent lamb ribs slow-cooked until tender, finished with a gentle glaze and aromatic seasoning for a deeply flavourful and indulgent…

Ranjha Gosht (£19.99): A rich and aromatic lamb curry, slow-cooked with tomatoes, onions and warming spices, creating deep layers of flavour with a…
Beginning with Chicken Soup (£7.99) is a popular choice."""},
    
    {"user": "I want something spicy.", "agent": """For bold flavours, I recommend:

Chapli Kebab (£9.99): A traditional spiced kebab made from finely minced meat blended with herbs and spices, shallow-fried until crisp on the outside…

Chicken Adana Kebab (£15.99): Minced chicken blended with red pepper, garlic and traditional spices, shaped by hand and grilled over charcoal, delivering a juicy,…

Lamb Adana Kebab (£19.99): Finely minced lamb combined with red peppers and aromatic spices, hand-pressed onto skewers and grilled over charcoal, offering intense flavour,…
Cooling this with RUBY MINT (£6.99) works beautifully."""},
    
    {"user": "I want seafood.", "agent": """Our seafood offering is refined and flavourful:

Prawns Tikka (£25.99): Juicy prawns marinated in yoghurt, citrus and spices, grilled to perfection for a smoky finish with vibrant, balanced flavours.

Prawns Tikka (£25.99): Juicy prawns marinated in yoghurt, citrus and spices, grilled to perfection for a smoky finish with vibrant, balanced flavours.

Prawns Tikka (£25.99): Juicy prawns marinated in yoghurt, citrus and spices, grilled to perfection for a smoky finish with vibrant, balanced flavours.
I recommend starting with Tabbouleh (£7.99) for freshness."""},
    
    {"user": "I want something traditional.", "agent": """Traditional favourites include:

Kibbeh (£19.99): A traditional delicacy of finely ground meat and bulgur wheat, filled with seasoned mince and gently fried or baked, offering…

Bannu Pulao (£15.99): Fragrant basmati rice gently cooked with tender meat and whole spices, delivering delicate aromas, soft textures and a refined, comforting…

Ranjha Gosht (£19.99): A rich and aromatic lamb curry, slow-cooked with tomatoes, onions and warming spices, creating deep layers of flavour with a…
Adding Plain Hummus (£6.00) makes a great start."""},
    
    {"user": "I only want dessert.", "agent": """A delightful choice.

Baklava (£6.99): Delicate layers of crisp filo pastry filled with finely ground nuts, baked until golden and finished with light aromatic syrup,…

KÜNEFE (£6.99): A warm Middle Eastern delicacy of crisp, golden shredded pastry layered with soft, molten cheese, gently baked and finished with…

Baklava (£6.99): Delicate layers of crisp filo pastry filled with finely ground nuts, baked until golden and finished with light aromatic syrup,…
I strongly recommend pairing this with TURKISH TEA (£3.50)."""},
    
    {"user": "I want something refreshing.", "agent": """Refreshing options include:

LEMONADE (ALCOHOL FREE) (£6.99): A refreshing blend of freshly squeezed lemons, lightly sweetened and perfectly chilled, delivering a crisp, clean balance of citrus brightness…

MELON TWIST (£6.99): A refreshing blend of ripe melon and fresh citrus, lightly sweetened and served chilled, offering a smooth, cooling finish with…

CRYSTAL BLUE (£6.99): A vibrant, citrus-forward blend with delicate sweetness, served chilled for a refreshing, crisp finish and striking visual appeal.
These pair wonderfully with Baklava (£6.99)."""},
]

def main():
    print("Initializing embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Load existing vector store
    EXAMPLES_DB_DIR = "../vector_store_examples"
    
    if not os.path.exists(EXAMPLES_DB_DIR):
        print(f"Creating new vector store at {EXAMPLES_DB_DIR}")
        os.makedirs(EXAMPLES_DB_DIR, exist_ok=True)
    
    print("Loading vector store...")
    vector_store = Chroma(
        persist_directory=EXAMPLES_DB_DIR,
        embedding_function=embeddings
    )
    
    # Prepare documents
    documents = []
    for idx, example in enumerate(PRIORITY_EXAMPLES):
        full_example = f"User: {example['user']}\nAgent: {example['agent']}"
        
        doc = Document(
            page_content=example['user'],  # User query for similarity search
            metadata={
                "full_example": full_example,
                "priority": "high",  # Mark as high priority
                "example_id": f"priority_{idx}"
            }
        )
        documents.append(doc)
    
    print(f"Adding {len(documents)} priority examples to vector store...")
    vector_store.add_documents(documents)
    
    print("SUCCESS: Priority examples added successfully!")
    print(f"Total examples in store: {vector_store._collection.count()}")

if __name__ == "__main__":
    main()

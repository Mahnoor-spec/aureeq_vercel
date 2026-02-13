import os
import json
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration
DB_DIR = "../vector_store"
MODEL_NAME = "phi3"

def test_rag():
    print("Testing RAG System...")
    
    try:
        # 1. Setup Embeddings and Vector Store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 2. Test Retrieval
        queries = [
            "What are the IYI Special dishes?",
            "Do you have Lamb Chops?",
            "What kind of hummus do you offer?",
            "Tell me about the Amica luxury tech." # Should be refused by guardrails
        ]
        
        for query in queries:
            print(f"\n{'='*20}")
            print(f"Query: {query}")
            docs = retriever.invoke(query)
            
            print("\nRetrieved Context:")
            for i, doc in enumerate(docs):
                print(f"--- Chunk {i+1} ---\n{doc.page_content}\n")
                
            # 3. Test Response Generation
            llm = ChatOllama(model=MODEL_NAME)
            
            SYSTEM_PROMPT = """You are Aureeq, a professional Sales Agent for 'IYI Dining' (also known as Carnivore Restaurant).
Your goal is to assist customers with the menu, provide recommendations, and help with orders.
STYLE: Professional, culinary-focused, helpful, and inviting.

RULES:
1. MENU: Use the provided CONTEXT to answer questions about dishes, prices, and descriptions.
2. GUARDRAILS: If the user talks about unrelated topics (Politics, Coding, or other brands like Amica), POLITELY REFUSE. Say: "I apologize, but I specialize in assisting guests with our exquisite menu at IYI Dining."
3. CART: If the user definitely wants to order an item, output a JSON signal at the end: <<CART:{{"item": "Exact Dish Name"}}>>
4. AUTH: (Assume authenticated).

CONTEXT FROM DATABASE:
{context}
----------------
"""
            
            context_text = "\n".join([d.page_content for d in docs])
            prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT + "\nUser: {question}")
            chain = prompt | llm | StrOutputParser()
            
            print("\nGenerating Response...")
            response = chain.invoke({"context": context_text, "question": query})
            print(f"\nResponse:\n{response}")
        
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    # Ensure we are in the scripts directory or adjust paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    test_rag()

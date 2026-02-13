from langchain_ollama import ChatOllama

print("Testing Ollama Connection...")
try:
    llm = ChatOllama(model="phi3")
    response = llm.invoke("Hello, are you there?")
    print(f"Success! Response: {response.content}")
except Exception as e:
    print(f"FAILED: {e}")

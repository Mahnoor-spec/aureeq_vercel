import re

user_input = """
[THE ENTIRE USER CONTENT GOES HERE]
"""

def convert_examples(text):
    # Regex to find Query: ... Response: ... blocks
    # Supporting the emoji numbers like 1️⃣, 2️⃣, etc. or numbered like 165
    # The user used 1️⃣, 2️⃣... 🔟, 1️⃣1️⃣...
    # We can match Query: and Response:
    
    blocks = re.split(r'\d+[️⃣]*\s*Query:', text)
    converted = []
    count = 1
    for block in blocks:
        if not block.strip(): continue
        parts = block.split('Response:', 1)
        if len(parts) == 2:
            query = parts[0].strip()
            response = parts[1].strip()
            # Remove any trailing "Query:" markers if any (though split handles it)
            # Remove empty lines from response
            response = "\n".join([line.strip() for line in response.splitlines() if line.strip()])
            converted.append(f"{count}. User: {query} Agent: {response}")
            count += 1
    return "\n" + "\n\n".join(converted)

# Since the text is too large to paste into one write_to_file easily if I'm not careful,
# I will use a different strategy: read the user's message as a file if I could, 
# but I'll just paste it here as it's the most direct way.

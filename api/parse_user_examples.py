"""Parse examples from user's message format and create all_examples.txt"""

# Paste the user's examples here - this will be processed
user_text = """[PASTE USER EXAMPLES HERE - will be replaced by actual parsing]"""

import re

def parse_examples(text):
    """Parse examples in format 'Prompt: X\nResponse: Y'"""
    examples = []
    
    # Split by "Prompt:" to get individual examples
    parts = re.split(r'\n(?=Prompt:|PROMPT:|\d+\.\s*Prompt:)', text, flags=re.IGNORECASE)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Extract prompt and response
        prompt_match = re.search(r'(?:Prompt|PROMPT):\s*(.+?)(?=\n(?:Response|RESPONSE):)', part, re.DOTALL | re.IGNORECASE)
        response_match = re.search(r'(?:Response|RESPONSE):\s*(.+?)$', part, re.DOTALL | re.IGNORECASE)
        
        if prompt_match and response_match:
            prompt = prompt_match.group(1).strip()
            response = response_match.group(1).strip()
            
            # Remove numbering if present
            prompt = re.sub(r'^\d+\.\s*', '', prompt)
            
            examples.append((prompt, response))
    
    return examples

# For now, manually create the file with the examples
# This script will be used if needed
print("Parser ready")

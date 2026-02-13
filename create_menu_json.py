import re
import json
import os

INPUT_FILE = "data/carnivore_menu.txt"
OUTPUT_FILE = "data/menu.json"

def clean_price(p):
    return p.replace('£', '').replace(')', '').replace('(', '').strip()

def generate_id(name):
    return name.lower().replace(" ", "_").replace("'", "").replace("-", "_").replace("(", "").replace(")", "")

def parse_menu():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    menu_items = []
    current_category = "General"

    # Regex for "Name (Price): Description"
    # Example: Chicken Soup (£7.99): A comforting...
    # Also handle "Name (Price)" without description if any
    pattern = re.compile(r"^(.+?)\s*\((.+?)\)(?::\s*(.*))?$")

    for line in lines:
        line = line.strip()
        if not line: continue

        if line.startswith("---") and line.endswith("---"):
            current_category = line.replace("---", "").strip()
            continue

        match = pattern.match(line)
        if match:
            name = match.group(1).strip()
            price_raw = match.group(2).strip()
            description = match.group(3).strip() if match.group(3) else ""
            
            item_id = generate_id(name)
            price_val = clean_price(price_raw)

            item = {
                "id": item_id,
                "name": name,
                "price": price_val,
                "description": description,
                "category": current_category,
                "tags": [current_category.lower()] # simple tagging
            }
            menu_items.append(item)
            print(f"Parsed: {name} [{item_id}]")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(menu_items, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully generated {OUTPUT_FILE} with {len(menu_items)} items.")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    parse_menu()

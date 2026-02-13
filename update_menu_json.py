import json
import os
import re

FETCHED_MENU_PATH = r"d:\Aureeq_multimodel\fetched_menu.json"
TARGET_MENU_PATH = r"d:\Aureeq_multimodel\model_training\data\menu.json"

def slugify(text):
    text = text.lower()
    return re.sub(r'[\W_]+', '_', text).strip('_')

def main():
    print(f"Reading from {FETCHED_MENU_PATH}")
    with open(FETCHED_MENU_PATH, 'r', encoding='utf-8') as f:
        fetched_data = json.load(f)

    new_menu = []
    
    for category, items in fetched_data.items():
        for item in items:
            # Create a new item dict matching the target schema
            new_item = {
                "id": slugify(f"{category}_{item['name']}"),
                "name": item['name'],
                "price": item['price'],
                "description": item['description'],
                "category": category,
                "tags": [slugify(category), "food"]  # Basic tags
            }
            new_menu.append(new_item)
            
    print(f"converted {len(new_menu)} items.")
    
    # Backup old menu just in case
    if os.path.exists(TARGET_MENU_PATH):
        import shutil
        backup_path = TARGET_MENU_PATH + ".bak"
        shutil.copy2(TARGET_MENU_PATH, backup_path)
        print(f"Backed up old menu to {backup_path}")

    with open(TARGET_MENU_PATH, 'w', encoding='utf-8') as f:
        json.dump(new_menu, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully wrote new menu to {TARGET_MENU_PATH}")

if __name__ == "__main__":
    main()

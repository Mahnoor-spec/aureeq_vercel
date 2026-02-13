import json

try:
    with open('pages.json', 'r', encoding='utf-8') as f:
        pages = json.load(f)
        for page in pages:
            print(f"ID: {page['id']}, Title: {page['title']['rendered']}, Link: {page['link']}")
except Exception as e:
    print(f"Error: {e}")

import json
import html

def clean_text(text):
    """Clean HTML tags and entities from text."""
    if not text:
        return ""
    # Remove HTML tags (simple approach)
    import re
    text = re.sub(r'<[^>]+>', '', text)
    # Decode entities
    text = html.unescape(text)
    return text.strip()

try:
    with open('wc_products.json', 'r', encoding='utf-8') as f:
        products = json.load(f)

    menu = {}

    for product in products:
        # Get categories
        categories = product.get('categories', [])
        if not categories:
            category_name = "Uncategorized"
        else:
            # Use the first category as the main heading
            category_name = categories[0]['name']

        if category_name not in menu:
            menu[category_name] = []

        # Get details
        name = product.get('name', 'Unknown')
        
        # Price logic (handles variations or simple price)
        price = "N/A"
        prices = product.get('prices', {})
        if prices:
            price_val = prices.get('price', '')
            currency_symbol = prices.get('currency_symbol', '£')
            # Look for formatted price if available, or construct it
            # wc/store/products usually has 'prices' object
            # Let's try to find a readable price. 
            # Often it has 'price_range' or just 'price'.
            # Based on standard WC Store API:
            # price is entered in minor units usually? No, usually formatted string is available.
            # Let's verify structure by printing one item if needed, but for now generic:
            if 'price' in prices: # raw value usually
                 # check if it's string or int
                 p = prices['price']
                 # Try to find formatted
                 # Actually, let's just use what we find. 
                 # Wait, 'prices' usually has 'price', 'regular_price', 'sale_price', 'price_range'
                 # 'currency_symbol' is usually '£'
                 
                 # Store API usually returns: 
                 # "prices": { "price": "1000", "regular_price": "1000", "sale_price": "1000", "price_range": null, "currency_code": "GBP", "currency_symbol": "£", "currency_minor_unit": 2, "currency_decimal_separator": ".", "currency_thousand_separator": ",", "currency_prefix": "£", "currency_suffix": "" }
                 # The values are in minor units (pence).

                 val = int(p) / 100
                 price = f"{currency_symbol}{val:.2f}"
        
        description = clean_text(product.get('description', ''))
        short_description = clean_text(product.get('short_description', ''))
        
        # Use full description if available, else short
        full_desc = description if description else short_description

        menu[category_name].append({
            "name": name,
            "price": price,
            "description": full_desc
        })

    # Output to files
    # Text format
    with open('fetched_menu.txt', 'w', encoding='utf-8') as f:
        for category, items in menu.items():
            f.write(f"## {category}\n")
            for item in items:
                # Price is already formatted in the loop above as f"{currency_symbol}{val:.2f}"
                # e.g. "£3.50"
                price = item['price']
                
                f.write(f"- **Dish**: {item['name']}\n")
                f.write(f"  **Price**: {price}\n")
                f.write(f"  **Description**: {item['description']}\n")
                f.write("\n")
    
    # JSON format
    with open('fetched_menu.json', 'w', encoding='utf-8') as f:
        json.dump(menu, f, indent=4, ensure_ascii=False)

    print("Menu saved to fetched_menu.txt and fetched_menu.json")

except Exception as e:
    print(f"Error parsing menu: {e}")

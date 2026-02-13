import os
import sys

source_file = r"d:\AI sales agent\INGRIDIENTS 2.xlsx"
output_file = r"d:\AI sales agent\model_training\data\ingredients.txt"

print(f"Starting conversion...")
print(f"Source: {source_file}")
print(f"Output: {output_file}")

try:
    if not os.path.exists(source_file):
        print(f"ERROR: Source file does not exist!")
        sys.exit(1)
        
    import openpyxl
    print("Imported openpyxl successfully.")
    
    wb = openpyxl.load_workbook(source_file)
    print("Loaded workbook.")
    
    sheet = wb.active
    print(f"Active sheet: {sheet.title}")
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# DISH INGREDIENTS DATA\n\n")
        for row in sheet.iter_rows(values_only=True):
            # Filter None
            row_data = [str(c).strip() for c in row if c is not None]
            if row_data:
                line = " | ".join(row_data)
                f.write(f"- {line}\n")
                print(f"Processed row: {line[:50]}...")
                count += 1
                
    print(f"Done. Wrote {count} lines to {output_file}")

except Exception as e:
    print(f"FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()

import os
import csv
import json
import glob

ROOT_DIR = "sandbox/artifacts"
OUTPUT_FILE = "gold_standard.json"

def calculate_metrics(file_path):
    ages = []
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['AGE_YRS']:
                    ages.append(float(row['AGE_YRS']))
        
        if not ages:
            return None

        avg_age = sum(ages) / len(ages)
        count = len(ages)
        
        return {
            "average_age": round(avg_age, 2), 
            "total_customers": count
        }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    gold_data = {}
    search_pattern = os.path.join(ROOT_DIR, "*", "*", "crm_export.csv")
    files = glob.glob(search_pattern)

    print(f"Found {len(files)} CSV files to process...")

    for file_path in files:
        path_parts = file_path.split(os.sep)
        key = os.path.relpath(file_path, start=ROOT_DIR)
        metrics = calculate_metrics(file_path)
        
        if metrics:
            gold_data[key] = metrics

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(gold_data, f, indent=4)
    
    print(f"Success! Gold standard saved to {OUTPUT_FILE} with {len(gold_data)} entries.")

if __name__ == "__main__":
    main()
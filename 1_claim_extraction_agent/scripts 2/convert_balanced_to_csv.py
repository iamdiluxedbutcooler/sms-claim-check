#!/usr/bin/env python3
"""
Convert balanced Label Studio JSON dataset to CSV format.
"""

import json
import pandas as pd
from pathlib import Path

def convert_to_csv(json_path, csv_path):
    """Convert Label Studio JSON to CSV format"""
    print(f"Loading dataset from {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    # Extract data into rows
    rows = []
    for task in tasks:
        text = task['data']['text']
        meta = task['meta']
        
        # Determine label
        label = meta.get('label', 'phishing')
        if label != 'ham':
            label = 'phishing'
        
        # Get additional metadata
        is_augmented = meta.get('is_augmented', False)
        source = meta.get('source', 'unknown')
        augmentation_type = meta.get('augmentation_type', 'N/A')
        
        row = {
            'text': text,
            'label': label,
            'is_augmented': is_augmented,
            'source': source,
            'augmentation_type': augmentation_type if is_augmented else 'original'
        }
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    print(f"Saving to {csv_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # Print statistics
    print("\n" + "="*60)
    print("CSV DATASET CREATED")
    print("="*60)
    print(f"Total rows: {len(df)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nAugmentation status:")
    print(df['is_augmented'].value_counts())
    print(f"\nSource distribution:")
    print(df['source'].value_counts())
    print(f"\nFile size: {csv_path.stat().st_size / 1024:.1f} KB")
    print("="*60)
    
    return df

def main():
    # Input and output paths
    json_path = Path("data/annotations/balanced_dataset_2000.json")
    csv_path = Path("data/processed/sms_phishing_ham_balanced_2000.csv")
    
    # Convert
    df = convert_to_csv(json_path, csv_path)
    
    # Show sample
    print("\nSample rows:")
    print(df.head(3).to_string())
    print("\n...")
    print(df.tail(3).to_string())

if __name__ == "__main__":
    main()

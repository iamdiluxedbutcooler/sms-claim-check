#!/usr/bin/env python3
"""
Add 1000 HAM (benign) messages from Mendeley dataset to create balanced dataset.
Combines with augmented phishing messages for 2000 total messages.
"""

import pandas as pd
import json
import random
from pathlib import Path
from datetime import datetime

# Set random seed for reproducibility
random.seed(42)

def load_mendeley_ham(csv_path):
    """Load HAM messages from Mendeley dataset"""
    print(f"Loading Mendeley dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter HAM messages
    ham_df = df[df['LABEL'].str.lower() == 'ham'].copy()
    print(f"Found {len(ham_df)} HAM messages in Mendeley dataset")
    
    return ham_df

def sample_ham_messages(ham_df, n=1000):
    """Sample n HAM messages"""
    if len(ham_df) < n:
        print(f"WARNING: Only {len(ham_df)} HAM messages available, using all of them")
        sampled = ham_df
    else:
        sampled = ham_df.sample(n=n, random_state=42)
        print(f"Sampled {n} HAM messages")
    
    return sampled

def create_label_studio_format(ham_df, start_id=1001):
    """Convert HAM messages to Label Studio format with 'O' labels"""
    tasks = []
    
    for idx, row in enumerate(ham_df.itertuples(), start=start_id):
        text = row.TEXT.strip()
        
        # Create task with all 'O' labels (no entities)
        task = {
            "id": idx,
            "data": {
                "text": text
            },
            "annotations": [{
                "id": idx,
                "result": [],  # Empty result = all tokens are 'O'
                "was_cancelled": False,
                "ground_truth": False,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "lead_time": 0
            }],
            "meta": {
                "is_augmented": False,
                "source": "mendeley_ham",
                "label": "ham"
            }
        }
        
        tasks.append(task)
    
    return tasks

def combine_with_phishing(phishing_path, ham_tasks):
    """Combine phishing and HAM messages into balanced dataset"""
    print(f"\nLoading phishing dataset from {phishing_path}")
    
    with open(phishing_path, 'r', encoding='utf-8') as f:
        phishing_tasks = json.load(f)
    
    print(f"Phishing messages: {len(phishing_tasks)}")
    print(f"HAM messages: {len(ham_tasks)}")
    
    # Combine and shuffle
    all_tasks = phishing_tasks + ham_tasks
    random.shuffle(all_tasks)
    
    print(f"Total balanced dataset: {len(all_tasks)} messages")
    
    return all_tasks

def main():
    # Paths
    mendeley_csv = Path("data/raw/mendeley.csv")
    phishing_json = Path("data/annotations/augmented_phishing_1000.json")
    output_json = Path("data/annotations/balanced_dataset_2000.json")
    metadata_json = Path("data/annotations/balanced_dataset_2000_metadata.json")
    
    # Load and sample HAM messages
    ham_df = load_mendeley_ham(mendeley_csv)
    sampled_ham = sample_ham_messages(ham_df, n=1000)
    
    # Convert to Label Studio format
    ham_tasks = create_label_studio_format(sampled_ham)
    
    # Combine with phishing messages
    balanced_tasks = combine_with_phishing(phishing_json, ham_tasks)
    
    # Save balanced dataset
    print(f"\nSaving balanced dataset to {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(balanced_tasks, f, indent=2, ensure_ascii=False)
    
    # Calculate statistics
    phishing_count = sum(1 for task in balanced_tasks if task['meta'].get('label') != 'ham')
    ham_count = sum(1 for task in balanced_tasks if task['meta'].get('label') == 'ham')
    
    augmented_count = sum(1 for task in balanced_tasks 
                          if task['meta'].get('is_augmented') and task['meta'].get('label') != 'ham')
    original_phishing_count = phishing_count - augmented_count
    
    # Save metadata
    metadata = {
        "total_messages": len(balanced_tasks),
        "phishing_messages": phishing_count,
        "ham_messages": ham_count,
        "original_phishing": original_phishing_count,
        "augmented_phishing": augmented_count,
        "balance_ratio": f"{phishing_count}:{ham_count}",
        "generation_date": datetime.now().isoformat(),
        "source_files": {
            "phishing": str(phishing_json),
            "ham": str(mendeley_csv)
        }
    }
    
    with open(metadata_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("BALANCED DATASET CREATED SUCCESSFULLY")
    print("="*60)
    print(f"Total messages: {len(balanced_tasks)}")
    print(f"  - Phishing: {phishing_count} ({phishing_count/len(balanced_tasks)*100:.1f}%)")
    print(f"    - Original: {original_phishing_count}")
    print(f"    - Augmented: {augmented_count}")
    print(f"  - HAM (benign): {ham_count} ({ham_count/len(balanced_tasks)*100:.1f}%)")
    print(f"\nBalance ratio: {phishing_count}:{ham_count}")
    print(f"Output file: {output_json} ({output_json.stat().st_size / 1024:.1f} KB)")
    print(f"Metadata file: {metadata_json}")
    print("="*60)

if __name__ == "__main__":
    main()

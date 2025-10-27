import os
import json
import csv
import random
from pathlib import Path
from typing import List, Dict
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": {
        "ham": 3875,
        "smishing": 510,
        "spam": 391
    },
    "test": {
        "ham": 969,
        "smishing": 128,
        "spam": 98
    }
}

ANNOTATION_TARGETS = {
    "smishing": 510,
    "ham": 100,
    "spam": 50
}

def download_dataset():
    print("\n" + "="*60)
    print("MENDELEY SMS DATASET DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nThe Mendeley SMS dataset requires manual download.")
    print("\nSteps:")
    print("1. Visit: https://data.mendeley.com/datasets/f45bkkt8pr/1")
    print("2. Download the dataset in CSV format")
    print("3. Place the CSV file in: data/raw/")
    print("4. Name it: sms_dataset.csv")
    print("\nExpected format:")
    print("  - Column 1: 'text' or 'message'")
    print("  - Column 2: 'label' or 'type'")
    print(f"\nPath: {RAW_DIR.absolute()}/sms_dataset.csv")
    print("\n" + "="*60 + "\n")
    
    dataset_path = RAW_DIR / "sms_dataset.csv"
    if dataset_path.exists():
        print(f"Dataset found at: {dataset_path}")
        return True
    else:
        print(f"Dataset not found. Please download it first.")
        return False

def load_dataset(filepath: Path) -> List[Dict]:
    messages = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = row.get('text') or row.get('message') or row.get('Text') or row.get('Message')
            label = row.get('label') or row.get('type') or row.get('Label') or row.get('Type')
            
            if text and label:
                messages.append({
                    'message_id': f"msg_{i:06d}",
                    'text': text.strip(),
                    'label': label.strip().lower()
                })
    
    print(f"\nLoaded {len(messages)} messages from dataset")
    return messages

def split_by_label(messages: List[Dict]) -> Dict[str, List[Dict]]:
    splits = {
        'ham': [],
        'spam': [],
        'smishing': []
    }
    
    for msg in messages:
        label = msg['label']
        if label in splits:
            splits[label].append(msg)
        elif label == 'phishing':
            splits['smishing'].append(msg)
    
    print("\nDataset Distribution:")
    for label, msgs in splits.items():
        print(f"  {label.capitalize()}: {len(msgs)} messages")
    
    return splits

def create_annotation_set(splits: Dict[str, List[Dict]]) -> List[Dict]:
    annotation_set = []
    
    smishing = splits['smishing'][:ANNOTATION_TARGETS['smishing']]
    annotation_set.extend(smishing)
    print(f"\nAdded {len(smishing)} smishing messages")
    
    ham = random.sample(splits['ham'], min(ANNOTATION_TARGETS['ham'], len(splits['ham'])))
    annotation_set.extend(ham)
    print(f"Added {len(ham)} ham messages")
    
    spam = random.sample(splits['spam'], min(ANNOTATION_TARGETS['spam'], len(splits['spam'])))
    annotation_set.extend(spam)
    print(f"Added {len(spam)} spam messages")
    
    random.shuffle(annotation_set)
    
    print(f"\nTotal annotation set: {len(annotation_set)} messages")
    return annotation_set

def split_annotation_set(messages: List[Dict]) -> Dict[str, List[Dict]]:
    total = len(messages)
    train_size = int(total * 0.67)
    val_size = int(total * 0.17)
    
    train = messages[:train_size]
    val = messages[train_size:train_size + val_size]
    test = messages[train_size + val_size:]
    
    print("\nAnnotation Split:")
    print(f"  Train: {len(train)} messages (67%)")
    print(f"  Validation: {len(val)} messages (17%)")
    print(f"  Test: {len(test)} messages (16%)")
    
    return {
        'train': train,
        'val': val,
        'test': test
    }

def save_dataset(messages: List[Dict], filepath: Path):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {filepath}")

def create_pilot_set(messages: List[Dict], n: int = 20) -> List[Dict]:
    pilot = random.sample(messages, min(n, len(messages)))
    return pilot

def main():
    parser = argparse.ArgumentParser(description="Prepare SMS dataset for annotation")
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--pilot', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("\n" + "="*60)
    print("SMS PHISHING ANNOTATION - DATA PREPARATION")
    print("="*60)
    
    if args.download or not (RAW_DIR / "sms_dataset.csv").exists():
        if not download_dataset():
            return
    
    dataset_path = RAW_DIR / "sms_dataset.csv"
    if not dataset_path.exists():
        print(f"\nError: Dataset not found at {dataset_path}")
        print("Run with --download flag for instructions.")
        return
    
    messages = load_dataset(dataset_path)
    splits = split_by_label(messages)
    annotation_set = create_annotation_set(splits)
    
    save_dataset(annotation_set, PROCESSED_DIR / "annotation_set.json")
    
    pilot_set = create_pilot_set(annotation_set, args.pilot)
    save_dataset(pilot_set, PROCESSED_DIR / "pilot_set.json")
    print(f"\nCreated pilot set: {len(pilot_set)} messages")
    
    if args.split:
        splits_dict = split_annotation_set(annotation_set)
        for split_name, split_messages in splits_dict.items():
            save_dataset(split_messages, PROCESSED_DIR / f"{split_name}_set.json")
    
    metadata = {
        "total_messages": len(annotation_set),
        "smishing": len([m for m in annotation_set if m['label'] == 'smishing']),
        "ham": len([m for m in annotation_set if m['label'] == 'ham']),
        "spam": len([m for m in annotation_set if m['label'] == 'spam']),
        "random_seed": args.seed,
        "pilot_set_size": len(pilot_set)
    }
    
    with open(PROCESSED_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nNext: python scripts/convert_to_label_studio.py")
    print()

if __name__ == "__main__":
    main()

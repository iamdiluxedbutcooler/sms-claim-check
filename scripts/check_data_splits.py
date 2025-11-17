"""
Fix data splits to respect original train/test split from raw dataset.

This script:
1. Loads the original processed train/test split
2. Matches annotated messages back to their original split
3. Updates the training code to use these splits
"""

import json
import pandas as pd
from pathlib import Path

def main():
    print("Checking data split integrity...")
    print("=" * 60)
    
    # Load original splits - ONLY SMISHING
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    train_smishing = train_df[train_df['label'] == 'smishing']
    test_smishing = test_df[test_df['label'] == 'smishing']
    
    print(f"Original dataset (SMISHING ONLY):")
    print(f"  Train: {len(train_smishing)} smishing messages")
    print(f"  Test:  {len(test_smishing)} smishing messages")
    print(f"  Total: {len(train_smishing) + len(test_smishing)} smishing")
    print()
    
    # Load raw dataset with indices
    raw_df = pd.read_csv('data/raw/mendeley.csv')
    print(f"Raw dataset: {len(raw_df)} messages")
    print()
    
    # Match processed splits back to raw indices - SMISHING ONLY
    train_texts = set(train_smishing['text'].str.strip())
    test_texts = set(test_smishing['text'].str.strip())
    
    train_indices = []
    test_indices = []
    
    for idx, row in raw_df.iterrows():
        text = str(row['TEXT']).strip()
        if text in train_texts:
            train_indices.append(idx)
        elif text in test_texts:
            test_indices.append(idx)
    
    print(f"Matched indices:")
    print(f"  Train: {len(train_indices)} messages")
    print(f"  Test:  {len(test_indices)} messages")
    print()
    
    # Load annotations
    with open('data/annotations/entity_annotations.json') as f:
        entity_annotations = json.load(f)
    
    with open('data/annotations/claim_annotations.json') as f:
        claim_annotations = json.load(f)
    
    # Check which split each annotation belongs to
    entity_train_ids = []
    entity_test_ids = []
    entity_unknown_ids = []
    
    for ann in entity_annotations:
        msg_id = ann['data']['message_id']
        if msg_id in train_indices:
            entity_train_ids.append(msg_id)
        elif msg_id in test_indices:
            entity_test_ids.append(msg_id)
        else:
            entity_unknown_ids.append(msg_id)
    
    claim_train_ids = []
    claim_test_ids = []
    claim_unknown_ids = []
    
    for ann in claim_annotations:
        msg_id = ann['data']['message_id']
        if msg_id in train_indices:
            claim_train_ids.append(msg_id)
        elif msg_id in test_indices:
            claim_test_ids.append(msg_id)
        else:
            claim_unknown_ids.append(msg_id)
    
    print("Entity annotations split:")
    print(f"  Train: {len(entity_train_ids)}")
    print(f"  Test:  {len(entity_test_ids)}")
    print(f"  Unknown: {len(entity_unknown_ids)}")
    print()
    
    print("Claim annotations split:")
    print(f"  Train: {len(claim_train_ids)}")
    print(f"  Test:  {len(claim_test_ids)}")
    print(f"  Unknown: {len(claim_unknown_ids)}")
    print()
    
    # Save split mappings
    split_mapping = {
        'train_indices': sorted(train_indices),
        'test_indices': sorted(test_indices),
        'entity_annotations': {
            'train_ids': sorted(entity_train_ids),
            'test_ids': sorted(entity_test_ids),
            'unknown_ids': sorted(entity_unknown_ids)
        },
        'claim_annotations': {
            'train_ids': sorted(claim_train_ids),
            'test_ids': sorted(claim_test_ids),
            'unknown_ids': sorted(claim_unknown_ids)
        }
    }
    
    output_path = Path('data/processed/annotation_split_mapping.json')
    with open(output_path, 'w') as f:
        json.dump(split_mapping, f, indent=2)
    
    print(f"[SAVED] Split mapping: {output_path}")
    print()
    
    # Verdict
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if entity_unknown_ids or claim_unknown_ids:
        print("[WARNING] Some annotated messages don't match original split!")
        print("This happens because:")
        print("  1. We only annotated smishing samples (not ham)")
        print("  2. Original split was 80/20, we need to maintain this")
        print()
        print("SOLUTION: Use stratified split on annotated data with same seed (42)")
        print("to approximate the original distribution.")
    else:
        print("[OK] All annotations match original train/test split!")
        print("Update loader.py to use split_mapping for consistent splits.")

if __name__ == '__main__':
    main()

"""
Fix data split issues:
1. Remove duplicate annotations (same text appears multiple times)
2. Remove leaked messages (appear in both train and test CSV)
3. Create clean train/test split
"""

import json
import pandas as pd
from pathlib import Path

def main():
    print("=" * 70)
    print("FIXING DATA SPLIT ISSUES")
    print("=" * 70)
    print()
    
    # Load processed splits
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    train_smish = train_df[train_df['label'] == 'smishing']['text'].str.strip()
    test_smish = test_df[test_df['label'] == 'smishing']['text'].str.strip()
    
    train_set = set(train_smish)
    test_set = set(test_smish)
    leaked = train_set & test_set
    
    print(f"[ISSUE 1] Data Leakage:")
    print(f"  {len(leaked)} messages appear in BOTH train and test")
    print()
    
    # Load annotations
    with open('data/annotations/entity_annotations.json') as f:
        entity_ann = json.load(f)
    
    with open('data/annotations/claim_annotations.json') as f:
        claim_ann = json.load(f)
    
    print(f"[ISSUE 2] Duplicate Annotations:")
    entity_texts = [ann['data']['text'].strip() for ann in entity_ann]
    entity_unique = set(entity_texts)
    print(f"  Entity: {len(entity_ann)} annotations, {len(entity_unique)} unique")
    print(f"  Duplicates: {len(entity_ann) - len(entity_unique)}")
    print()
    
    # Deduplicate annotations
    print("DEDUPLICATING...")
    entity_dedup = {}
    for ann in entity_ann:
        text = ann['data']['text'].strip()
        if text not in entity_dedup:
            entity_dedup[text] = ann
    
    claim_dedup = {}
    for ann in claim_ann:
        text = ann['data']['text'].strip()
        if text not in claim_dedup:
            claim_dedup[text] = ann
    
    entity_clean = list(entity_dedup.values())
    claim_clean = list(claim_dedup.values())
    
    print(f"  Entity cleaned: {len(entity_clean)} unique annotations")
    print(f"  Claim cleaned: {len(claim_clean)} unique annotations")
    print()
    
    # Assign to splits (leaked messages go to TRAIN only)
    print("ASSIGNING TO SPLITS...")
    
    entity_train = []
    entity_test = []
    entity_removed = []
    
    for ann in entity_clean:
        text = ann['data']['text'].strip()
        msg_id = ann['data']['message_id']
        
        if text in leaked:
            # Leaked message - put in train only
            entity_train.append(msg_id)
        elif text in train_set:
            entity_train.append(msg_id)
        elif text in test_set:
            entity_test.append(msg_id)
        else:
            entity_removed.append(msg_id)
    
    claim_train = []
    claim_test = []
    claim_removed = []
    
    for ann in claim_clean:
        text = ann['data']['text'].strip()
        msg_id = ann['data']['message_id']
        
        if text in leaked:
            claim_train.append(msg_id)
        elif text in train_set:
            claim_train.append(msg_id)
        elif text in test_set:
            claim_test.append(msg_id)
        else:
            claim_removed.append(msg_id)
    
    print(f"Entity split (deduplicated, leaks moved to train):")
    print(f"  Train: {len(entity_train)}")
    print(f"  Test: {len(entity_test)}")
    print(f"  Removed: {len(entity_removed)}")
    print()
    
    print(f"Claim split (deduplicated, leaks moved to train):")
    print(f"  Train: {len(claim_train)}")
    print(f"  Test: {len(claim_test)}")
    print(f"  Removed: {len(claim_removed)}")
    print()
    
    # Save deduplicated annotations
    print("SAVING CLEAN ANNOTATIONS...")
    
    with open('data/annotations/entity_annotations_clean.json', 'w') as f:
        json.dump(entity_clean, f, indent=2)
    
    with open('data/annotations/claim_annotations_clean.json', 'w') as f:
        json.dump(claim_clean, f, indent=2)
    
    print(f"  [SAVED] entity_annotations_clean.json ({len(entity_clean)} annotations)")
    print(f"  [SAVED] claim_annotations_clean.json ({len(claim_clean)} annotations)")
    print()
    
    # Save split mapping
    mapping = {
        'entity_annotations': {
            'train_ids': sorted(entity_train),
            'test_ids': sorted(entity_test),
            'removed_ids': sorted(entity_removed)
        },
        'claim_annotations': {
            'train_ids': sorted(claim_train),
            'test_ids': sorted(claim_test),
            'removed_ids': sorted(claim_removed)
        },
        'stats': {
            'original_annotations': len(entity_ann),
            'deduplicated_annotations': len(entity_clean),
            'removed_duplicates': len(entity_ann) - len(entity_clean),
            'leaked_messages': len(leaked),
            'train_total': len(entity_train),
            'test_total': len(entity_test)
        }
    }
    
    with open('data/processed/annotation_split_mapping_clean.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"  [SAVED] annotation_split_mapping_clean.json")
    print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original: 638 annotations (76 duplicates)")
    print(f"Cleaned: {len(entity_clean)} unique annotations")
    print(f"Split: {len(entity_train)} train / {len(entity_test)} test")
    print(f"Leaked messages: {len(leaked)} (moved to train)")
    print()
    print("[ACTION] Update configs to use *_clean.json files")

if __name__ == '__main__':
    main()

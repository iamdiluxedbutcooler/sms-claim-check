"""
Properly match annotations to train/test split by checking which CSV they came from.
"""

import json
import pandas as pd

def main():
    print("Matching annotations to train/test CSVs...")
    print("=" * 60)
    
    # Load splits
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    train_smishing = train_df[train_df['label'] == 'smishing']
    test_smishing = test_df[test_df['label'] == 'smishing']
    
    print(f"Train: {len(train_smishing)} smishing")
    print(f"Test: {len(test_smishing)} smishing")
    print()
    
    # Create text to split mapping (handle duplicates by keeping first occurrence)
    text_to_split = {}
    
    for idx, text in enumerate(train_smishing['text'].str.strip()):
        if text not in text_to_split:
            text_to_split[text] = 'train'
    
    for idx, text in enumerate(test_smishing['text'].str.strip()):
        if text not in text_to_split:
            text_to_split[text] = 'test'
    
    # Load annotations
    with open('data/annotations/entity_annotations.json') as f:
        entity_ann = json.load(f)
    
    with open('data/annotations/claim_annotations.json') as f:
        claim_ann = json.load(f)
    
    # Match annotations
    entity_train = []
    entity_test = []
    entity_nomatch = []
    
    for ann in entity_ann:
        text = ann['data']['text'].strip()
        msg_id = ann['data']['message_id']
        
        split = text_to_split.get(text)
        if split == 'train':
            entity_train.append(msg_id)
        elif split == 'test':
            entity_test.append(msg_id)
        else:
            entity_nomatch.append(msg_id)
    
    claim_train = []
    claim_test = []
    claim_nomatch = []
    
    for ann in claim_ann:
        text = ann['data']['text'].strip()
        msg_id = ann['data']['message_id']
        
        split = text_to_split.get(text)
        if split == 'train':
            claim_train.append(msg_id)
        elif split == 'test':
            claim_test.append(msg_id)
        else:
            claim_nomatch.append(msg_id)
    
    print(f"Entity annotations:")
    print(f"  Train: {len(entity_train)}")
    print(f"  Test: {len(entity_test)}")
    print(f"  No match: {len(entity_nomatch)}")
    print()
    
    print(f"Claim annotations:")
    print(f"  Train: {len(claim_train)}")
    print(f"  Test: {len(claim_test)}")
    print(f"  No match: {len(claim_nomatch)}")
    print()
    
    # Save mapping
    mapping = {
        'entity_annotations': {
            'train_ids': sorted(entity_train),
            'test_ids': sorted(entity_test),
            'no_match_ids': sorted(entity_nomatch)
        },
        'claim_annotations': {
            'train_ids': sorted(claim_train),
            'test_ids': sorted(claim_test),
            'no_match_ids': sorted(claim_nomatch)
        },
        'notes': {
            'train_smishing_total': len(train_smishing),
            'test_smishing_total': len(test_smishing),
            'matched_train': len(entity_train),
            'matched_test': len(entity_test)
        }
    }
    
    with open('data/processed/annotation_split_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print("[SAVED] annotation_split_mapping.json")
    print()
    
    if len(entity_train) + len(entity_test) == 638:
        print(f"[OK] All 638 annotations matched!")
        print(f"     Train: {len(entity_train)} ({len(entity_train)/638*100:.1f}%)")
        print(f"     Test: {len(entity_test)} ({len(entity_test)/638*100:.1f}%)")
    else:
        print(f"[WARNING] {len(entity_nomatch)} annotations didn't match!")

if __name__ == '__main__':
    main()

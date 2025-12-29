#!/usr/bin/env python3
"""
Remove near-duplicate messages from claim_annotations_2000.json
Keeps only the first occurrence of each near-duplicate group
"""

import json
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

def find_all_near_duplicates(data, threshold=0.98):
    """Find all near-duplicate groups"""
    print("Finding near-duplicates...")
    
    # Track which messages to keep/remove
    to_remove = set()
    duplicate_groups = []
    
    texts = [(i, entry['data']['text'].strip().lower(), entry) for i, entry in enumerate(data)]
    
    checked = set()
    
    for i in range(len(texts)):
        if i in to_remove:
            continue
            
        group = [i]  # Start group with current index
        idx1, text1, _ = texts[i]
        
        for j in range(i + 1, len(texts)):
            if j in to_remove:
                continue
                
            if (i, j) in checked:
                continue
                
            idx2, text2, _ = texts[j]
            
            similarity = SequenceMatcher(None, text1, text2).ratio()
            
            if similarity >= threshold:
                group.append(j)
                to_remove.add(j)  # Mark for removal (keep first one only)
                checked.add((i, j))
        
        if len(group) > 1:
            duplicate_groups.append(group)
    
    return duplicate_groups, to_remove

def deduplicate_dataset(input_file, output_file, threshold=0.98):
    """Remove near-duplicates from dataset"""
    
    print("="*70)
    print("DEDUPLICATING CLAIM ANNOTATIONS DATASET")
    print("="*70)
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nOriginal dataset: {len(data)} messages")
    
    # Count HAM vs SMISH before
    ham_before = 0
    smish_before = 0
    
    for entry in data:
        has_claims = False
        if entry.get('annotations') and len(entry['annotations']) > 0:
            annotations = entry['annotations'][0]
            if 'result' in annotations and annotations['result']:
                has_claims = True
        
        if has_claims:
            smish_before += 1
        else:
            ham_before += 1
    
    print(f"  HAM: {ham_before}")
    print(f"  SMISH: {smish_before}")
    
    # Find duplicates
    duplicate_groups, to_remove_indices = find_all_near_duplicates(data, threshold)
    
    print(f"\nFound {len(duplicate_groups)} near-duplicate groups")
    print(f"Removing {len(to_remove_indices)} duplicate entries")
    
    # Show some examples
    print("\nExample duplicate groups (showing first 5):")
    for i, group in enumerate(duplicate_groups[:5], 1):
        print(f"\n{i}. Group of {len(group)} similar messages:")
        for idx in group[:3]:  # Show first 3 in group
            text = data[idx]['data']['text']
            entry_id = data[idx].get('id', 'unknown')
            status = "KEEP" if idx == group[0] else "REMOVE"
            print(f"   [{status}] [{idx}] {entry_id}: {text[:70]}...")
    
    # Create deduplicated dataset
    deduped_data = []
    removed_ham = 0
    removed_smish = 0
    
    for idx, entry in enumerate(data):
        if idx not in to_remove_indices:
            deduped_data.append(entry)
        else:
            # Count what we're removing
            has_claims = False
            if entry.get('annotations') and len(entry['annotations']) > 0:
                annotations = entry['annotations'][0]
                if 'result' in annotations and annotations['result']:
                    has_claims = True
            
            if has_claims:
                removed_smish += 1
            else:
                removed_ham += 1
    
    # Count after deduplication
    ham_after = 0
    smish_after = 0
    
    for entry in deduped_data:
        has_claims = False
        if entry.get('annotations') and len(entry['annotations']) > 0:
            annotations = entry['annotations'][0]
            if 'result' in annotations and annotations['result']:
                has_claims = True
        
        if has_claims:
            smish_after += 1
        else:
            ham_after += 1
    
    # Save deduplicated data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deduped_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("DEDUPLICATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nRemoved: {len(to_remove_indices)} duplicate messages")
    print(f"  - HAM removed: {removed_ham}")
    print(f"  - SMISH removed: {removed_smish}")
    
    print(f"\nRemaining: {len(deduped_data)} unique messages")
    print(f"  - HAM: {ham_after}")
    print(f"  - SMISH: {smish_after}")
    
    print(f"\n{'='*70}")
    print("TO REACH 2000 MESSAGES (1000 HAM, 1000 SMISH):")
    print(f"{'='*70}")
    
    total_needed = 2000 - len(deduped_data)
    ham_needed = 1000 - ham_after
    smish_needed = 1000 - smish_after
    
    print(f"Total messages needed: {total_needed}")
    print(f"  - HAM needed: {ham_needed}")
    print(f"  - SMISH needed: {smish_needed}")
    
    print(f"\nAUGMENTATION REQUIRED:")
    if ham_needed > 0:
        print(f"  Generate {ham_needed} more HAM messages")
    if smish_needed > 0:
        print(f"  Generate {smish_needed} more SMISH messages")
    
    print(f"\nDeduplicated data saved to: {output_file}")
    print(f"{'='*70}")
    
    return {
        'original_count': len(data),
        'deduped_count': len(deduped_data),
        'removed_count': len(to_remove_indices),
        'ham_before': ham_before,
        'smish_before': smish_before,
        'ham_after': ham_after,
        'smish_after': smish_after,
        'ham_needed': ham_needed,
        'smish_needed': smish_needed,
        'total_needed': total_needed
    }

if __name__ == '__main__':
    input_file = Path(__file__).parent / 'data' / 'annotations' / 'claim_annotations_2000.json'
    output_file = Path(__file__).parent / 'data' / 'annotations' / 'claim_annotations_deduped.json'
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        exit(1)
    
    stats = deduplicate_dataset(input_file, output_file, threshold=0.98)

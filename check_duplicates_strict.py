#!/usr/bin/env python3
"""
Check for duplicate messages in claim_annotations_2000.json
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

def check_duplicates(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("="*70)
    print("DUPLICATE CHECK - claim_annotations_2000.json")
    print("="*70)
    
    # Track messages
    message_texts = []
    message_to_ids = defaultdict(list)
    
    for entry in data:
        text = entry['data']['text'].strip()
        entry_id = entry.get('id', 'unknown')
        
        message_texts.append(text)
        message_to_ids[text].append(entry_id)
    
    # Find duplicates
    text_counts = Counter(message_texts)
    duplicates = {text: count for text, count in text_counts.items() if count > 1}
    
    print(f"\nTotal messages: {len(data)}")
    print(f"Unique messages: {len(text_counts)}")
    print(f"Duplicate messages: {len(duplicates)}")
    
    if duplicates:
        print(f"\n{'='*70}")
        print(f"DUPLICATES FOUND: {len(duplicates)} messages")
        print(f"{'='*70}")
        
        total_duplicate_entries = sum(count for count in duplicates.values())
        entries_to_remove = total_duplicate_entries - len(duplicates)
        
        print(f"\nDuplicate entries to remove: {entries_to_remove}")
        
        for text, count in sorted(duplicates.items(), key=lambda x: -x[1])[:20]:
            print(f"\n[{count}x] {text[:100]}...")
            print(f"  IDs: {message_to_ids[text]}")
        
        if len(duplicates) > 20:
            print(f"\n... and {len(duplicates) - 20} more duplicates")
        
        # Check HAM vs SMISH distribution
        ham_count = 0
        smish_count = 0
        
        for entry in data:
            text = entry['data']['text'].strip()
            has_claims = False
            
            if entry.get('annotations') and len(entry['annotations']) > 0:
                annotations = entry['annotations'][0]
                if 'result' in annotations and annotations['result']:
                    has_claims = True
            
            if has_claims:
                smish_count += 1
            else:
                ham_count += 1
        
        print(f"\n{'='*70}")
        print(f"CURRENT DISTRIBUTION")
        print(f"{'='*70}")
        print(f"HAM: {ham_count}")
        print(f"SMISH: {smish_count}")
        print(f"Total: {len(data)}")
        
        # After removing duplicates
        unique_count = len(text_counts)
        after_removal = len(data) - entries_to_remove
        
        print(f"\n{'='*70}")
        print(f"AFTER REMOVING DUPLICATES")
        print(f"{'='*70}")
        print(f"Messages remaining: {after_removal}")
        print(f"Need to reach 2000 (1000 HAM, 1000 SMISH)")
        print(f"Need to add: {2000 - after_removal} messages")
        
        # Estimate how many of each type to add
        ham_needed = max(0, 1000 - ham_count + (entries_to_remove // 2))
        smish_needed = max(0, 1000 - smish_count + (entries_to_remove // 2))
        
        print(f"\nEstimated needed (rough):")
        print(f"  HAM: ~{ham_needed}")
        print(f"  SMISH: ~{smish_needed}")
        
        return duplicates, message_to_ids
    else:
        print("\nNo duplicates found!")
        return None, None

if __name__ == '__main__':
    data_file = Path(__file__).parent / 'data' / 'annotations' / 'claim_annotations_2000.json'
    
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        exit(1)
    
    duplicates, message_to_ids = check_duplicates(data_file)
    
    if duplicates:
        # Create deduplication script
        print(f"\n{'='*70}")
        print("Creating deduplication script...")
        print(f"{'='*70}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Keep first occurrence of each message
        seen = set()
        deduped_data = []
        removed_count = 0
        
        for entry in data:
            text = entry['data']['text'].strip()
            if text not in seen:
                seen.add(text)
                deduped_data.append(entry)
            else:
                removed_count += 1
        
        # Save deduped version
        output_file = data_file.parent / 'claim_annotations_deduped.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(deduped_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDeduplicated data saved to: {output_file}")
        print(f"Removed {removed_count} duplicate entries")
        print(f"Remaining: {len(deduped_data)} unique messages")

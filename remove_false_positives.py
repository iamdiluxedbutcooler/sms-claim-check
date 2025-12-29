#!/usr/bin/env python3
"""
Remove false positive annotations that were incorrectly added
"""

import json
from pathlib import Path

def remove_false_positives(input_file, output_file):
    """Remove false positive annotations"""
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load false positives
    with open('false_positives.json', 'r', encoding='utf-8') as f:
        false_positives = json.load(f)
    
    print("="*70)
    print("REMOVING FALSE POSITIVE ANNOTATIONS")
    print("="*70)
    
    removed_count = 0
    
    # Group by entry index
    fp_by_entry = {}
    for fp in false_positives:
        idx = fp['idx']
        if idx not in fp_by_entry:
            fp_by_entry[idx] = []
        fp_by_entry[idx].append(fp)
    
    print(f"\nProcessing {len(fp_by_entry)} entries with false positives...")
    
    for idx, fps in fp_by_entry.items():
        entry = data[idx]
        
        if not entry.get('annotations') or not entry['annotations']:
            continue
        
        annotations = entry['annotations'][0]
        if 'result' not in annotations or not annotations['result']:
            continue
        
        # Remove results that match false positives
        new_results = []
        for result in annotations['result']:
            value = result.get('value', {})
            text = value.get('text', '')
            start = value.get('start')
            
            # Check if this matches a false positive
            is_fp = False
            for fp in fps:
                if text == fp['text'] and start == fp['start']:
                    is_fp = True
                    removed_count += 1
                    print(f"  Removed: {entry.get('id')} - '{text[:50]}...'")
                    break
            
            if not is_fp:
                new_results.append(result)
        
        # Update entry
        annotations['result'] = new_results
        annotations['result_count'] = len(new_results)
    
    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"False positives removed: {removed_count}")
    print(f"Saved to: {output_file}")
    print(f"{'='*70}")

if __name__ == '__main__':
    input_file = Path('data/annotations/claim_annotations_2000_fixed.json')
    output_file = Path('data/annotations/claim_annotations_2000_clean.json')
    
    remove_false_positives(input_file, output_file)
    
    # Verify stats
    print("\nVerifying cleaned dataset...")
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    total_urgency = 0
    total_action = 0
    total_reward = 0
    total_claims = 0
    
    for entry in data:
        if entry.get('annotations') and entry['annotations']:
            annotations = entry['annotations'][0]
            if 'result' in annotations and annotations['result']:
                for result in annotations['result']:
                    labels = result.get('value', {}).get('labels', [])
                    if labels:
                        total_claims += 1
                        if labels[0] == 'URGENCY_CLAIM':
                            total_urgency += 1
                        elif labels[0] == 'ACTION_CLAIM':
                            total_action += 1
                        elif labels[0] == 'REWARD_CLAIM':
                            total_reward += 1
    
    print(f"\nFinal dataset stats:")
    print(f"  Total claims: {total_claims}")
    print(f"  URGENCY_CLAIM: {total_urgency}")
    print(f"  ACTION_CLAIM: {total_action}")
    print(f"  REWARD_CLAIM: {total_reward}")
    print("\nDataset is now clean and ready!")

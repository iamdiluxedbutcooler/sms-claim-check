#!/usr/bin/env python3
"""Quick check to verify GPT didn't alter original messages"""

import json
from pathlib import Path

def main():
    base_path = Path(__file__).parent
    
    # Load datasets
    with open(base_path / 'data/annotations/claim_annotations_2000.json', 'r') as f:
        claim_data = json.load(f)
    
    with open(base_path / 'data/annotations/entity_annotations_2000.json', 'r') as f:
        entity_data = json.load(f)
    
    with open(base_path / 'data/annotations/balanced_dataset_2000.json', 'r') as f:
        original_data = json.load(f)
    
    # Build original map
    original_map = {e['id']: e['data']['text'].strip() for e in original_data}
    
    print("="*60)
    print("GPT DATA INTEGRITY CHECK")
    print("="*60)
    
    # Check non-augmented entries in claim dataset
    claim_altered = []
    claim_total = 0
    for entry in claim_data:
        meta = entry.get('meta', {})
        if not meta.get('is_augmented', False):
            claim_total += 1
            entry_id = entry['id']
            current_text = entry['data']['text'].strip()
            if entry_id in original_map:
                if original_map[entry_id] != current_text:
                    claim_altered.append({
                        'id': entry_id,
                        'original': original_map[entry_id][:80],
                        'current': current_text[:80]
                    })
    
    # Check non-augmented entries in entity dataset
    entity_altered = []
    entity_total = 0
    for entry in entity_data:
        meta = entry.get('meta', {})
        if not meta.get('is_augmented', False):
            entity_total += 1
            entry_id = entry['id']
            current_text = entry['data']['text'].strip()
            if entry_id in original_map:
                if original_map[entry_id] != current_text:
                    entity_altered.append({
                        'id': entry_id,
                        'original': original_map[entry_id][:80],
                        'current': current_text[:80]
                    })
    
    print(f"\nClaim Annotations:")
    print(f"  Total non-augmented entries: {claim_total}")
    if claim_altered:
        print(f"  ❌ {len(claim_altered)} were altered by GPT:")
        for item in claim_altered[:3]:
            print(f"     ID: {item['id']}")
            print(f"       Original: '{item['original']}'...")
            print(f"       Current:  '{item['current']}'...")
    else:
        print(f"  ✅ All {claim_total} match original exactly - GPT did NOT alter data!")
    
    print(f"\nEntity Annotations:")
    print(f"  Total non-augmented entries: {entity_total}")
    if entity_altered:
        print(f"  ❌ {len(entity_altered)} were altered by GPT:")
        for item in entity_altered[:3]:
            print(f"     ID: {item['id']}")
            print(f"       Original: '{item['original']}'...")
            print(f"       Current:  '{item['current']}'...")
    else:
        print(f"  ✅ All {entity_total} match original exactly - GPT did NOT alter data!")
    
    print("\n" + "="*60)
    if not claim_altered and not entity_altered:
        print("🎉 PERFECT! GPT preserved all original message texts!")
    else:
        print("⚠️  GPT may have modified some original messages")
    print("="*60)

if __name__ == '__main__':
    main()

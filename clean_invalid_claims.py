#!/usr/bin/env python3
"""
Clean up claim_annotations_2000.json by removing annotations with invalid claim types.
Valid claim types are the 12 defined in the schema.
"""

import json
from pathlib import Path
from datetime import datetime

# Define valid claim types
VALID_CLAIM_TYPES = {
    'IDENTITY_CLAIM',
    'DELIVERY_CLAIM',
    'FINANCIAL_CLAIM',
    'ACCOUNT_CLAIM',
    'URGENCY_CLAIM',
    'ACTION_CLAIM',
    'VERIFICATION_CLAIM',
    'SECURITY_CLAIM',
    'REWARD_CLAIM',
    'LEGAL_CLAIM',
    'SOCIAL_CLAIM',
    'CREDENTIALS_CLAIM'
}

def clean_annotations(json_path):
    """Remove annotations with invalid claim types"""
    
    # Load data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create backup
    backup_path = json_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Backup created: {backup_path.name}")
    
    # Track statistics
    total_annotations = 0
    removed_annotations = 0
    invalid_claims_found = {}
    
    # Clean data
    for entry in data:
        if 'annotations' not in entry or len(entry['annotations']) == 0:
            continue
        
        annotations = entry['annotations'][0]
        if 'result' not in annotations:
            continue
        
        # Filter results to keep only valid claim types
        original_results = annotations['result']
        filtered_results = []
        
        for result in original_results:
            total_annotations += 1
            labels = result.get('value', {}).get('labels', [])
            
            # Check if all labels are valid
            invalid_labels = [label for label in labels if label not in VALID_CLAIM_TYPES]
            
            if invalid_labels:
                removed_annotations += 1
                for invalid_label in invalid_labels:
                    invalid_claims_found[invalid_label] = invalid_claims_found.get(invalid_label, 0) + 1
                print(f"  Removing: {invalid_labels} from entry {entry.get('id')}")
            else:
                filtered_results.append(result)
        
        # Update the annotations with filtered results
        annotations['result'] = filtered_results
    
    # Save cleaned data
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Total annotations processed: {total_annotations}")
    print(f"Annotations removed: {removed_annotations}")
    print(f"Annotations kept: {total_annotations - removed_annotations}")
    
    if invalid_claims_found:
        print("\nInvalid claim types removed:")
        for claim_type, count in sorted(invalid_claims_found.items()):
            print(f"  - {claim_type}: {count} occurrences")
    else:
        print("\n✨ No invalid claims found! Dataset is clean.")
    
    print("\n" + "=" * 60)
    print("Valid claim types (12 total):")
    for i, claim_type in enumerate(sorted(VALID_CLAIM_TYPES), 1):
        print(f"  {i:2d}. {claim_type}")
    print("=" * 60)
    
    return {
        'total': total_annotations,
        'removed': removed_annotations,
        'kept': total_annotations - removed_annotations,
        'invalid_types': invalid_claims_found
    }

def main():
    json_path = Path(__file__).parent / 'data' / 'annotations' / 'claim_annotations_2000.json'
    
    if not json_path.exists():
        print(f"❌ Error: Could not find {json_path}")
        return
    
    print("🧹 Cleaning invalid claim types from dataset...")
    print(f"📁 File: {json_path.name}")
    print("-" * 60)
    
    stats = clean_annotations(json_path)
    
    print(f"\n✅ Cleanup complete! File saved: {json_path.name}")

if __name__ == '__main__':
    main()

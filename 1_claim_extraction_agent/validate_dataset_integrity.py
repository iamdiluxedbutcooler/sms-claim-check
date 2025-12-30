#!/usr/bin/env python3
"""
Comprehensive validation script to check:
1. Duplicates in claim_annotations_2000.json
2. Duplicates in entity_annotations_2000.json
3. Data integrity - verify GPT didn't alter original message texts
4. Cross-check both datasets for consistency
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_duplicates(data, dataset_name):
    """Check for duplicate entries in dataset"""
    print(f"\n{'='*60}")
    print(f"CHECKING DUPLICATES: {dataset_name}")
    print('='*60)
    
    # Check duplicate IDs
    ids = [entry.get('id') for entry in data]
    id_counts = Counter(ids)
    duplicate_ids = {id_: count for id_, count in id_counts.items() if count > 1}
    
    if duplicate_ids:
        print(f"[ERROR] Found {len(duplicate_ids)} duplicate IDs:")
        for id_, count in duplicate_ids.items():
            print(f"   - ID '{id_}' appears {count} times")
    else:
        print(f"[OK] No duplicate IDs found ({len(ids)} unique IDs)")
    
    # Check duplicate message texts
    texts = [entry.get('data', {}).get('text', '').strip() for entry in data]
    text_counts = Counter(texts)
    duplicate_texts = {text: count for text, count in text_counts.items() if count > 1 and text}
    
    if duplicate_texts:
        print(f"\n[ERROR] Found {len(duplicate_texts)} duplicate message texts:")
        for i, (text, count) in enumerate(list(duplicate_texts.items())[:5]):
            preview = text[:60] + '...' if len(text) > 60 else text
            print(f"   {i+1}. '{preview}' ({count} times)")
        if len(duplicate_texts) > 5:
            print(f"   ... and {len(duplicate_texts) - 5} more")
    else:
        print(f"[OK] No duplicate message texts found ({len([t for t in texts if t])} unique messages)")
    
    return {
        'duplicate_ids': duplicate_ids,
        'duplicate_texts': duplicate_texts,
        'total_entries': len(data),
        'unique_ids': len(set(ids)),
        'unique_texts': len(set(texts))
    }

def compare_with_original(claim_data, entity_data, original_data):
    """Compare claim and entity datasets with original balanced dataset"""
    print(f"\n{'='*60}")
    print("VERIFYING DATA INTEGRITY AGAINST ORIGINAL")
    print('='*60)
    
    # Build mapping of original messages by ID
    original_map = {}
    for entry in original_data:
        entry_id = entry.get('id')
        text = entry.get('data', {}).get('text', '').strip()
        original_map[entry_id] = text
    
    print(f"Original dataset: {len(original_map)} entries")
    
    # Check claim annotations
    claim_issues = []
    claim_augmented = 0
    claim_matched = 0
    
    for entry in claim_data:
        entry_id = entry.get('id')
        current_text = entry.get('data', {}).get('text', '').strip()
        meta = entry.get('meta', {})
        
        if meta.get('is_augmented', False):
            claim_augmented += 1
            # Check if original_text matches the original dataset
            if 'original_text' in meta:
                original_text = meta['original_text'].strip()
                if entry_id in original_map:
                    if original_map[entry_id] != original_text:
                        claim_issues.append({
                            'id': entry_id,
                            'issue': 'Original text mismatch in metadata',
                            'expected': original_map[entry_id][:60],
                            'got': original_text[:60]
                        })
        else:
            # Non-augmented should match exactly
            if entry_id in original_map:
                if original_map[entry_id] != current_text:
                    claim_issues.append({
                        'id': entry_id,
                        'issue': 'Message text altered',
                        'expected': original_map[entry_id][:60],
                        'got': current_text[:60]
                    })
                else:
                    claim_matched += 1
    
    # Check entity annotations
    entity_issues = []
    entity_augmented = 0
    entity_matched = 0
    
    for entry in entity_data:
        entry_id = entry.get('id')
        current_text = entry.get('data', {}).get('text', '').strip()
        meta = entry.get('meta', {})
        
        if meta.get('is_augmented', False):
            entity_augmented += 1
            if 'original_text' in meta:
                original_text = meta['original_text'].strip()
                if entry_id in original_map:
                    if original_map[entry_id] != original_text:
                        entity_issues.append({
                            'id': entry_id,
                            'issue': 'Original text mismatch in metadata',
                            'expected': original_map[entry_id][:60],
                            'got': original_text[:60]
                        })
        else:
            if entry_id in original_map:
                if original_map[entry_id] != current_text:
                    entity_issues.append({
                        'id': entry_id,
                        'issue': 'Message text altered',
                        'expected': original_map[entry_id][:60],
                        'got': current_text[:60]
                    })
                else:
                    entity_matched += 1
    
    # Print results
    print(f"\nClaim Annotations:")
    print(f"  - Augmented entries: {claim_augmented}")
    print(f"  - Original entries matched: {claim_matched}")
    if claim_issues:
        print(f"  [ERROR] Found {len(claim_issues)} integrity issues:")
        for issue in claim_issues[:3]:
            print(f"     - {issue['id']}: {issue['issue']}")
            print(f"       Expected: '{issue['expected']}'...")
            print(f"       Got: '{issue['got']}'...")
        if len(claim_issues) > 3:
            print(f"     ... and {len(claim_issues) - 3} more")
    else:
        print(f"  [OK] No integrity issues found")
    
    print(f"\nEntity Annotations:")
    print(f"  - Augmented entries: {entity_augmented}")
    print(f"  - Original entries matched: {entity_matched}")
    if entity_issues:
        print(f"  [ERROR] Found {len(entity_issues)} integrity issues:")
        for issue in entity_issues[:3]:
            print(f"     - {issue['id']}: {issue['issue']}")
            print(f"       Expected: '{issue['expected']}'...")
            print(f"       Got: '{issue['got']}'...")
        if len(entity_issues) > 3:
            print(f"     ... and {len(entity_issues) - 3} more")
    else:
        print(f"  [OK] No integrity issues found")
    
    return {
        'claim_issues': claim_issues,
        'entity_issues': entity_issues,
        'claim_augmented': claim_augmented,
        'entity_augmented': entity_augmented
    }

def cross_check_datasets(claim_data, entity_data):
    """Cross-check claim and entity datasets for consistency"""
    print(f"\n{'='*60}")
    print("CROSS-CHECKING CLAIM vs ENTITY DATASETS")
    print('='*60)
    
    # Build maps
    claim_map = {entry.get('id'): entry.get('data', {}).get('text', '').strip() for entry in claim_data}
    entity_map = {entry.get('id'): entry.get('data', {}).get('text', '').strip() for entry in entity_data}
    
    print(f"Claim dataset: {len(claim_map)} entries")
    print(f"Entity dataset: {len(entity_map)} entries")
    
    # Find IDs in both
    common_ids = set(claim_map.keys()) & set(entity_map.keys())
    claim_only = set(claim_map.keys()) - set(entity_map.keys())
    entity_only = set(entity_map.keys()) - set(claim_map.keys())
    
    print(f"\n[OK] Common IDs: {len(common_ids)}")
    print(f"[WARN]  Claim-only IDs: {len(claim_only)}")
    print(f"[WARN]  Entity-only IDs: {len(entity_only)}")
    
    # Check if texts match for common IDs
    mismatches = []
    for entry_id in common_ids:
        if claim_map[entry_id] != entity_map[entry_id]:
            mismatches.append({
                'id': entry_id,
                'claim_text': claim_map[entry_id][:60],
                'entity_text': entity_map[entry_id][:60]
            })
    
    if mismatches:
        print(f"\n[ERROR] Found {len(mismatches)} text mismatches between datasets:")
        for mismatch in mismatches[:3]:
            print(f"   ID '{mismatch['id']}':")
            print(f"     Claim:  '{mismatch['claim_text']}'...")
            print(f"     Entity: '{mismatch['entity_text']}'...")
        if len(mismatches) > 3:
            print(f"   ... and {len(mismatches) - 3} more")
    else:
        print(f"\n[OK] All common IDs have matching texts")
    
    return {
        'common_ids': len(common_ids),
        'claim_only': len(claim_only),
        'entity_only': len(entity_only),
        'mismatches': mismatches
    }

def check_annotation_validity(data, dataset_name):
    """Check if all annotations have valid structure"""
    print(f"\n{'='*60}")
    print(f"VALIDATING ANNOTATION STRUCTURE: {dataset_name}")
    print('='*60)
    
    issues = []
    total_annotations = 0
    
    for entry in data:
        entry_id = entry.get('id')
        
        if 'annotations' not in entry:
            issues.append({'id': entry_id, 'issue': 'Missing annotations field'})
            continue
        
        if not entry['annotations']:
            # Empty annotations is OK for ham messages
            continue
        
        annotations = entry['annotations'][0]
        
        if 'result' not in annotations:
            issues.append({'id': entry_id, 'issue': 'Missing result field'})
            continue
        
        for idx, result in enumerate(annotations['result']):
            total_annotations += 1
            
            if 'value' not in result:
                issues.append({'id': entry_id, 'issue': f'Result {idx} missing value field'})
                continue
            
            value = result['value']
            
            # Check required fields
            if 'start' not in value or 'end' not in value:
                issues.append({'id': entry_id, 'issue': f'Result {idx} missing start/end'})
            
            if 'text' not in value:
                issues.append({'id': entry_id, 'issue': f'Result {idx} missing text'})
            
            if 'labels' not in value:
                issues.append({'id': entry_id, 'issue': f'Result {idx} missing labels'})
    
    print(f"Total annotations: {total_annotations}")
    
    if issues:
        print(f"[ERROR] Found {len(issues)} structural issues:")
        for issue in issues[:5]:
            print(f"   - {issue['id']}: {issue['issue']}")
        if len(issues) > 5:
            print(f"   ... and {len(issues) - 5} more")
    else:
        print(f"[OK] All annotations have valid structure")
    
    return issues

def main():
    base_path = Path(__file__).parent
    
    print(" DATASET INTEGRITY VALIDATION")
    print("="*60)
    
    # Load datasets
    print("\nLoading datasets...")
    claim_path = base_path / 'data' / 'annotations' / 'claim_annotations_2000.json'
    entity_path = base_path / 'data' / 'annotations' / 'entity_annotations_2000.json'
    original_path = base_path / 'data' / 'annotations' / 'balanced_dataset_2000.json'
    
    if not claim_path.exists():
        print(f"[ERROR] Error: {claim_path} not found")
        return
    
    if not entity_path.exists():
        print(f"[ERROR] Error: {entity_path} not found")
        return
    
    if not original_path.exists():
        print(f"[ERROR] Error: {original_path} not found")
        return
    
    claim_data = load_json(claim_path)
    entity_data = load_json(entity_path)
    original_data = load_json(original_path)
    
    print(f"[OK] Loaded claim_annotations_2000.json ({len(claim_data)} entries)")
    print(f"[OK] Loaded entity_annotations_2000.json ({len(entity_data)} entries)")
    print(f"[OK] Loaded balanced_dataset_2000.json ({len(original_data)} entries)")
    
    # Run checks
    results = {}
    
    # 1. Check duplicates
    results['claim_duplicates'] = check_duplicates(claim_data, "claim_annotations_2000.json")
    results['entity_duplicates'] = check_duplicates(entity_data, "entity_annotations_2000.json")
    
    # 2. Validate annotation structure
    results['claim_structure'] = check_annotation_validity(claim_data, "claim_annotations_2000.json")
    results['entity_structure'] = check_annotation_validity(entity_data, "entity_annotations_2000.json")
    
    # 3. Compare with original
    results['integrity'] = compare_with_original(claim_data, entity_data, original_data)
    
    # 4. Cross-check datasets
    results['cross_check'] = cross_check_datasets(claim_data, entity_data)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print('='*60)
    
    total_issues = 0
    
    if results['claim_duplicates']['duplicate_ids'] or results['claim_duplicates']['duplicate_texts']:
        print("[ERROR] Claim dataset has duplicates")
        total_issues += 1
    else:
        print("[OK] Claim dataset: No duplicates")
    
    if results['entity_duplicates']['duplicate_ids'] or results['entity_duplicates']['duplicate_texts']:
        print("[ERROR] Entity dataset has duplicates")
        total_issues += 1
    else:
        print("[OK] Entity dataset: No duplicates")
    
    if results['claim_structure']:
        print(f"[ERROR] Claim dataset: {len(results['claim_structure'])} structural issues")
        total_issues += 1
    else:
        print("[OK] Claim dataset: Valid structure")
    
    if results['entity_structure']:
        print(f"[ERROR] Entity dataset: {len(results['entity_structure'])} structural issues")
        total_issues += 1
    else:
        print("[OK] Entity dataset: Valid structure")
    
    if results['integrity']['claim_issues'] or results['integrity']['entity_issues']:
        print(f"[ERROR] Data integrity issues found")
        total_issues += 1
    else:
        print("[OK] Data integrity: All original texts preserved")
    
    if results['cross_check']['mismatches']:
        print(f"[ERROR] Cross-check: {len(results['cross_check']['mismatches'])} mismatches")
        total_issues += 1
    else:
        print("[OK] Cross-check: Datasets are consistent")
    
    print(f"\n{'='*60}")
    if total_issues == 0:
        print(" ALL CHECKS PASSED! Datasets are clean and valid.")
    else:
        print(f"[WARN]  Found {total_issues} categories of issues to review.")
    print('='*60)

if __name__ == '__main__':
    main()

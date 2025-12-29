#!/usr/bin/env python3
"""
Deep duplicate check - look for exact and near-duplicates
"""

import json
from pathlib import Path
from difflib import SequenceMatcher

def find_message(json_file, search_text):
    """Find all occurrences of a specific message"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    search_lower = search_text.lower().strip()
    matches = []
    
    for idx, entry in enumerate(data):
        text = entry['data']['text'].strip()
        
        # Exact match
        if text.lower() == search_lower:
            matches.append({
                'index': idx,
                'id': entry.get('id'),
                'text': text,
                'match_type': 'exact'
            })
        # Very similar (95%+)
        elif SequenceMatcher(None, text.lower(), search_lower).ratio() > 0.95:
            matches.append({
                'index': idx,
                'id': entry.get('id'),
                'text': text,
                'match_type': 'similar'
            })
    
    return matches

def check_all_near_duplicates(json_file, threshold=0.98):
    """Find all near-duplicate pairs"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Checking for near-duplicates (this may take a moment)...")
    
    texts = [(i, entry['data']['text'].strip().lower()) for i, entry in enumerate(data)]
    duplicates = []
    
    checked = set()
    
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if (i, j) in checked:
                continue
            
            idx1, text1 = texts[i]
            idx2, text2 = texts[j]
            
            similarity = SequenceMatcher(None, text1, text2).ratio()
            
            if similarity >= threshold:
                duplicates.append({
                    'idx1': idx1,
                    'idx2': idx2,
                    'text1': data[idx1]['data']['text'],
                    'text2': data[idx2]['data']['text'],
                    'similarity': similarity,
                    'id1': data[idx1].get('id'),
                    'id2': data[idx2].get('id')
                })
                checked.add((i, j))
    
    return duplicates

if __name__ == '__main__':
    data_file = Path(__file__).parent / 'data' / 'annotations' / 'claim_annotations_2000.json'
    
    # Search for the specific message you found
    search_text = "Last Chance! Claim ur £150 worth of discount vouchers today! Text SHOP to 85023 now!"
    
    print("="*70)
    print("SEARCHING FOR SPECIFIC MESSAGE")
    print("="*70)
    print(f"Searching for: {search_text}")
    print()
    
    matches = find_message(data_file, search_text)
    
    if matches:
        print(f"FOUND {len(matches)} occurrence(s):")
        for match in matches:
            print(f"\n  Match type: {match['match_type']}")
            print(f"  Index: {match['index']}")
            print(f"  ID: {match['id']}")
            print(f"  Text: {match['text']}")
    else:
        print("Message not found!")
    
    # Check for near-duplicates across entire dataset
    print(f"\n{'='*70}")
    print("CHECKING FOR NEAR-DUPLICATES (98%+ similar)")
    print("="*70)
    
    near_dupes = check_all_near_duplicates(data_file, threshold=0.98)
    
    if near_dupes:
        print(f"\nFOUND {len(near_dupes)} near-duplicate pairs:")
        for i, dupe in enumerate(near_dupes[:10], 1):
            print(f"\n{i}. Similarity: {dupe['similarity']:.2%}")
            print(f"   [{dupe['idx1']}] ID: {dupe['id1']}")
            print(f"   Text: {dupe['text1'][:80]}...")
            print(f"   [{dupe['idx2']}] ID: {dupe['id2']}")
            print(f"   Text: {dupe['text2'][:80]}...")
        
        if len(near_dupes) > 10:
            print(f"\n... and {len(near_dupes) - 10} more near-duplicates")
    else:
        print("\nNo near-duplicates found!")

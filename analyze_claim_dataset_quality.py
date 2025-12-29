#!/usr/bin/env python3
"""
Analyze claim annotation dataset quality for NER training
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

def analyze_annotation_quality(json_file):
    """Analyze the quality and consistency of claim annotations"""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("="*70)
    print("CLAIM ANNOTATION DATASET QUALITY ANALYSIS")
    print("="*70)
    
    # Basic stats
    total_messages = len(data)
    annotated_messages = sum(1 for entry in data if entry.get('annotations'))
    
    print(f"\nBasic Statistics:")
    print(f"  Total messages: {total_messages}")
    print(f"  Annotated messages: {annotated_messages}")
    print(f"  Unannotated: {total_messages - annotated_messages}")
    
    # Claim type distribution
    claim_type_counts = Counter()
    claim_lengths = defaultdict(list)
    messages_by_claim_count = defaultdict(int)
    span_overlaps = 0
    inconsistent_boundaries = 0
    
    # Detailed analysis
    ham_messages = 0
    smish_messages = 0
    
    for entry in data:
        if not entry.get('annotations') or len(entry['annotations']) == 0:
            continue
        
        text = entry['data']['text']
        annotations = entry['annotations'][0]
        
        # Extract claims
        claims = []
        if 'result' in annotations and annotations['result']:
            for result in annotations['result']:
                value = result.get('value', {})
                labels_list = value.get('labels', [])
                
                if labels_list:
                    claim_text = value.get('text', '')
                    claim_type = labels_list[0]
                    start = value.get('start', 0)
                    end = value.get('end', 0)
                    
                    claims.append({
                        'type': claim_type,
                        'text': claim_text,
                        'start': start,
                        'end': end
                    })
                    
                    claim_type_counts[claim_type] += 1
                    claim_lengths[claim_type].append(len(claim_text))
        
        # Categorize
        if len(claims) == 0:
            ham_messages += 1
        else:
            smish_messages += 1
            messages_by_claim_count[len(claims)] += 1
        
        # Check for overlaps
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                if not (claim1['end'] <= claim2['start'] or claim2['end'] <= claim1['start']):
                    span_overlaps += 1
        
        # Check boundary consistency
        for claim in claims:
            # Check if annotated text matches actual text
            actual_text = text[claim['start']:claim['end']]
            if actual_text != claim['text']:
                inconsistent_boundaries += 1
    
    # Print results
    print(f"\nMessage Distribution:")
    print(f"  HAM (no claims): {ham_messages} ({ham_messages/total_messages*100:.1f}%)")
    print(f"  SMISH (with claims): {smish_messages} ({smish_messages/total_messages*100:.1f}%)")
    
    print(f"\nClaims per SMISH message:")
    for count in sorted(messages_by_claim_count.keys()):
        print(f"  {count} claims: {messages_by_claim_count[count]} messages")
    
    print(f"\nClaim Type Distribution:")
    total_claims = sum(claim_type_counts.values())
    for claim_type, count in sorted(claim_type_counts.items(), key=lambda x: -x[1]):
        pct = count / total_claims * 100
        avg_len = sum(claim_lengths[claim_type]) / len(claim_lengths[claim_type])
        print(f"  {claim_type:25} : {count:4} ({pct:5.1f}%) | Avg length: {avg_len:.1f} chars")
    
    print(f"\nQuality Issues:")
    print(f"  Overlapping spans: {span_overlaps}")
    print(f"  Boundary mismatches: {inconsistent_boundaries}")
    
    # Identify rare/problematic claim types
    print(f"\nRARE CLAIM TYPES (< 50 instances):")
    rare_claims = [(ct, cnt) for ct, cnt in claim_type_counts.items() if cnt < 50]
    if rare_claims:
        for claim_type, count in sorted(rare_claims, key=lambda x: x[1]):
            print(f"  {claim_type:25} : {count:4} instances ⚠️")
    else:
        print("  None - all claim types have sufficient data")
    
    # Check for very short/long claims
    print(f"\nClaim Length Analysis:")
    for claim_type in claim_type_counts:
        lengths = claim_lengths[claim_type]
        min_len = min(lengths)
        max_len = max(lengths)
        avg_len = sum(lengths) / len(lengths)
        
        if min_len < 3 or max_len > 100:
            print(f"  {claim_type:25} : min={min_len:3} max={max_len:3} avg={avg_len:5.1f} ⚠️")
    
    # Overall assessment
    print(f"\n" + "="*70)
    print("OVERALL ASSESSMENT:")
    print("="*70)
    
    issues = []
    
    if ham_messages < total_messages * 0.3:
        issues.append(f"⚠️  Low HAM ratio ({ham_messages/total_messages*100:.1f}%) - may bias model")
    
    if span_overlaps > 0:
        issues.append(f"⚠️  {span_overlaps} overlapping spans - will confuse NER model")
    
    if inconsistent_boundaries > 0:
        issues.append(f"⚠️  {inconsistent_boundaries} boundary mismatches - annotation errors")
    
    rare_count = len(rare_claims)
    if rare_count > 0:
        issues.append(f"⚠️  {rare_count} claim types with <50 instances - insufficient training data")
    
    if not issues:
        print("✅ Dataset quality is GOOD - ready for training!")
    else:
        print("Dataset has the following issues:")
        for issue in issues:
            print(f"  {issue}")
        
        print(f"\nRecommendations:")
        if rare_count > 0:
            print("  1. Add more examples for rare claim types via data augmentation")
        if span_overlaps > 0:
            print("  2. Fix overlapping spans in annotations")
        if inconsistent_boundaries > 0:
            print("  3. Re-annotate messages with boundary errors")
        if ham_messages < total_messages * 0.3:
            print("  4. Add more HAM messages for balance")
    
    print("="*70)
    
    return {
        'total_messages': total_messages,
        'ham_messages': ham_messages,
        'smish_messages': smish_messages,
        'claim_type_counts': dict(claim_type_counts),
        'rare_claims': rare_claims,
        'span_overlaps': span_overlaps,
        'boundary_mismatches': inconsistent_boundaries
    }

if __name__ == '__main__':
    data_file = Path(__file__).parent / 'data' / 'annotations' / 'claim_annotations_2000.json'
    
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        exit(1)
    
    results = analyze_annotation_quality(data_file)

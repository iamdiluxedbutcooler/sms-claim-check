#!/usr/bin/env python3
"""
Automatically fix missing URGENCY_CLAIM and ACTION_CLAIM annotations
"""

import json
import re
from pathlib import Path

def add_missing_annotation(entry, claim_type, text_to_add, start, end):
    """Add a missing claim annotation to an entry"""
    
    # Ensure annotations structure exists
    if 'annotations' not in entry or not entry['annotations']:
        entry['annotations'] = [{
            'result': [],
            'was_cancelled': False,
            'ground_truth': False,
            'created_at': '2024-12-07T12:00:00.000000Z',
            'updated_at': '2024-12-07T12:00:00.000000Z',
            'lead_time': 0.0,
            'prediction': {},
            'result_count': 0,
            'completed_by': 1
        }]
    
    annotations = entry['annotations'][0]
    
    if 'result' not in annotations:
        annotations['result'] = []
    
    # Add the new claim
    new_claim = {
        'value': {
            'start': start,
            'end': end,
            'text': text_to_add,
            'labels': [claim_type]
        },
        'from_name': 'label',
        'to_name': 'text',
        'type': 'labels'
    }
    
    annotations['result'].append(new_claim)
    annotations['result_count'] = len(annotations['result'])
    
    return entry

def fix_annotations(input_file, output_file):
    """Fix all missing annotations"""
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load issues
    with open('annotation_issues.json', 'r', encoding='utf-8') as f:
        issues = json.load(f)
    
    print("="*70)
    print("FIXING MISSING ANNOTATIONS")
    print("="*70)
    
    urgency_fixed = 0
    action_fixed = 0
    
    # Fix missing urgency claims (only non-casual SMISH)
    print(f"\nProcessing {len(issues['missing_urgency'])} urgency issues...")
    for issue in issues['missing_urgency']:
        idx = issue['idx']
        
        # Skip HAM messages
        if issue.get('is_ham'):
            continue
        
        # Skip casual contexts
        text = data[idx]['data']['text'].lower()
        casual_contexts = ['i cant', 'anything urgent', 'quick shower', 'quick question', 'maybe', 'could get', 'better quickly']
        if any(ctx in text for ctx in casual_contexts):
            continue
        
        # Find exact position of keyword
        keyword = issue['keyword']
        start = issue['start']
        end = issue['end']
        
        data[idx] = add_missing_annotation(data[idx], 'URGENCY_CLAIM', keyword, start, end)
        urgency_fixed += 1
        
        print(f"  Fixed: {data[idx].get('id')} - Added '{keyword}'")
    
    # Fix missing action claims (only SMISH)
    print(f"\nProcessing {len(issues['missing_action'])} action issues...")
    for issue in issues['missing_action']:
        idx = issue['idx']
        text = data[idx]['data']['text']
        
        # Find more complete action phrase
        keyword_pos = issue['start']
        
        # Try to extract fuller phrase
        action_patterns = [
            r'(please )?(call|contact|text|click|visit|tap|reply|claim|redeem)([^.!?]*?)(?=[.!?]|$)',
            r'(To )?(claim|redeem|get|receive|verify)([^.!?]*?)(?=[.!?]|$)'
        ]
        
        best_match = None
        for pattern in action_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.start() <= keyword_pos <= match.end():
                    if best_match is None or len(match.group()) > len(best_match.group()):
                        best_match = match
        
        if best_match:
            action_text = best_match.group().strip()
            start = best_match.start()
            end = best_match.end()
            
            # Limit length to avoid too long spans
            if len(action_text) > 100:
                # Just use keyword + next few words
                words = action_text.split()[:8]
                action_text = ' '.join(words)
                end = start + len(action_text)
            
            data[idx] = add_missing_annotation(data[idx], 'ACTION_CLAIM', action_text, start, end)
            action_fixed += 1
            
            if action_fixed <= 5:
                print(f"  Fixed: {data[idx].get('id')} - Added '{action_text[:50]}...'")
    
    # Save fixed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Urgency claims added: {urgency_fixed}")
    print(f"Action claims added: {action_fixed}")
    print(f"Total fixed: {urgency_fixed + action_fixed}")
    print(f"\nSaved to: {output_file}")
    print(f"{'='*70}")

if __name__ == '__main__':
    input_file = Path('data/annotations/claim_annotations_2000_balanced.json')
    output_file = Path('data/annotations/claim_annotations_2000_fixed.json')
    
    fix_annotations(input_file, output_file)
    print("\nDataset fixed and ready for training!")

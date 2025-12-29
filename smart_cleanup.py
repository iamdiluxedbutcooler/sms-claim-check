#!/usr/bin/env python3
"""
Smart removal of ONLY truly false positive annotations
"""

import json
from pathlib import Path

def is_truly_false_positive(label, text, full_message):
    """
    Determine if annotation is truly a false positive
    """
    text_lower = text.lower().strip()
    
    # URGENCY_CLAIM false positives
    if label == 'URGENCY_CLAIM':
        # Time ranges (e.g., "between 9am-11pm", "10am-9pm")
        if 'between' in text_lower and ('am' in text_lower or 'pm' in text_lower):
            return True
        if text_lower.count('am') + text_lower.count('pm') >= 2 and '-' in text:
            return True
        
        # Bare URLs labeled as urgency
        if text_lower.startswith(('http://', 'https://', 'www.', 'bit.ly', 'bit.do')):
            if not any(word in text_lower for word in ['urgent', 'immediate', 'now', 'quick', 'hurry', 'act']):
                return True
        
        # Phone numbers with "between" time context
        if 'between' in full_message.lower()[max(0, full_message.lower().find(text_lower)-50):]:
            return True
    
    # ACTION_CLAIM false positives
    elif label == 'ACTION_CLAIM':
        # Bare URLs WITHOUT action verbs
        if text_lower.startswith(('http://', 'https://', 'www.', 'bit.ly', 'bit.do')):
            # Check if there's an action verb in the text
            action_verbs = ['visit', 'click', 'go to', 'check', 'see', 'view', 'access', 'tap', 'press']
            has_action_verb = any(verb in text_lower for verb in action_verbs)
            
            if not has_action_verb:
                return True  # Just a URL, no action
        
        # Very short generic phrases
        if len(text) < 5 and text_lower in ['to', 'the', 'and', 'or', 'at', 'from', 'with']:
            return True
        
        # Common false patterns
        if text_lower.startswith(('the ', 'and ', 'or ', 'if ', 'when ', 'that ', 'this ')):
            if len(text) < 15:
                return True
    
    # REWARD_CLAIM false positives
    elif label == 'REWARD_CLAIM':
        # Generic short phrases
        if len(text) < 8 and text_lower in ['the', 'and', 'or', 'if', 'when', 'that', 'this']:
            return True
        
        # URLs alone
        if text_lower.startswith(('http://', 'https://', 'www.')):
            return True
    
    return False

def smart_cleanup(input_file, output_file):
    """Smart cleanup keeping valid ACTION_CLAIMs with URLs"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("="*70)
    print("SMART CLEANUP - Removing ONLY True False Positives")
    print("="*70)
    
    removed_count = 0
    kept_count = 0
    
    for idx, entry in enumerate(data):
        entry_id = entry.get('id')
        full_message = entry['data']['text']
        
        if not entry.get('annotations') or not entry['annotations']:
            continue
        
        annotations = entry['annotations'][0]
        if 'result' not in annotations or not annotations['result']:
            continue
        
        # Filter results
        new_results = []
        for result in annotations['result']:
            value = result.get('value', {})
            text = value.get('text', '')
            labels = value.get('labels', [])
            
            if not labels:
                new_results.append(result)
                continue
            
            label = labels[0]
            
            # Check if truly false positive
            if is_truly_false_positive(label, text, full_message):
                removed_count += 1
                if removed_count <= 10:
                    print(f"  Removed: {entry_id} - {label}: '{text[:50]}...'")
            else:
                new_results.append(result)
                kept_count += 1
        
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
    print(f"Valid claims kept: {kept_count}")
    print(f"Saved to: {output_file}")
    
    # Stats
    total_urgency = 0
    total_action = 0
    total_reward = 0
    
    for entry in data:
        if entry.get('annotations') and entry['annotations']:
            annotations = entry['annotations'][0]
            if 'result' in annotations and annotations['result']:
                for result in annotations['result']:
                    labels = result.get('value', {}).get('labels', [])
                    if labels:
                        if labels[0] == 'URGENCY_CLAIM':
                            total_urgency += 1
                        elif labels[0] == 'ACTION_CLAIM':
                            total_action += 1
                        elif labels[0] == 'REWARD_CLAIM':
                            total_reward += 1
    
    print(f"\nFinal claim counts:")
    print(f"  URGENCY_CLAIM: {total_urgency}")
    print(f"  ACTION_CLAIM: {total_action}")
    print(f"  REWARD_CLAIM: {total_reward}")
    print(f"{'='*70}")

if __name__ == '__main__':
    input_file = Path('data/annotations/claim_annotations_2000_fixed.json')
    output_file = Path('data/annotations/claim_annotations_2000_clean.json')
    
    smart_cleanup(input_file, output_file)
    print("\nDataset is now properly cleaned!")

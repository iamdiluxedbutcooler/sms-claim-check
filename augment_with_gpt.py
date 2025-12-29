#!/usr/bin/env python3
"""
Generate 36 SMISH messages with claim annotations using OpenAI API
Then merge with deduplicated dataset to create final balanced dataset
"""

import json
import os
from pathlib import Path
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # Use environment variable

# Claim types to use
CLAIM_TYPES = [
    'IDENTITY_CLAIM', 'DELIVERY_CLAIM', 'FINANCIAL_CLAIM', 'ACCOUNT_CLAIM',
    'URGENCY_CLAIM', 'ACTION_CLAIM', 'VERIFICATION_CLAIM', 'SECURITY_CLAIM',
    'REWARD_CLAIM', 'LEGAL_CLAIM', 'SOCIAL_CLAIM', 'CREDENTIALS_CLAIM'
]

def generate_smish_messages(num_messages=36):
    """Generate SMISH messages with claim annotations using OpenAI API"""
    
    print("="*70)
    print(f"GENERATING {num_messages} SMISH MESSAGES WITH CLAIM ANNOTATIONS")
    print("="*70)
    
    prompt = f"""Generate {num_messages} diverse phishing SMS messages with claim annotations.

CLAIM TYPES:
{', '.join(CLAIM_TYPES)}

REQUIREMENTS:
1. Each message should be realistic phishing SMS (smishing)
2. Include 2-4 different claim types per message
3. Vary the phishing tactics: delivery scams, bank alerts, prize scams, account suspensions, etc.
4. Make them diverse - different brands, scenarios, urgency levels
5. Keep messages 50-150 characters typical of SMS

OUTPUT FORMAT (valid JSON array):
[
  {{
    "text": "Your Amazon package is delayed. Click here urgently to reschedule delivery.",
    "claims": [
      {{"text": "Amazon", "start": 5, "end": 11, "label": "IDENTITY_CLAIM"}},
      {{"text": "package is delayed", "start": 12, "end": 30, "label": "DELIVERY_CLAIM"}},
      {{"text": "Click here", "start": 32, "end": 42, "label": "ACTION_CLAIM"}},
      {{"text": "urgently", "start": 43, "end": 51, "label": "URGENCY_CLAIM"}}
    ]
  }},
  ...
]

Generate exactly {num_messages} diverse phishing messages. Return ONLY the JSON array, no other text."""

    print("\nCalling OpenAI API...")
    print(f"Requesting {num_messages} messages...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at generating realistic phishing SMS messages with precise claim annotations for training fraud detection models."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=8000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON
        generated_data = json.loads(content)
        
        print(f"\nSuccessfully generated {len(generated_data)} messages")
        
        return generated_data
        
    except Exception as e:
        print(f"\nError generating messages: {e}")
        return []

def convert_to_label_studio_format(generated_data, start_id=10000):
    """Convert generated data to Label Studio annotation format"""
    
    label_studio_data = []
    
    for idx, item in enumerate(generated_data):
        text = item['text']
        claims = item.get('claims', [])
        
        # Create Label Studio format
        results = []
        for claim in claims:
            results.append({
                "value": {
                    "start": claim['start'],
                    "end": claim['end'],
                    "text": claim['text'],
                    "labels": [claim['label']]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels"
            })
        
        entry = {
            "id": start_id + idx,
            "data": {
                "text": text
            },
            "annotations": [
                {
                    "result": results,
                    "was_cancelled": False,
                    "ground_truth": False,
                    "created_at": "2024-12-07T12:00:00.000000Z",
                    "updated_at": "2024-12-07T12:00:00.000000Z",
                    "lead_time": 0.0,
                    "prediction": {},
                    "result_count": 0,
                    "completed_by": 1
                }
            ],
            "file_upload": "mendeley-augmented.csv",
            "drafts": [],
            "predictions": [],
            "project": 1,
            "updated_by": None,
            "updated_at": "2024-12-07T12:00:00.000000Z",
            "inner_id": start_id + idx,
            "total_annotations": 1,
            "cancelled_annotations": 0,
            "total_predictions": 0,
            "comment_count": 0,
            "unresolved_comment_count": 0,
            "last_comment_updated_at": None,
            "comment_authors": []
        }
        
        label_studio_data.append(entry)
    
    return label_studio_data

def merge_datasets(deduped_file, augmented_data, output_file):
    """Merge deduplicated dataset with augmented data"""
    
    print("\n" + "="*70)
    print("MERGING DATASETS")
    print("="*70)
    
    # Load deduplicated data
    with open(deduped_file, 'r', encoding='utf-8') as f:
        deduped_data = json.load(f)
    
    print(f"\nDeduplicated dataset: {len(deduped_data)} messages")
    print(f"Augmented data: {len(augmented_data)} messages")
    
    # Remove 5 HAM messages to balance (have 1005, need 1000)
    ham_indices = []
    for idx, entry in enumerate(deduped_data):
        has_claims = False
        if entry.get('annotations') and len(entry['annotations']) > 0:
            annotations = entry['annotations'][0]
            if 'result' in annotations and annotations['result']:
                has_claims = True
        if not has_claims:
            ham_indices.append(idx)
    
    import random
    random.seed(42)
    ham_to_remove = set(random.sample(ham_indices, min(5, len(ham_indices))))
    
    # Filter out selected HAM messages
    filtered_data = [entry for idx, entry in enumerate(deduped_data) if idx not in ham_to_remove]
    
    print(f"Removed {len(ham_to_remove)} HAM messages for balance")
    
    # Merge
    final_data = filtered_data + augmented_data
    
    # Count final distribution
    ham_count = 0
    smish_count = 0
    
    for entry in final_data:
        has_claims = False
        if entry.get('annotations') and len(entry['annotations']) > 0:
            annotations = entry['annotations'][0]
            if 'result' in annotations and annotations['result']:
                has_claims = True
        
        if has_claims:
            smish_count += 1
        else:
            ham_count += 1
    
    # Save final dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("FINAL BALANCED DATASET")
    print(f"{'='*70}")
    print(f"Total messages: {len(final_data)}")
    print(f"  HAM: {ham_count}")
    print(f"  SMISH: {smish_count}")
    print(f"\nSaved to: {output_file}")
    print(f"{'='*70}")
    
    return final_data

def main():
    # File paths
    deduped_file = Path(__file__).parent / 'data' / 'annotations' / 'claim_annotations_deduped.json'
    output_file = Path(__file__).parent / 'data' / 'annotations' / 'claim_annotations_2000_balanced.json'
    
    # Generate messages
    generated_data = generate_smish_messages(num_messages=36)
    
    if not generated_data:
        print("Failed to generate messages!")
        return
    
    # Show samples
    print("\n" + "="*70)
    print("SAMPLE GENERATED MESSAGES (first 3):")
    print("="*70)
    for i, item in enumerate(generated_data[:3], 1):
        print(f"\n{i}. {item['text']}")
        print(f"   Claims: {len(item['claims'])}")
        for claim in item['claims']:
            print(f"     - {claim['label']:20} : '{claim['text']}'")
    
    # Convert to Label Studio format
    print("\nConverting to Label Studio format...")
    augmented_label_studio = convert_to_label_studio_format(generated_data, start_id=10000)
    
    # Merge with deduplicated dataset
    final_data = merge_datasets(deduped_file, augmented_label_studio, output_file)
    
    print("\nAugmentation complete!")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
OOD Testing: SmishTank Dataset Claim Extraction

Tests the trained NER model on real-world phishing SMS messages
from SmishTank to evaluate generalization to unseen data.
"""

import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pathlib import Path

# Claim type normalization
RARE_CLAIMS = ['SECURITY_CLAIM', 'IDENTITY_CLAIM', 'CREDENTIALS_CLAIM', 'LEGAL_CLAIM', 'SOCIAL_CLAIM']

def normalize_claim_type(claim_type):
    return 'OTHER_CLAIM' if claim_type in RARE_CLAIMS else claim_type


def extract_claims_with_ner(text, model, tokenizer, id2label, confidence_threshold=0.5):
    """Extract claims using NER model"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        return_offsets_mapping=True
    )
    
    offset_mapping = inputs.pop('offset_mapping')[0]
    
    # Move to correct device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0]
    probabilities = torch.softmax(outputs.logits, dim=2)[0]
    
    # Build claims with better merging
    claims = []
    current_claim = None
    
    for idx, (pred, prob, (start, end)) in enumerate(zip(predictions, probabilities, offset_mapping)):
        if start == 0 and end == 0:
            continue
        
        label = id2label[pred.item()]
        confidence = prob[pred].item()
        
        if label.startswith('B-'):
            if current_claim:
                claims.append(current_claim)
            
            current_claim = {
                'type': label[2:],
                'start': start.item(),
                'end': end.item(),
                'confidence': confidence,
                'token_count': 1
            }
        
        elif label.startswith('I-') and current_claim:
            if label[2:] == current_claim['type']:
                current_claim['end'] = end.item()
                current_claim['token_count'] += 1
                current_claim['confidence'] = (
                    current_claim['confidence'] * (current_claim['token_count'] - 1) + confidence
                ) / current_claim['token_count']
        
        elif label == 'O':
            if current_claim:
                claims.append(current_claim)
                current_claim = None
    
    if current_claim:
        claims.append(current_claim)
    
    # Extract text and apply quality filters
    filtered_claims = []
    for claim in claims:
        claim['text'] = text[claim['start']:claim['end']].strip()
        claim.pop('token_count')
        
        # Filters
        if claim['confidence'] < confidence_threshold:
            continue
        if len(claim['text']) < 3:
            continue
        
        stopwords = {'to', 'a', 'an', 'the', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from'}
        if claim['text'].lower() in stopwords:
            continue
        
        if not any(c.isalnum() for c in claim['text']):
            continue
        
        filtered_claims.append(claim)
    
    return filtered_claims


def main():
    print("="*80)
    print("OOD TESTING: SmishTank Dataset Claim Extraction")
    print("="*80)
    
    # Load SmishTank data
    print("\nLoading SmishTank dataset...")
    data_path = Path("data/raw/smishtank.csv")
    
    if not data_path.exists():
        print(f"ERROR: {data_path} not found!")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} messages from SmishTank")
    
    # Check for text column
    text_col = None
    for col in ['Fulltext', 'MainText', 'text', 'message', 'sms', 'content', 'body']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print(f"Available columns: {list(df.columns)}")
        text_col = input("Enter the name of the text column: ").strip()
    
    print(f"Using column: '{text_col}'")
    
    # Load model
    print("\nLoading trained NER model...")
    
    # Try different possible model locations
    possible_paths = [
        "experiments/approach5_pure_ner/final_model",
        "models/approach5_pure_ner",
        "approach5_final_model",
        "final_model"
    ]
    
    model_path = None
    for path in possible_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if model_path is None:
        print(f"\nERROR: Model not found!")
        print("\nPlease download the model from Google Drive:")
        print("1. Go to: /content/drive/MyDrive/sms_claim_models/approach5_pure_ner/final_model/")
        print("2. Download all files (config.json, pytorch_model.bin, etc.)")
        print("3. Place them in: experiments/approach5_pure_ner/final_model/")
        print("\nOr specify the model path as a command line argument:")
        print("  python ood_test_smishtank.py --model-path /path/to/model")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # Load label mappings
    with open(Path(model_path) / "config.json", 'r') as f:
        config = json.load(f)
    
    id2label = {int(k): v for k, v in config['label_mappings']['id2label'].items()}
    
    print(f"Model loaded: {config['model_name']}")
    print(f"Trained on {config['num_train_examples']} examples")
    
    # Process first 100 messages
    print("\n" + "="*80)
    print("EXTRACTING CLAIMS FROM FIRST 100 MESSAGES")
    print("="*80)
    
    results = []
    claim_type_counts = {}
    total_claims = 0
    messages_with_claims = 0
    
    for idx in range(min(100, len(df))):
        text = str(df.iloc[idx][text_col])
        
        # Skip if empty
        if pd.isna(text) or len(text.strip()) == 0:
            continue
        
        # Extract claims
        claims = extract_claims_with_ner(text, model, tokenizer, id2label)
        
        if claims:
            messages_with_claims += 1
            total_claims += len(claims)
            
            # Count by type
            for claim in claims:
                claim_type = claim['type']
                claim_type_counts[claim_type] = claim_type_counts.get(claim_type, 0) + 1
        
        result = {
            'index': idx,
            'text': text,
            'num_claims': len(claims),
            'claims': claims
        }
        results.append(result)
        
        # Print progress every 10 messages
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/100 messages...")
    
    # Save results
    output_path = Path("evaluation_output/smishtank_ood_results.json")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total messages processed: {len(results)}")
    print(f"Messages with claims: {messages_with_claims} ({messages_with_claims/len(results)*100:.1f}%)")
    print(f"Total claims extracted: {total_claims}")
    print(f"Avg claims per message: {total_claims/len(results):.2f}")
    print(f"Avg claims per message (with claims): {total_claims/messages_with_claims:.2f}" if messages_with_claims > 0 else "N/A")
    
    print(f"\n{'='*80}")
    print("CLAIM TYPE DISTRIBUTION")
    print(f"{'='*80}")
    for claim_type, count in sorted(claim_type_counts.items(), key=lambda x: -x[1]):
        print(f"{claim_type:25} {count:4} ({count/total_claims*100:5.1f}%)")
    
    # Show sample extractions
    print(f"\n{'='*80}")
    print("SAMPLE EXTRACTIONS (First 20 messages with claims)")
    print(f"{'='*80}")
    
    shown = 0
    for result in results:
        if result['num_claims'] > 0 and shown < 20:
            print(f"\n{'-'*80}")
            print(f"Message {result['index'] + 1}:")
            print(f"Text: {result['text'][:150]}{'...' if len(result['text']) > 150 else ''}")
            print(f"\nExtracted {result['num_claims']} claims:")
            for i, claim in enumerate(result['claims'], 1):
                print(f"  {i}. [{claim['type']}] \"{claim['text']}\" (conf: {claim['confidence']:.3f})")
            shown += 1
    
    print(f"\n{'='*80}")
    print(f"Full results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Data Augmentation Pipeline for SMS Phishing Dataset

This script:
1. Cleans and deduplicates Mendeley dataset
2. Keeps original messages intact
3. Generates high-quality augmented variations
4. Ensures coverage of all 12 claim types
5. Produces 1000 diverse phishing messages

"""

import json
import pandas as pd
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re
from openai import OpenAI
import os
from tqdm import tqdm
import time

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 12 Claim Types for comprehensive coverage
CLAIM_TYPES = [
    "IDENTITY_CLAIM",       # "We are Amazon/PayPal/IRS"
    "DELIVERY_CLAIM",       # "Your package is delayed"
    "FINANCIAL_CLAIM",      # "You won $5000"
    "ACCOUNT_CLAIM",        # "Your account is suspended"
    "URGENCY_CLAIM",        # "Act now within 24h"
    "ACTION_CLAIM",         # "Click here / Call now"
    "VERIFICATION_CLAIM",   # "Verify your identity"
    "SECURITY_CLAIM",       # "Suspicious activity detected"
    "REWARD_CLAIM",         # "Loyalty bonus available"
    "LEGAL_CLAIM",          # "Legal action pending"
    "SOCIAL_CLAIM",         # "Friend needs help"
    "CREDENTIALS_CLAIM"     # "Update password"
]


class DatasetCleaner:
    """Clean and deduplicate Mendeley phishing dataset"""
    
    def __init__(self, mendeley_path: str):
        self.mendeley_path = mendeley_path
        self.seen_hashes: Set[str] = set()
    
    def _compute_hash(self, text: str) -> str:
        """Compute normalized hash for duplicate detection"""
        # Normalize: lowercase, remove extra spaces, remove punctuation
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = ' '.join(normalized.split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def load_and_clean(self) -> Tuple[List[str], Dict]:
        """Load Mendeley dataset and remove duplicates"""
        print("Loading Mendeley dataset...")
        df = pd.read_csv(self.mendeley_path)
        
        # Get phishing messages (both 'Smishing' and 'smishing')
        phishing_mask = df['LABEL'].str.lower() == 'smishing'
        phishing_df = df[phishing_mask].copy()
        
        print(f"   Found {len(phishing_df)} raw phishing messages")
        
        # Remove duplicates
        unique_messages = []
        duplicates_removed = 0
        
        for text in phishing_df['TEXT']:
            text = str(text).strip()
            
            # Skip empty or very short messages
            if len(text) < 20:
                continue
            
            # Check for duplicates
            text_hash = self._compute_hash(text)
            if text_hash in self.seen_hashes:
                duplicates_removed += 1
                continue
            
            self.seen_hashes.add(text_hash)
            unique_messages.append(text)
        
        print(f"   Removed {duplicates_removed} duplicates")
        print(f"   {len(unique_messages)} unique phishing messages")
        
        stats = {
            'total_raw': len(phishing_df),
            'duplicates_removed': duplicates_removed,
            'unique_clean': len(unique_messages)
        }
        
        return unique_messages, stats


class ClaimAnalyzer:
    """Analyze and classify claims in phishing messages"""
    
    def __init__(self):
        self.claim_patterns = self._build_claim_patterns()
    
    def _build_claim_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for each claim type"""
        return {
            "IDENTITY_CLAIM": [
                r'\b(amazon|paypal|dhl|usps|fedex|irs|apple|netflix|bank)\b',
                r'(we are|this is|from)\s+\w+',
            ],
            "DELIVERY_CLAIM": [
                r'\b(package|parcel|delivery|shipment|order)\b',
                r'\b(delayed|stuck|waiting|arrived|failed)\b',
            ],
            "FINANCIAL_CLAIM": [
                r'\$\d+|\d+\s*(dollars|usd|gbp|euro)',
                r'\b(won|prize|reward|cash|refund|bonus)\b',
            ],
            "ACCOUNT_CLAIM": [
                r'\b(account|profile|membership)\s+(suspended|locked|blocked|closed)',
                r'\b(unauthorized|unusual)\s+activity',
            ],
            "URGENCY_CLAIM": [
                r'\b(urgent|immediately|now|today|tonight|asap)\b',
                r'within\s+\d+\s+(hours|days|minutes)',
                r'\b(expires|expiring|limited time)\b',
            ],
            "ACTION_CLAIM": [
                r'\b(click|tap|call|reply|respond|confirm|update)\b',
                r'\b(visit|go to|check|verify)\b',
            ],
            "VERIFICATION_CLAIM": [
                r'\b(verify|confirm|validate|authenticate)\b',
                r'\b(identity|details|information)\b',
            ],
            "SECURITY_CLAIM": [
                r'\b(security|suspicious|unauthorized|breach|compromised)\b',
                r'\b(fraud|scam) (alert|warning|detected)',
            ],
            "REWARD_CLAIM": [
                r'\b(reward|loyalty|cashback|points|bonus)\b',
                r'\b(redeem|claim|collect)\b',
            ],
            "LEGAL_CLAIM": [
                r'\b(legal|court|lawsuit|penalty|fine|tax)\b',
                r'\b(summons|warrant|arrest|prosecution)\b',
            ],
            "SOCIAL_CLAIM": [
                r'\b(mom|dad|son|daughter|friend|family)\b',
                r'\b(help|emergency|accident|hospital)\b',
            ],
            "CREDENTIALS_CLAIM": [
                r'\b(password|pin|code|credentials|login)\b',
                r'\b(reset|change|update|expired)\b',
            ],
        }
    
    def detect_claims(self, text: str) -> List[str]:
        """Detect which claim types are present in a message"""
        text_lower = text.lower()
        detected_claims = []
        
        for claim_type, patterns in self.claim_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_claims.append(claim_type)
                    break
        
        return detected_claims


class MessageAugmenter:
    """Generate high-quality augmented variations of phishing messages"""
    
    def __init__(self, augmentation_prob: float = 0.3):
        self.augmentation_prob = augmentation_prob
        self.claim_analyzer = ClaimAnalyzer()
    
    def _call_gpt_for_variation(
        self, 
        original_message: str, 
        claim_types: List[str],
        variation_type: str = "rephrase"
    ) -> str:
        """Use GPT to generate high-quality variation"""
        
        claims_str = ", ".join(claim_types) if claim_types else "various phishing tactics"
        
        if variation_type == "rephrase":
            prompt = f"""You are a cybersecurity researcher creating training data for phishing detection.

Original phishing SMS: "{original_message}"

Detected claim types: {claims_str}

Task: Create a HIGH-QUALITY variation of this phishing message that:
1. Preserves the same phishing tactics and claim types
2. Uses completely different wording and phrasing
3. Maintains realistic SMS length (50-160 characters)
4. Sounds natural and convincing (improve quality if original is poor)
5. Includes realistic brand names, URLs, phone numbers if present
6. Keeps the same level of urgency and social engineering

Output ONLY the new SMS message (no quotes, no explanations).
"""
        
        elif variation_type == "enhance":
            prompt = f"""You are a cybersecurity researcher improving phishing detection training data.

Original phishing SMS: "{original_message}"

Detected claim types: {claims_str}

Task: ENHANCE this phishing message to create a more sophisticated version:
1. Fix any grammar/spelling issues
2. Make it more convincing and realistic
3. Add subtle urgency or authority cues
4. Keep SMS length appropriate (50-160 characters)
5. Maintain the core phishing tactics
6. Make it sound more professional/legitimate

Output ONLY the enhanced SMS message (no quotes, no explanations).
"""
        
        else:  # rewrite
            prompt = f"""You are a cybersecurity researcher creating diverse phishing training data.

Original phishing SMS: "{original_message}"

Claim types to include: {claims_str}

Task: COMPLETELY REWRITE this as a NEW phishing message that:
1. Uses the same claim types but different scenario
2. Targets a different brand/service (if applicable)
3. Uses different social engineering tactics
4. Maintains SMS length (50-160 characters)
5. Is highly convincing and realistic
6. Sounds natural for 2024/2025 phishing attempts

Output ONLY the new SMS message (no quotes, no explanations).
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cybersecurity expert helping create high-quality training data for phishing detection research."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            # Remove quotes if GPT added them
            result = result.strip('"\'')
            return result
            
        except Exception as e:
            print(f"   WARNING: GPT API error: {e}")
            return None
    
    def generate_variations(
        self, 
        message: str, 
        num_variations: int = 2
    ) -> List[Dict]:
        """Generate multiple variations of a message"""
        
        # Detect claim types in original
        claim_types = self.claim_analyzer.detect_claims(message)
        
        # Decide augmentation strategy
        if random.random() > self.augmentation_prob:
            return []  # No augmentation for this message
        
        variations = []
        
        # Generate different types of variations
        variation_types = ["rephrase", "enhance", "rewrite"]
        
        for i in range(num_variations):
            var_type = random.choice(variation_types)
            
            augmented = self._call_gpt_for_variation(
                message,
                claim_types,
                variation_type=var_type
            )
            
            if augmented and len(augmented) > 20:  # Valid output
                variations.append({
                    'text': augmented,
                    'original_text': message,
                    'claim_types': claim_types,
                    'augmentation_type': var_type,
                    'is_augmented': True
                })
            
            # Rate limiting
            time.sleep(0.5)
        
        return variations


class BalancedDatasetBuilder:
    """Build balanced dataset with 1000 phishing messages covering all claim types"""
    
    def __init__(self, target_size: int = 1000):
        self.target_size = target_size
        self.claim_analyzer = ClaimAnalyzer()
    
    def build_dataset(
        self, 
        clean_messages: List[str],
        augmenter: MessageAugmenter
    ) -> List[Dict]:
        """Build final dataset with originals + augmented"""
        
        print(f"\nBuilding dataset (target: {self.target_size} messages)...")
        
        # Track claim type coverage
        claim_coverage = defaultdict(int)
        
        # Start with original clean messages
        dataset = []
        
        print(f"   Adding {len(clean_messages)} original messages...")
        for idx, msg in enumerate(clean_messages):
            claims = self.claim_analyzer.detect_claims(msg)
            
            entry = {
                'id': f'mendeley_{idx}',
                'text': msg,
                'claim_types': claims,
                'is_augmented': False,
                'source': 'mendeley_original'
            }
            dataset.append(entry)
            
            # Update coverage
            for claim in claims:
                claim_coverage[claim] += 1
        
        print(f"   Added {len(dataset)} original messages")
        
        # Calculate how many augmented messages we need
        needed = self.target_size - len(dataset)
        print(f"   Need {needed} augmented messages to reach {self.target_size}")
        
        if needed <= 0:
            print(f"   Already have {len(dataset)} messages!")
            return dataset
        
        # Generate augmented variations
        print(f"\nGenerating {needed} augmented variations...")
        
        augmented_count = 0
        pbar = tqdm(total=needed, desc="   Augmenting")
        
        while augmented_count < needed:
            # Randomly select message to augment
            source_msg = random.choice(clean_messages)
            
            # Generate variations (2-3 per message)
            num_vars = min(3, needed - augmented_count)
            variations = augmenter.generate_variations(source_msg, num_variations=num_vars)
            
            for var in variations:
                var['id'] = f'augmented_{augmented_count}'
                var['source'] = 'gpt_augmented'
                dataset.append(var)
                
                # Update coverage
                for claim in var['claim_types']:
                    claim_coverage[claim] += 1
                
                augmented_count += 1
                pbar.update(1)
                
                if augmented_count >= needed:
                    break
        
        pbar.close()
        
        # Print final statistics
        print(f"\nDataset complete!")
        print(f"   Total messages: {len(dataset)}")
        print(f"   Original: {len(dataset) - augmented_count}")
        print(f"   Augmented: {augmented_count}")
        
        print(f"\nClaim Type Coverage:")
        for claim_type in CLAIM_TYPES:
            count = claim_coverage[claim_type]
            percentage = (count / len(dataset)) * 100
            print(f"   {claim_type:25s}: {count:4d} messages ({percentage:.1f}%)")
        
        return dataset


def save_dataset(dataset: List[Dict], output_path: str):
    """Save augmented dataset in Label Studio format"""
    
    print(f"\nSaving dataset to {output_path}...")
    
    # Convert to Label Studio format
    label_studio_format = []
    
    for entry in dataset:
        annotation = {
            "id": entry['id'],
            "data": {
                "text": entry['text']
            },
            "annotations": [{
                "result": [],  # Will be annotated later
                "was_cancelled": False,
                "ground_truth": False
            }],
            "meta": {
                "claim_types": entry['claim_types'],
                "is_augmented": entry['is_augmented'],
                "source": entry['source']
            }
        }
        
        if entry['is_augmented']:
            annotation['meta']['original_text'] = entry.get('original_text', '')
            annotation['meta']['augmentation_type'] = entry.get('augmentation_type', '')
        
        label_studio_format.append(annotation)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(label_studio_format, f, indent=2, ensure_ascii=False)
    
    print(f"   Saved {len(label_studio_format)} messages")
    
    # Also save metadata summary
    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    metadata = {
        'total_messages': len(label_studio_format),
        'original_count': sum(1 for e in dataset if not e['is_augmented']),
        'augmented_count': sum(1 for e in dataset if e['is_augmented']),
        'claim_type_coverage': {
            claim: sum(1 for e in dataset if claim in e['claim_types'])
            for claim in CLAIM_TYPES
        },
        'generation_date': '2025-12-05',
        'target_size': 1000
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Saved metadata to {metadata_path}")


def main():
    """Main augmentation pipeline"""
    
    print("=" * 80)
    print("SMS PHISHING DATASET AUGMENTATION PIPELINE")
    print("=" * 80)
    print()
    
    # Configuration
    MENDELEY_PATH = 'data/raw/mendeley.csv'
    OUTPUT_PATH = 'data/annotations/augmented_phishing_1000.json'
    TARGET_SIZE = 1000
    AUGMENTATION_PROB = 0.5  # 50% chance to augment each message
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Step 1: Clean and deduplicate
    cleaner = DatasetCleaner(MENDELEY_PATH)
    clean_messages, stats = cleaner.load_and_clean()
    
    # Step 2: Initialize augmenter
    augmenter = MessageAugmenter(augmentation_prob=AUGMENTATION_PROB)
    
    # Step 3: Build balanced dataset
    builder = BalancedDatasetBuilder(target_size=TARGET_SIZE)
    final_dataset = builder.build_dataset(clean_messages, augmenter)
    
    # Step 4: Save dataset
    save_dataset(final_dataset, OUTPUT_PATH)
    
    print("\n" + "=" * 80)
    print("AUGMENTATION COMPLETE!")
    print("=" * 80)
    print(f"\nOriginal Mendeley: {stats['unique_clean']} unique messages (kept as-is)")
    print(f"Augmented: {len(final_dataset) - stats['unique_clean']} new variations")
    print(f"Total dataset: {len(final_dataset)} messages")
    print(f"\nOutput saved to: {OUTPUT_PATH}")
    print("\nNext steps:")
    print("1. Review augmented messages for quality")
    print("2. Annotate claims in Label Studio")
    print("3. Train models with new dataset")
    print()


if __name__ == "__main__":
    main()

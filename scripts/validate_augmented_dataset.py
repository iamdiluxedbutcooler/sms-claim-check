#!/usr/bin/env python3
"""
Validate Augmented Dataset Quality

Checks:
1. Duplicate detection
2. Length distribution
3. Claim type coverage
4. Quality metrics
"""

import json
import hashlib
import re
from pathlib import Path
from collections import Counter, defaultdict
import statistics

def compute_hash(text: str) -> str:
    """Compute normalized hash"""
    normalized = re.sub(r'[^\w\s]', '', text.lower())
    normalized = ' '.join(normalized.split())
    return hashlib.md5(normalized.encode()).hexdigest()

def validate_dataset(dataset_path: str):
    """Validate augmented dataset"""
    
    print("Validating Augmented Dataset")
    print("=" * 80)
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nBasic Statistics:")
    print(f"   Total messages: {len(data)}")
    
    # Count originals vs augmented
    original_count = sum(1 for d in data if not d['meta']['is_augmented'])
    augmented_count = sum(1 for d in data if d['meta']['is_augmented'])
    
    print(f"   Original: {original_count}")
    print(f"   Augmented: {augmented_count}")
    
    # Check for duplicates
    print(f"\nDuplicate Check:")
    hashes = set()
    duplicates = []
    
    for entry in data:
        text = entry['data']['text']
        text_hash = compute_hash(text)
        
        if text_hash in hashes:
            duplicates.append(entry['id'])
        else:
            hashes.add(text_hash)
    
    if duplicates:
        print(f"   WARNING: Found {len(duplicates)} duplicates!")
        print(f"   IDs: {duplicates[:5]}...")
    else:
        print(f"   No duplicates found!")
    
    # Length analysis
    print(f"\nLength Analysis:")
    lengths = [len(d['data']['text']) for d in data]
    
    print(f"   Min: {min(lengths)} chars")
    print(f"   Max: {max(lengths)} chars")
    print(f"   Mean: {statistics.mean(lengths):.1f} chars")
    print(f"   Median: {statistics.median(lengths):.1f} chars")
    
    # Count by length bins
    short = sum(1 for l in lengths if l < 50)
    normal = sum(1 for l in lengths if 50 <= l <= 160)
    long = sum(1 for l in lengths if l > 160)
    
    print(f"\n   Distribution:")
    print(f"      Too short (<50):  {short:4d} ({short/len(lengths)*100:.1f}%)")
    print(f"      Normal (50-160): {normal:4d} ({normal/len(lengths)*100:.1f}%)")
    print(f"      Too long (>160):  {long:4d} ({long/len(lengths)*100:.1f}%)")
    
    # Claim type coverage
    print(f"\nClaim Type Coverage:")
    claim_counts = Counter()
    messages_per_claim = defaultdict(list)
    
    for entry in data:
        for claim_type in entry['meta']['claim_types']:
            claim_counts[claim_type] += 1
            messages_per_claim[claim_type].append(entry['id'])
    
    # Sort by count
    for claim_type, count in sorted(claim_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(data)) * 100
        print(f"   {claim_type:25s}: {count:4d} ({percentage:5.1f}%)")
    
    # Check for missing claim types
    expected_claims = [
        "IDENTITY_CLAIM", "DELIVERY_CLAIM", "FINANCIAL_CLAIM",
        "ACCOUNT_CLAIM", "URGENCY_CLAIM", "ACTION_CLAIM",
        "VERIFICATION_CLAIM", "SECURITY_CLAIM", "REWARD_CLAIM",
        "LEGAL_CLAIM", "SOCIAL_CLAIM", "CREDENTIALS_CLAIM"
    ]
    
    missing = [c for c in expected_claims if c not in claim_counts]
    if missing:
        print(f"\n   WARNING: Missing claim types: {missing}")
    else:
        print(f"\n   All 12 claim types present!")
    
    # Low coverage warning
    low_coverage = [c for c, cnt in claim_counts.items() if cnt < 30]
    if low_coverage:
        print(f"\n   WARNING: Low coverage (<30 messages):")
        for claim in low_coverage:
            print(f"      - {claim}: {claim_counts[claim]} messages")
    
    # Augmentation type distribution
    print(f"\nAugmentation Types:")
    aug_types = Counter()
    
    for entry in data:
        if entry['meta']['is_augmented']:
            aug_type = entry['meta'].get('augmentation_type', 'unknown')
            aug_types[aug_type] += 1
    
    for aug_type, count in sorted(aug_types.items(), key=lambda x: -x[1]):
        percentage = (count / augmented_count * 100) if augmented_count > 0 else 0
        print(f"   {aug_type:15s}: {count:4d} ({percentage:5.1f}% of augmented)")
    
    # Sample quality check
    print(f"\nSample Messages:")
    print(f"\n   Original samples:")
    originals = [d for d in data if not d['meta']['is_augmented']][:3]
    for i, entry in enumerate(originals, 1):
        text = entry['data']['text'][:80] + "..." if len(entry['data']['text']) > 80 else entry['data']['text']
        claims = ", ".join(entry['meta']['claim_types'][:2])
        print(f"   {i}. {text}")
        print(f"      Claims: {claims}")
    
    print(f"\n   Augmented samples:")
    augmented = [d for d in data if d['meta']['is_augmented']][:3]
    for i, entry in enumerate(augmented, 1):
        text = entry['data']['text'][:80] + "..." if len(entry['data']['text']) > 80 else entry['data']['text']
        claims = ", ".join(entry['meta']['claim_types'][:2])
        aug_type = entry['meta'].get('augmentation_type', 'unknown')
        print(f"   {i}. {text}")
        print(f"      Type: {aug_type}, Claims: {claims}")
    
    # Final verdict
    print(f"\n" + "=" * 80)
    
    issues = []
    if duplicates:
        issues.append(f"{len(duplicates)} duplicates")
    if short > len(data) * 0.05:
        issues.append(f"{short} too-short messages")
    if long > len(data) * 0.1:
        issues.append(f"{long} too-long messages")
    if missing:
        issues.append(f"{len(missing)} missing claim types")
    
    if issues:
        print(f"WARNING: Found issues: {', '.join(issues)}")
        print(f"   Consider regenerating with adjusted parameters")
    else:
        print(f"Dataset quality looks good!")
        print(f"   Ready for annotation")
    
    print()

if __name__ == "__main__":
    dataset_path = 'data/annotations/augmented_phishing_1000.json'
    
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print(f"   Run augmentation first: ./run_augmentation.sh")
    else:
        validate_dataset(dataset_path)

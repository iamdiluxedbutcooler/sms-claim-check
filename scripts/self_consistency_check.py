#!/usr/bin/env python3
"""
Cheaper quality control using self-consistency with GPT-4o-mini.

Strategy:
1. Run GPT-4o-mini 3 times on same message (with temperature=0.3)
2. Check for consistency across runs
3. Flag messages where annotations are inconsistent
4. Much cheaper than using GPT-4o!

Cost comparison (per 100 messages):
- GPT-4o validation: ~$0.50-1.00
- Self-consistency (3x GPT-4o-mini): ~$0.06-0.12
"""

import json
import os
import random
from typing import List, Dict
from openai import OpenAI
from collections import Counter, defaultdict
import time

# Load API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('OPENAI_API_KEY='):
                api_key = line.split('=', 1)[1].strip()
                break

client = OpenAI(api_key=api_key)

# ============================================================================
# SELF-CONSISTENCY CHECK
# ============================================================================

def annotate_with_variation(message: str, run_id: int) -> List[Dict]:
    """Annotate message with slight temperature variation"""
    
    prompt = f"""Extract claim phrases from this SMS phishing message. Be precise and conservative.

Claim types: IDENTITY_CLAIM, DELIVERY_CLAIM, FINANCIAL_CLAIM, ACCOUNT_CLAIM, URGENCY_CLAIM, ACTION_CLAIM, VERIFICATION_CLAIM, SECURITY_CLAIM, REWARD_CLAIM, LEGAL_CLAIM, SOCIAL_CLAIM, CREDENTIALS_CLAIM

IMPORTANT:
- IDENTITY_CLAIM: Only mark if sender explicitly claims identity (e.g., "We are Amazon", "From PayPal")
  - DO NOT mark greetings like "Dear Customer" 
- Only extract claims that are ACTUALLY present in the text

SMS: {message}

JSON:
{{
  "claims": [
    {{"text": "exact phrase", "label": "CLAIM_TYPE"}}
  ]
}}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert NLP annotator. Be precise and conservative."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Slight variation
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        return result.get('claims', [])
        
    except Exception as e:
        print(f"  ERROR in run {run_id}: {e}")
        return []

def check_consistency(claims_runs: List[List[Dict]], message: str) -> Dict:
    """Check consistency across multiple annotation runs"""
    
    # Normalize claims (text + label pairs)
    all_claims = []
    for run_claims in claims_runs:
        run_set = {(c['text'].lower().strip(), c['label']) for c in run_claims}
        all_claims.append(run_set)
    
    # Find claims that appear in all runs (high confidence)
    if not all_claims:
        return {
            'consistent': [],
            'inconsistent': [],
            'consistency_score': 0.0,
            'issue': 'No claims found'
        }
    
    consistent_claims = set.intersection(*all_claims) if len(all_claims) > 0 else set()
    
    # Find claims that appear in some but not all runs (uncertain)
    all_unique = set.union(*all_claims)
    inconsistent_claims = all_unique - consistent_claims
    
    # Calculate consistency score
    consistency_score = len(consistent_claims) / len(all_unique) if all_unique else 1.0
    
    # Identify problematic patterns
    issues = []
    for text, label in inconsistent_claims:
        count = sum(1 for run in all_claims if (text, label) in run)
        if label == 'IDENTITY_CLAIM' and count < len(all_claims):
            issues.append(f"Uncertain IDENTITY_CLAIM: '{text}' (appears in {count}/{len(all_claims)} runs)")
    
    return {
        'consistent': list(consistent_claims),
        'inconsistent': list(inconsistent_claims),
        'consistency_score': consistency_score,
        'total_claims': len(all_unique),
        'issues': issues
    }

# ============================================================================
# QUALITY CONTROL
# ============================================================================

def run_self_consistency_check(annotation_file: str, sample_size: int = 200, num_runs: int = 3):
    """Run self-consistency quality control"""
    
    print("="*60)
    print("SELF-CONSISTENCY QUALITY CONTROL")
    print("="*60)
    print(f"Input: {annotation_file}")
    print(f"Sample size: {sample_size}")
    print(f"Consistency runs: {num_runs}")
    print(f"Model: GPT-4o-mini (temperature=0.3)")
    print()
    
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"Total annotations: {len(annotations)}")
    
    # Sample phishing messages only
    phishing = [a for a in annotations if a['meta'].get('label') != 'ham']
    sampled = random.sample(phishing, min(sample_size, len(phishing)))
    print(f"Sampled: {len(sampled)} phishing messages")
    
    # Run consistency checks
    results = []
    start_time = time.time()
    
    for idx, ann in enumerate(sampled, 1):
        message = ann['data']['text']
        
        print(f"\n[{idx}/{len(sampled)}] Checking consistency...")
        print(f"  Text: {message[:80]}{'...' if len(message) > 80 else ''}")
        
        # Run multiple times
        claims_runs = []
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...", end=" ")
            claims = annotate_with_variation(message, run)
            print(f"{len(claims)} claims")
            claims_runs.append(claims)
            time.sleep(0.3)
        
        # Check consistency
        consistency = check_consistency(claims_runs, message)
        
        print(f"  Consistency score: {consistency['consistency_score']:.2f}")
        print(f"  Consistent claims: {len(consistency['consistent'])}")
        print(f"  Inconsistent claims: {len(consistency['inconsistent'])}")
        
        if consistency['issues']:
            print(f"  [WARNING] Issues detected:")
            for issue in consistency['issues'][:3]:
                print(f"    - {issue}")
        
        results.append({
            'message': message,
            'original_annotation': ann,
            'consistency': consistency,
            'runs': claims_runs
        })
        
        # Progress
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = (len(sampled) - idx) * avg_time
        print(f"  Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min")
    
    # Generate report
    output_path = annotation_file.replace('.json', '_consistency_report.json')
    
    # Aggregate statistics
    avg_consistency = sum(r['consistency']['consistency_score'] for r in results) / len(results)
    low_consistency = [r for r in results if r['consistency']['consistency_score'] < 0.7]
    identity_issues = [r for r in results if any('IDENTITY_CLAIM' in issue for issue in r['consistency']['issues'])]
    
    report = {
        'summary': {
            'total_samples': len(results),
            'avg_consistency_score': round(avg_consistency, 3),
            'low_consistency_count': len(low_consistency),
            'identity_claim_issues': len(identity_issues)
        },
        'low_consistency_messages': low_consistency[:30],
        'identity_claim_issues': identity_issues[:30],
        'all_results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("CONSISTENCY CHECK SUMMARY")
    print("="*60)
    print(f"Samples analyzed: {len(results)}")
    print(f"Average consistency: {avg_consistency:.3f}")
    print(f"Low consistency (<0.7): {len(low_consistency)}")
    print(f"Identity claim issues: {len(identity_issues)}")
    print(f"\nReport saved to: {output_path}")
    print("="*60)
    
    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    random.seed(42)
    
    # Run self-consistency check
    run_self_consistency_check(
        annotation_file="data/annotations/claim_annotations_2000.json",
        sample_size=100,  # Start with 100 to see results quickly
        num_runs=3
    )

if __name__ == "__main__":
    main()

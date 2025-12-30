#!/usr/bin/env python3
"""
Quality control for automated annotations using cross-validation with GPT-4o.

Strategy:
1. Sample 100-200 annotated messages (stratified by claim types)
2. Re-annotate with GPT-4o (smarter, more accurate)
3. Compare annotations and identify disagreements
4. Generate quality report with problematic cases
"""

import json
import os
import random
from typing import List, Dict, Tuple
from openai import OpenAI
from collections import defaultdict, Counter
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
# GPT-4o PROMPTS (More detailed for quality control)
# ============================================================================

GPT4_CLAIM_PROMPT = """You are an expert at analyzing SMS phishing messages and extracting semantic claims.

IMPORTANT CLARIFICATIONS:
- **IDENTITY_CLAIM**: WHO the sender claims to be (e.g., "We are Amazon", "From PayPal", "Your bank")
  - DO NOT mark greetings like "Dear Customer" as identity claims
  - Only mark explicit or implicit assertions about sender identity
  
- **ACTION_CLAIM**: Specific actions requested (e.g., "Click here", "Call 1-800-123", "Reply YES")
  - Must be clear call-to-action, not just descriptive text
  
- **URGENCY_CLAIM**: Time pressure tactics (e.g., "within 24 hours", "expires today", "act now")
  - Must convey time constraint or urgency

- **FINANCIAL_CLAIM**: Money-related assertions (e.g., "You won $500", "Refund pending", "Unauthorized charge")

- **DELIVERY_CLAIM**: Package/shipment assertions (e.g., "Your package is delayed")

- **ACCOUNT_CLAIM**: Account status issues (e.g., "Account suspended", "Unusual activity detected")

- **VERIFICATION_CLAIM**: Requests to verify/confirm (e.g., "Verify your identity", "Confirm your details")

- **SECURITY_CLAIM**: Security threats (e.g., "Suspicious login detected", "Your account is at risk")

- **REWARD_CLAIM**: Prizes/rewards (e.g., "You've won", "Claim your prize")

- **LEGAL_CLAIM**: Legal threats (e.g., "Legal action pending", "Court summons")

- **SOCIAL_CLAIM**: Social engineering (e.g., "Your friend sent", "Family member needs help")

- **CREDENTIALS_CLAIM**: Requests for passwords/PINs (e.g., "Enter your password")

Extract claim phrases that are ACTUALLY PRESENT in the text. Be precise and conservative.

SMS Message:
{message}

Return JSON:
{{
  "claims": [
    {{"text": "exact phrase from message", "label": "CLAIM_TYPE", "reasoning": "why this is this claim type"}}
  ]
}}

JSON:"""

# ============================================================================
# SAMPLING STRATEGY
# ============================================================================

def load_annotations(file_path: str) -> List[Dict]:
    """Load annotated dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def stratified_sample(annotations: List[Dict], sample_size: int = 150) -> List[Dict]:
    """Sample messages stratified by number of claims"""
    # Group by claim count
    by_claim_count = defaultdict(list)
    
    for ann in annotations:
        label = ann['meta'].get('label', 'phishing')
        if label == 'ham':
            by_claim_count[0].append(ann)
        else:
            claim_count = len(ann['annotations'][0]['result'])
            by_claim_count[claim_count].append(ann)
    
    # Sample proportionally
    sampled = []
    for claim_count, items in sorted(by_claim_count.items()):
        # Sample proportional to size, with minimum
        proportion = len(items) / len(annotations)
        sample_n = max(int(sample_size * proportion), 5)
        sample_n = min(sample_n, len(items))
        
        sampled.extend(random.sample(items, sample_n))
    
    # If we have too many, trim
    if len(sampled) > sample_size:
        sampled = random.sample(sampled, sample_size)
    
    return sampled

# ============================================================================
# GPT-4o RE-ANNOTATION
# ============================================================================

def reannotate_with_gpt4(message: str) -> List[Dict]:
    """Re-annotate with GPT-4o for quality comparison"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # More expensive but more accurate
            messages=[
                {"role": "system", "content": "You are an expert NLP annotator specializing in phishing claim extraction. Be precise and conservative."},
                {"role": "user", "content": GPT4_CLAIM_PROMPT.format(message=message)}
            ],
            temperature=0,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        return result.get('claims', [])
        
    except Exception as e:
        print(f"  ERROR in GPT-4o: {e}")
        return []

# ============================================================================
# COMPARISON & QUALITY METRICS
# ============================================================================

def extract_claims_from_annotation(annotation: Dict) -> List[Dict]:
    """Extract claims from Label Studio annotation"""
    claims = []
    for result in annotation['annotations'][0]['result']:
        claims.append({
            'text': result['value']['text'],
            'label': result['value']['labels'][0],
            'start': result['value']['start'],
            'end': result['value']['end']
        })
    return claims

def compare_annotations(mini_claims: List[Dict], gpt4_claims: List[Dict], message: str) -> Dict:
    """Compare GPT-4o-mini vs GPT-4o annotations"""
    
    # Extract claim texts and labels
    mini_set = {(c['text'].lower().strip(), c['label']) for c in mini_claims}
    gpt4_set = {(c['text'].lower().strip(), c['label']) for c in gpt4_claims}
    
    # Find agreements and disagreements
    agreement = mini_set & gpt4_set
    only_mini = mini_set - gpt4_set
    only_gpt4 = gpt4_set - mini_set
    
    # Calculate metrics
    precision = len(agreement) / len(mini_set) if mini_set else 1.0
    recall = len(agreement) / len(gpt4_set) if gpt4_set else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'agreement': list(agreement),
        'only_mini': list(only_mini),
        'only_gpt4': list(only_gpt4),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mini_count': len(mini_set),
        'gpt4_count': len(gpt4_set)
    }

# ============================================================================
# QUALITY REPORT GENERATION
# ============================================================================

def generate_quality_report(comparisons: List[Dict], output_path: str):
    """Generate detailed quality report"""
    
    # Aggregate statistics
    total_samples = len(comparisons)
    avg_precision = sum(c['metrics']['precision'] for c in comparisons) / total_samples
    avg_recall = sum(c['metrics']['recall'] for c in comparisons) / total_samples
    avg_f1 = sum(c['metrics']['f1'] for c in comparisons) / total_samples
    
    # Find problematic claim types
    claim_errors = defaultdict(int)
    for comp in comparisons:
        for text, label in comp['metrics']['only_mini']:
            claim_errors[label] += 1
    
    # Find common issues
    identity_issues = []
    for comp in comparisons:
        for text, label in comp['metrics']['only_mini']:
            if label == 'IDENTITY_CLAIM':
                identity_issues.append({
                    'message': comp['message'][:100],
                    'text': text,
                    'issue': 'Likely false positive (greeting, not identity claim)'
                })
    
    # Write report
    report = {
        'summary': {
            'total_samples': total_samples,
            'avg_precision': round(avg_precision, 3),
            'avg_recall': round(avg_recall, 3),
            'avg_f1': round(avg_f1, 3)
        },
        'problematic_claim_types': dict(sorted(claim_errors.items(), key=lambda x: x[1], reverse=True)),
        'identity_claim_issues': identity_issues[:20],  # Top 20 issues
        'detailed_comparisons': comparisons[:50]  # Top 50 for manual review
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("QUALITY CONTROL SUMMARY")
    print("="*60)
    print(f"Samples analyzed: {total_samples}")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    print(f"\nProblematic claim types:")
    for claim_type, count in sorted(claim_errors.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {claim_type}: {count} errors")
    print(f"\nIdentity claim issues found: {len(identity_issues)}")
    print(f"\nDetailed report saved to: {output_path}")
    print("="*60)

# ============================================================================
# MAIN QUALITY CONTROL PIPELINE
# ============================================================================

def run_quality_control(annotation_file: str, sample_size: int = 150):
    """Run quality control on annotations"""
    
    print("="*60)
    print("ANNOTATION QUALITY CONTROL")
    print("="*60)
    print(f"Input: {annotation_file}")
    print(f"Sample size: {sample_size}")
    print(f"Validation model: GPT-4o")
    print()
    
    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations(annotation_file)
    print(f"Total annotations: {len(annotations)}")
    
    # Sample
    print(f"Sampling {sample_size} messages...")
    sampled = stratified_sample(annotations, sample_size)
    print(f"Sampled: {len(sampled)} messages")
    
    # Compare
    comparisons = []
    start_time = time.time()
    
    for idx, ann in enumerate(sampled, 1):
        message = ann['data']['text']
        label = ann['meta'].get('label', 'phishing')
        
        print(f"\n[{idx}/{len(sampled)}] Validating...")
        print(f"  Text: {message[:80]}{'...' if len(message) > 80 else ''}")
        
        if label == 'ham':
            print(f"  Skipping HAM message")
            continue
        
        # Get original claims
        mini_claims = extract_claims_from_annotation(ann)
        print(f"  GPT-4o-mini: {len(mini_claims)} claims")
        
        # Re-annotate with GPT-4o
        print(f"  Re-annotating with GPT-4o...")
        gpt4_claims = reannotate_with_gpt4(message)
        print(f"  GPT-4o: {len(gpt4_claims)} claims")
        
        # Compare
        metrics = compare_annotations(mini_claims, gpt4_claims, message)
        print(f"  Agreement F1: {metrics['f1']:.3f}")
        
        if metrics['only_mini']:
            print(f"  [WARNING] GPT-4o-mini added {len(metrics['only_mini'])} claims that GPT-4o didn't:")
            for text, label_type in list(metrics['only_mini'])[:3]:
                print(f"    - '{text}' ({label_type})")
        
        comparisons.append({
            'message': message,
            'label': label,
            'mini_claims': mini_claims,
            'gpt4_claims': gpt4_claims,
            'metrics': metrics
        })
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = (len(sampled) - idx) * avg_time
        print(f"  Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Generate report
    output_path = annotation_file.replace('.json', '_quality_report.json')
    generate_quality_report(comparisons, output_path)
    
    return comparisons

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Set random seed
    random.seed(42)
    
    # Run quality control on claim annotations
    run_quality_control(
        annotation_file="data/annotations/claim_annotations_2000.json",
        sample_size=150  # Adjust based on budget (GPT-4o is more expensive)
    )

if __name__ == "__main__":
    main()

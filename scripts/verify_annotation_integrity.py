"""
Data Integrity Verification Script

Compares raw SMS messages with annotated messages to ensure:
1. No text modifications by GPT-4o
2. No hallucinated messages
3. All raw messages were annotated
4. Character-level exactness
"""

import json
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher, unified_diff
from typing import Dict, List, Tuple
import argparse


class IntegrityChecker:
    """Verify annotation data integrity against raw data"""
    
    def __init__(self, raw_path: str, entity_path: str, claim_path: str):
        self.raw_path = Path(raw_path)
        self.entity_path = Path(entity_path)
        self.claim_path = Path(claim_path)
        
        # Load data
        print("[LOADING] Raw data...")
        self.raw_df = pd.read_csv(self.raw_path)
        
        print("[LOADING] Entity annotations...")
        with open(self.entity_path, 'r') as f:
            self.entity_data = json.load(f)
        
        print("[LOADING] Claim annotations...")
        with open(self.claim_path, 'r') as f:
            self.claim_data = json.load(f)
        
        print(f"[OK] Loaded {len(self.raw_df)} raw messages")
        print(f"[OK] Loaded {len(self.entity_data)} entity annotations")
        print(f"[OK] Loaded {len(self.claim_data)} claim annotations")
        print()
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison (handle whitespace variations)"""
        if pd.isna(text):
            return ""
        return str(text).strip()
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity ratio between two texts"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def check_exact_matches(self) -> Dict:
        """Check for exact character-level matches"""
        print("=" * 60)
        print("EXACT MATCH VERIFICATION")
        print("=" * 60)
        
        results = {
            'entity_exact': 0,
            'claim_exact': 0,
            'entity_mismatches': [],
            'claim_mismatches': [],
            'entity_missing': [],
            'claim_missing': []
        }
        
        # Create lookup for raw messages
        raw_messages = {}
        # Handle both 'TEXT' and 'text' column names
        text_col = 'TEXT' if 'TEXT' in self.raw_df.columns else 'text'
        for idx, row in self.raw_df.iterrows():
            msg_id = str(row.get('id', idx))
            text = self.normalize_text(row[text_col])
            raw_messages[msg_id] = text
        
        print(f"[CHECK] Verifying {len(self.entity_data)} entity annotations...")
        
        # Check entity annotations
        for annotation in self.entity_data:
            msg_id = str(annotation['data']['message_id'])
            annotated_text = self.normalize_text(annotation['data']['text'])
            
            if msg_id not in raw_messages:
                results['entity_missing'].append({
                    'id': msg_id,
                    'text': annotated_text[:100]
                })
                continue
            
            raw_text = raw_messages[msg_id]
            
            if raw_text == annotated_text:
                results['entity_exact'] += 1
            else:
                similarity = self.compute_similarity(raw_text, annotated_text)
                results['entity_mismatches'].append({
                    'id': msg_id,
                    'similarity': similarity,
                    'raw_length': len(raw_text),
                    'annotated_length': len(annotated_text),
                    'raw_text': raw_text[:200],
                    'annotated_text': annotated_text[:200]
                })
        
        print(f"[CHECK] Verifying {len(self.claim_data)} claim annotations...")
        
        # Check claim annotations
        for annotation in self.claim_data:
            msg_id = str(annotation['data']['message_id'])
            annotated_text = self.normalize_text(annotation['data']['text'])
            
            if msg_id not in raw_messages:
                results['claim_missing'].append({
                    'id': msg_id,
                    'text': annotated_text[:100]
                })
                continue
            
            raw_text = raw_messages[msg_id]
            
            if raw_text == annotated_text:
                results['claim_exact'] += 1
            else:
                similarity = self.compute_similarity(raw_text, annotated_text)
                results['claim_mismatches'].append({
                    'id': msg_id,
                    'similarity': similarity,
                    'raw_length': len(raw_text),
                    'annotated_length': len(annotated_text),
                    'raw_text': raw_text[:200],
                    'annotated_text': annotated_text[:200]
                })
        
        return results
    
    def check_coverage(self) -> Dict:
        """Check if all raw messages were annotated"""
        print("=" * 60)
        print("COVERAGE VERIFICATION")
        print("=" * 60)
        
        raw_ids = set(str(idx) for idx in self.raw_df.index)
        entity_ids = set(str(ann['data']['message_id']) for ann in self.entity_data)
        claim_ids = set(str(ann['data']['message_id']) for ann in self.claim_data)
        
        results = {
            'total_raw': len(raw_ids),
            'entity_annotated': len(entity_ids),
            'claim_annotated': len(claim_ids),
            'entity_missing_ids': list(raw_ids - entity_ids),
            'claim_missing_ids': list(raw_ids - claim_ids),
            'entity_extra_ids': list(entity_ids - raw_ids),
            'claim_extra_ids': list(claim_ids - raw_ids)
        }
        
        return results
    
    def check_annotation_spans(self) -> Dict:
        """Verify annotation spans are within message bounds"""
        print("=" * 60)
        print("SPAN BOUNDARY VERIFICATION")
        print("=" * 60)
        
        results = {
            'entity_invalid_spans': [],
            'claim_invalid_spans': []
        }
        
        # Check entity spans
        for annotation in self.entity_data:
            text = annotation['data']['text']
            text_length = len(text)
            msg_id = str(annotation['data']['message_id'])
            
            for result in annotation.get('predictions', [{}])[0].get('result', []):
                if result['type'] != 'labels':
                    continue
                
                start = result['value']['start']
                end = result['value']['end']
                extracted = result['value']['text']
                
                # Check bounds
                if start < 0 or end > text_length or start >= end:
                    results['entity_invalid_spans'].append({
                        'id': msg_id,
                        'text_length': text_length,
                        'start': start,
                        'end': end,
                        'extracted': extracted
                    })
                    continue
                
                # Check if extracted text matches span
                actual_text = text[start:end]
                if actual_text != extracted:
                    results['entity_invalid_spans'].append({
                        'id': msg_id,
                        'issue': 'text_mismatch',
                        'start': start,
                        'end': end,
                        'extracted': extracted,
                        'actual': actual_text
                    })
        
        # Check claim spans
        for annotation in self.claim_data:
            text = annotation['data']['text']
            text_length = len(text)
            msg_id = str(annotation['data']['message_id'])
            
            for result in annotation.get('predictions', [{}])[0].get('result', []):
                if result['type'] != 'labels':
                    continue
                
                start = result['value']['start']
                end = result['value']['end']
                extracted = result['value']['text']
                
                # Check bounds
                if start < 0 or end > text_length or start >= end:
                    results['claim_invalid_spans'].append({
                        'id': msg_id,
                        'text_length': text_length,
                        'start': start,
                        'end': end,
                        'extracted': extracted
                    })
                    continue
                
                # Check if extracted text matches span
                actual_text = text[start:end]
                if actual_text != extracted:
                    results['claim_invalid_spans'].append({
                        'id': msg_id,
                        'issue': 'text_mismatch',
                        'start': start,
                        'end': end,
                        'extracted': extracted,
                        'actual': actual_text
                    })
        
        return results
    
    def print_detailed_diff(self, mismatch: Dict):
        """Print detailed character-level diff"""
        print(f"\n[MISMATCH] ID: {mismatch['id']}")
        print(f"Similarity: {mismatch['similarity']:.2%}")
        print(f"Length: Raw={mismatch['raw_length']}, Annotated={mismatch['annotated_length']}")
        print("\n--- RAW ---")
        print(mismatch['raw_text'])
        print("\n--- ANNOTATED ---")
        print(mismatch['annotated_text'])
        
        # Character-level diff
        diff = unified_diff(
            mismatch['raw_text'].splitlines(keepends=True),
            mismatch['annotated_text'].splitlines(keepends=True),
            fromfile='raw',
            tofile='annotated',
            lineterm=''
        )
        print("\n--- DIFF ---")
        print(''.join(diff))
        print("-" * 60)
    
    def generate_report(self):
        """Generate comprehensive integrity report"""
        print("\n")
        print("=" * 60)
        print("DATA INTEGRITY VERIFICATION REPORT")
        print("=" * 60)
        print()
        
        # Check exact matches
        match_results = self.check_exact_matches()
        print()
        
        # Check coverage
        coverage_results = self.check_coverage()
        print()
        
        # Check annotation spans
        span_results = self.check_annotation_spans()
        print()
        
        # Summary
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print()
        
        print("[EXACT MATCHES]")
        print(f"  Entity:  {match_results['entity_exact']}/{len(self.entity_data)} "
              f"({match_results['entity_exact']/len(self.entity_data)*100:.1f}%)")
        print(f"  Claim:   {match_results['claim_exact']}/{len(self.claim_data)} "
              f"({match_results['claim_exact']/len(self.claim_data)*100:.1f}%)")
        print()
        
        print("[COVERAGE]")
        print(f"  Raw messages: {coverage_results['total_raw']}")
        print(f"  Entity annotated: {coverage_results['entity_annotated']}")
        print(f"  Claim annotated: {coverage_results['claim_annotated']}")
        print(f"  Entity missing: {len(coverage_results['entity_missing_ids'])}")
        print(f"  Claim missing: {len(coverage_results['claim_missing_ids'])}")
        print(f"  Entity extra: {len(coverage_results['entity_extra_ids'])}")
        print(f"  Claim extra: {len(coverage_results['claim_extra_ids'])}")
        print()
        
        print("[SPAN INTEGRITY]")
        print(f"  Entity invalid spans: {len(span_results['entity_invalid_spans'])}")
        print(f"  Claim invalid spans: {len(span_results['claim_invalid_spans'])}")
        print()
        
        # Detailed issues
        if match_results['entity_mismatches']:
            print("=" * 60)
            print(f"ENTITY TEXT MISMATCHES ({len(match_results['entity_mismatches'])})")
            print("=" * 60)
            for mismatch in match_results['entity_mismatches'][:5]:  # Show first 5
                self.print_detailed_diff(mismatch)
        
        if match_results['claim_mismatches']:
            print("=" * 60)
            print(f"CLAIM TEXT MISMATCHES ({len(match_results['claim_mismatches'])})")
            print("=" * 60)
            for mismatch in match_results['claim_mismatches'][:5]:  # Show first 5
                self.print_detailed_diff(mismatch)
        
        if span_results['entity_invalid_spans']:
            print("=" * 60)
            print(f"ENTITY INVALID SPANS ({len(span_results['entity_invalid_spans'])})")
            print("=" * 60)
            for span in span_results['entity_invalid_spans'][:10]:
                print(f"\n{span}")
        
        if span_results['claim_invalid_spans']:
            print("=" * 60)
            print(f"CLAIM INVALID SPANS ({len(span_results['claim_invalid_spans'])})")
            print("=" * 60)
            for span in span_results['claim_invalid_spans'][:10]:
                print(f"\n{span}")
        
        # Final verdict
        print()
        print("=" * 60)
        print("VERDICT")
        print("=" * 60)
        
        total_issues = (
            len(match_results['entity_mismatches']) +
            len(match_results['claim_mismatches']) +
            len(coverage_results['entity_missing_ids']) +
            len(coverage_results['claim_missing_ids']) +
            len(coverage_results['entity_extra_ids']) +
            len(coverage_results['claim_extra_ids']) +
            len(span_results['entity_invalid_spans']) +
            len(span_results['claim_invalid_spans'])
        )
        
        if total_issues == 0:
            print("[PASS] All integrity checks passed!")
            print("[OK] Data is ready for training.")
        else:
            print(f"[WARNING] Found {total_issues} integrity issues")
            print("[ACTION] Review issues above before training")
        
        print()
        
        # Save detailed report
        report = {
            'exact_matches': match_results,
            'coverage': coverage_results,
            'span_integrity': span_results,
            'total_issues': total_issues
        }
        
        output_path = Path('data/eda/integrity_report.json')
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[SAVED] Detailed report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Verify annotation data integrity')
    parser.add_argument('--raw', default='data/raw/mendeley.csv',
                       help='Path to raw CSV file')
    parser.add_argument('--entity', default='data/annotations/entity_annotations.json',
                       help='Path to entity annotations')
    parser.add_argument('--claim', default='data/annotations/claim_annotations.json',
                       help='Path to claim annotations')
    
    args = parser.parse_args()
    
    checker = IntegrityChecker(args.raw, args.entity, args.claim)
    checker.generate_report()


if __name__ == '__main__':
    main()

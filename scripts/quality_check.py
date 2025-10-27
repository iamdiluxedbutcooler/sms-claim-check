import json
import re
import argparse
from pathlib import Path
from typing import List, Dict
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

import sys
sys.path.append(str(PROJECT_ROOT / "config"))
from entity_schema import ENTITY_TYPES, validate_entity_annotation

class AnnotationValidator:
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
    
    def check_trailing_punctuation(self, annotations: List[Dict]) -> List[Dict]:
        issues = []
        
        for ann in annotations:
            span_text = ann.get('text', '')
            if span_text and span_text[-1] in '.!?,;:':
                issues.append({
                    'message_id': ann.get('message_id'),
                    'entity_type': ann.get('label'),
                    'span_text': span_text,
                    'issue': 'Trailing punctuation',
                    'suggestion': span_text.rstrip('.!?,;:')
                })
        
        return issues
    
    def check_whitespace(self, annotations: List[Dict]) -> List[Dict]:
        issues = []
        
        for ann in annotations:
            span_text = ann.get('text', '')
            if span_text != span_text.strip():
                issues.append({
                    'message_id': ann.get('message_id'),
                    'entity_type': ann.get('label'),
                    'span_text': repr(span_text),
                    'issue': 'Leading/trailing whitespace',
                    'suggestion': span_text.strip()
                })
        
        return issues
    
    def check_entity_validity(self, annotations: List[Dict]) -> List[Dict]:
        issues = []
        
        for ann in annotations:
            entity_type = ann.get('label')
            span_text = ann.get('text', '')
            
            if entity_type == 'PHONE':
                if not any(c.isdigit() for c in span_text):
                    issues.append({
                        'message_id': ann.get('message_id'),
                        'entity_type': entity_type,
                        'span_text': span_text,
                        'issue': 'PHONE without digits',
                    })
            
            elif entity_type == 'URL':
                if '.' not in span_text and '/' not in span_text:
                    issues.append({
                        'message_id': ann.get('message_id'),
                        'entity_type': entity_type,
                        'span_text': span_text,
                        'issue': 'URL without domain indicators',
                    })
            
            elif entity_type == 'AMOUNT':
                if not any(c.isdigit() for c in span_text):
                    issues.append({
                        'message_id': ann.get('message_id'),
                        'entity_type': entity_type,
                        'span_text': span_text,
                        'issue': 'AMOUNT without numbers',
                    })
        
        return issues
    
    def check_overlapping_annotations(self, annotations: List[Dict]) -> List[Dict]:
        issues = []
        
        by_message = defaultdict(list)
        for ann in annotations:
            by_message[ann.get('message_id')].append(ann)
        
        for message_id, msg_anns in by_message.items():
            sorted_anns = sorted(msg_anns, key=lambda x: x.get('start', 0))
            
            for i in range(len(sorted_anns) - 1):
                curr = sorted_anns[i]
                next_ann = sorted_anns[i + 1]
                
                if curr.get('end', 0) > next_ann.get('start', 0):
                    issues.append({
                        'message_id': message_id,
                        'issue': 'Overlapping annotations',
                        'annotation_1': f"{curr.get('label')}: {curr.get('text')}",
                        'annotation_2': f"{next_ann.get('label')}: {next_ann.get('text')}"
                    })
        
        return issues
    
    def get_entity_distribution(self, annotations: List[Dict]) -> Dict[str, int]:
        entity_counts = Counter()
        
        for ann in annotations:
            entity_counts[ann.get('label')] += 1
        
        return dict(entity_counts)
    
    def get_message_stats(self, annotations: List[Dict]) -> Dict:
        by_message = defaultdict(list)
        for ann in annotations:
            by_message[ann.get('message_id')].append(ann)
        
        entities_per_message = [len(anns) for anns in by_message.values()]
        
        return {
            'total_messages': len(by_message),
            'total_entities': len(annotations),
            'avg_entities_per_message': sum(entities_per_message) / len(entities_per_message) if entities_per_message else 0,
            'messages_with_no_entities': sum(1 for count in entities_per_message if count == 0),
            'max_entities_in_message': max(entities_per_message) if entities_per_message else 0,
        }
    
    def validate_all(self, annotations: List[Dict]) -> Dict:
        results = {
            'trailing_punctuation': self.check_trailing_punctuation(annotations),
            'whitespace_issues': self.check_whitespace(annotations),
            'entity_validity': self.check_entity_validity(annotations),
            'overlapping': self.check_overlapping_annotations(annotations),
            'entity_distribution': self.get_entity_distribution(annotations),
            'message_stats': self.get_message_stats(annotations),
        }
        
        return results

def load_annotations(filepath: Path) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_annotations = []
    
    for item in data:
        message_id = item.get('data', {}).get('message_id')
        
        for annotation in item.get('annotations', []):
            for result in annotation.get('result', []):
                if result.get('type') == 'labels':
                    value = result.get('value', {})
                    all_annotations.append({
                        'message_id': message_id,
                        'label': value.get('labels', [''])[0],
                        'text': value.get('text'),
                        'start': value.get('start'),
                        'end': value.get('end'),
                    })
    
    return all_annotations

def print_report(results: Dict):
    print("\n" + "="*60)
    print("ANNOTATION QUALITY REPORT")
    print("="*60)
    
    stats = results['message_stats']
    print("\nMessage Statistics:")
    print(f"  Total messages annotated: {stats['total_messages']}")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Avg entities per message: {stats['avg_entities_per_message']:.2f}")
    print(f"  Messages with no entities: {stats['messages_with_no_entities']}")
    print(f"  Max entities in a message: {stats['max_entities_in_message']}")
    
    print("\nEntity Type Distribution:")
    for entity_type, count in sorted(results['entity_distribution'].items(), key=lambda x: -x[1]):
        percentage = (count / stats['total_entities'] * 100) if stats['total_entities'] > 0 else 0
        print(f"  {entity_type:20s}: {count:4d} ({percentage:5.2f}%)")
    
    total_issues = (
        len(results['trailing_punctuation']) +
        len(results['whitespace_issues']) +
        len(results['entity_validity']) +
        len(results['overlapping'])
    )
    
    if total_issues == 0:
        print("\nNo issues found.")
    else:
        print(f"\nFound {total_issues} issues:")
        
        if results['trailing_punctuation']:
            print(f"\n  Trailing Punctuation ({len(results['trailing_punctuation'])} issues):")
            for issue in results['trailing_punctuation'][:5]:
                print(f"    - {issue['entity_type']}: '{issue['span_text']}' -> '{issue['suggestion']}'")
            if len(results['trailing_punctuation']) > 5:
                print(f"    ... and {len(results['trailing_punctuation']) - 5} more")
        
        if results['whitespace_issues']:
            print(f"\n  Whitespace Issues ({len(results['whitespace_issues'])} issues):")
            for issue in results['whitespace_issues'][:5]:
                print(f"    - {issue['entity_type']}: {issue['span_text']} -> '{issue['suggestion']}'")
            if len(results['whitespace_issues']) > 5:
                print(f"    ... and {len(results['whitespace_issues']) - 5} more")
        
        if results['entity_validity']:
            print(f"\n  Entity Validity Issues ({len(results['entity_validity'])} issues):")
            for issue in results['entity_validity'][:5]:
                print(f"    - {issue['entity_type']}: '{issue['span_text']}' - {issue['issue']}")
            if len(results['entity_validity']) > 5:
                print(f"    ... and {len(results['entity_validity']) - 5} more")
        
        if results['overlapping']:
            print(f"\n  Overlapping Annotations ({len(results['overlapping'])} issues):")
            for issue in results['overlapping'][:5]:
                print(f"    - Message {issue['message_id']}: {issue['annotation_1']} overlaps {issue['annotation_2']}")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="Validate annotation quality")
    parser.add_argument('--annotations', type=str)
    parser.add_argument('--last', type=int)
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ANNOTATION QUALITY CHECK")
    print("="*60)
    
    if args.annotations:
        ann_file = Path(args.annotations)
    else:
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        ann_files = list(ANNOTATIONS_DIR.glob("*.json"))
        
        if not ann_files:
            print("\nNo annotation files found in data/annotations/")
            print("\nExport annotations from Label Studio:")
            print("  1. Click 'Export'")
            print("  2. Choose 'JSON' format")
            print("  3. Save to data/annotations/")
            return
        
        ann_file = max(ann_files, key=lambda p: p.stat().st_mtime)
        print(f"\nUsing most recent: {ann_file.name}")
    
    if not ann_file.exists():
        print(f"\nError: File not found: {ann_file}")
        return
    
    print(f"\nLoading annotations...")
    annotations = load_annotations(ann_file)
    print(f"Loaded {len(annotations)} entity annotations")
    
    if args.last:
        by_message = defaultdict(list)
        for ann in annotations:
            by_message[ann['message_id']].append(ann)
        
        last_messages = list(by_message.keys())[-args.last:]
        annotations = [ann for ann in annotations if ann['message_id'] in last_messages]
        print(f"Filtered to last {args.last} messages ({len(annotations)} annotations)")
    
    print(f"\nRunning quality checks...")
    validator = AnnotationValidator()
    results = validator.validate_all(annotations)
    
    print_report(results)
    
    report_file = ANNOTATIONS_DIR / "quality_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed report saved to: {report_file}")

if __name__ == "__main__":
    main()

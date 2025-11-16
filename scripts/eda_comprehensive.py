"""
Comprehensive EDA for Annotation Quality Analysis

Analyzes both entity-based and claim-based annotations to understand:
1. Distribution of entities/claims
2. Annotation quality and consistency
3. Message characteristics
4. Comparison between annotation types
"""

import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class AnnotationEDA:
    def __init__(self, annotation_file: str, annotation_type: str = "entity"):
        """
        Args:
            annotation_file: Path to annotation JSON file
            annotation_type: "entity" or "claim"
        """
        self.annotation_file = annotation_file
        self.annotation_type = annotation_type
        
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} annotated messages")
    
    def extract_annotations(self) -> List[Dict]:
        """Extract annotations from Label Studio format"""
        annotations = []
        
        for item in self.data:
            msg_id = item['data'].get('message_id', 'unknown')
            text = item['data']['text']
            label = item['data'].get('label', 'unknown')
            
            # Get predictions or completions
            if 'predictions' in item and item['predictions']:
                results = item['predictions'][0].get('result', [])
            elif 'annotations' in item and item['annotations']:
                results = item['annotations'][0].get('result', [])
            else:
                results = []
            
            for ann in results:
                if ann.get('type') == 'labels':
                    value = ann.get('value', {})
                    annotations.append({
                        'message_id': msg_id,
                        'message_text': text,
                        'message_label': label,
                        'annotation_text': value.get('text', ''),
                        'annotation_label': value.get('labels', ['UNKNOWN'])[0],
                        'start': value.get('start', 0),
                        'end': value.get('end', 0),
                        'annotation_length': len(value.get('text', ''))
                    })
        
        return annotations
    
    def basic_statistics(self, annotations: List[Dict]) -> Dict:
        """Compute basic statistics"""
        df = pd.DataFrame(annotations)
        
        stats = {
            'total_messages': df['message_id'].nunique(),
            'total_annotations': len(df),
            'avg_annotations_per_message': len(df) / df['message_id'].nunique(),
            'messages_with_no_annotations': len(self.data) - df['message_id'].nunique(),
            'avg_annotation_length': df['annotation_length'].mean(),
            'median_annotation_length': df['annotation_length'].median(),
            'max_annotation_length': df['annotation_length'].max(),
            'min_annotation_length': df['annotation_length'].min()
        }
        
        # Annotations per message distribution
        anns_per_msg = df.groupby('message_id').size()
        stats['min_annotations_per_message'] = anns_per_msg.min()
        stats['max_annotations_per_message'] = anns_per_msg.max()
        stats['std_annotations_per_message'] = anns_per_msg.std()
        
        return stats
    
    def label_distribution(self, annotations: List[Dict]) -> Dict:
        """Analyze label distribution"""
        df = pd.DataFrame(annotations)
        
        label_counts = df['annotation_label'].value_counts().to_dict()
        total = len(df)
        label_percentages = {k: (v/total)*100 for k, v in label_counts.items()}
        
        return {
            'counts': label_counts,
            'percentages': label_percentages
        }
    
    def annotation_overlap_analysis(self, annotations: List[Dict]) -> Dict:
        """Detect overlapping annotations"""
        df = pd.DataFrame(annotations)
        
        overlaps = []
        
        for msg_id in df['message_id'].unique():
            msg_anns = df[df['message_id'] == msg_id].sort_values('start')
            
            for i in range(len(msg_anns) - 1):
                curr = msg_anns.iloc[i]
                next_ann = msg_anns.iloc[i + 1]
                
                if curr['end'] > next_ann['start']:
                    overlaps.append({
                        'message_id': msg_id,
                        'ann1_label': curr['annotation_label'],
                        'ann1_text': curr['annotation_text'],
                        'ann2_label': next_ann['annotation_label'],
                        'ann2_text': next_ann['annotation_text'],
                        'overlap_chars': curr['end'] - next_ann['start']
                    })
        
        return {
            'total_overlaps': len(overlaps),
            'overlap_rate': len(overlaps) / len(df) if len(df) > 0 else 0,
            'overlaps': overlaps[:20]  # Sample
        }
    
    def message_length_analysis(self, annotations: List[Dict]) -> Dict:
        """Analyze message lengths"""
        df = pd.DataFrame(annotations)
        
        msg_lengths = df.groupby('message_id').agg({
            'message_text': lambda x: len(x.iloc[0]),
            'annotation_label': 'count'
        }).rename(columns={'annotation_label': 'num_annotations'})
        
        return {
            'avg_message_length': msg_lengths['message_text'].mean(),
            'median_message_length': msg_lengths['message_text'].median(),
            'correlation_length_annotations': msg_lengths.corr().iloc[0, 1]
        }
    
    def annotation_quality_checks(self, annotations: List[Dict]) -> Dict:
        """Check annotation quality issues"""
        df = pd.DataFrame(annotations)
        
        issues = {
            'trailing_punctuation': [],
            'leading_whitespace': [],
            'trailing_whitespace': [],
            'very_short_annotations': [],
            'very_long_annotations': [],
            'empty_annotations': []
        }
        
        for idx, row in df.iterrows():
            text = row['annotation_text']
            
            if not text:
                issues['empty_annotations'].append(row.to_dict())
                continue
            
            if text[-1] in '.,!?;:':
                issues['trailing_punctuation'].append(row.to_dict())
            
            if text[0].isspace():
                issues['leading_whitespace'].append(row.to_dict())
            
            if text[-1].isspace():
                issues['trailing_whitespace'].append(row.to_dict())
            
            if len(text) == 1:
                issues['very_short_annotations'].append(row.to_dict())
            
            if len(text) > 100:
                issues['very_long_annotations'].append(row.to_dict())
        
        return {
            'issue_counts': {k: len(v) for k, v in issues.items()},
            'sample_issues': {k: v[:5] for k, v in issues.items()}
        }
    
    def label_cooccurrence(self, annotations: List[Dict]) -> Dict:
        """Analyze which labels frequently appear together"""
        df = pd.DataFrame(annotations)
        
        cooccurrence = defaultdict(Counter)
        
        for msg_id in df['message_id'].unique():
            labels = df[df['message_id'] == msg_id]['annotation_label'].tolist()
            
            for i, label1 in enumerate(labels):
                for label2 in labels[i+1:]:
                    if label1 != label2:
                        key = tuple(sorted([label1, label2]))
                        cooccurrence[key[0]][key[1]] += 1
        
        # Convert to sorted list
        cooccurrence_list = []
        for label1, counter in cooccurrence.items():
            for label2, count in counter.most_common(10):
                cooccurrence_list.append({
                    'label1': label1,
                    'label2': label2,
                    'count': count
                })
        
        return sorted(cooccurrence_list, key=lambda x: x['count'], reverse=True)[:20]
    
    def generate_plots(self, annotations: List[Dict], output_dir: str):
        """Generate visualization plots"""
        df = pd.DataFrame(annotations)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Label distribution
        plt.figure(figsize=(12, 6))
        label_counts = df['annotation_label'].value_counts()
        plt.bar(range(len(label_counts)), label_counts.values)
        plt.xticks(range(len(label_counts)), label_counts.index, rotation=45, ha='right')
        plt.ylabel('Count')
        plt.title(f'{self.annotation_type.capitalize()} Label Distribution')
        plt.tight_layout()
        plt.savefig(output_path / f'{self.annotation_type}_label_distribution.png', dpi=300)
        plt.close()
        
        # 2. Annotations per message
        plt.figure(figsize=(10, 6))
        anns_per_msg = df.groupby('message_id').size()
        plt.hist(anns_per_msg, bins=20, edgecolor='black')
        plt.xlabel('Number of Annotations')
        plt.ylabel('Number of Messages')
        plt.title(f'Distribution of Annotations per Message ({self.annotation_type})')
        plt.tight_layout()
        plt.savefig(output_path / f'{self.annotation_type}_annotations_per_message.png', dpi=300)
        plt.close()
        
        # 3. Annotation length distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['annotation_length'], bins=30, edgecolor='black')
        plt.xlabel('Annotation Length (characters)')
        plt.ylabel('Count')
        plt.title(f'Annotation Length Distribution ({self.annotation_type})')
        plt.tight_layout()
        plt.savefig(output_path / f'{self.annotation_type}_length_distribution.png', dpi=300)
        plt.close()
        
        # 4. Annotation length by label
        plt.figure(figsize=(12, 6))
        df.boxplot(column='annotation_length', by='annotation_label', rot=45)
        plt.ylabel('Annotation Length (characters)')
        plt.title(f'Annotation Length by Label ({self.annotation_type})')
        plt.suptitle('')  # Remove default title
        plt.tight_layout()
        plt.savefig(output_path / f'{self.annotation_type}_length_by_label.png', dpi=300)
        plt.close()
        
        print(f"\n[STAT] Plots saved to {output_path}/")
    
    def run_full_analysis(self, output_dir: str = "data/eda"):
        """Run complete EDA pipeline"""
        print(f"\n{'='*80}")
        print(f"EDA for {self.annotation_type.upper()} Annotations")
        print(f"{'='*80}\n")
        
        annotations = self.extract_annotations()
        
        if not annotations:
            print("[WARNING]  No annotations found!")
            return
        
        # Basic statistics
        print("[STAT] Basic Statistics:")
        stats = self.basic_statistics(annotations)
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Label distribution
        print(f"\n[STAT] {self.annotation_type.capitalize()} Label Distribution:")
        label_dist = self.label_distribution(annotations)
        for label, count in sorted(label_dist['counts'].items(), key=lambda x: x[1], reverse=True):
            pct = label_dist['percentages'][label]
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        # Overlap analysis
        print("\n[STAT] Overlap Analysis:")
        overlaps = self.annotation_overlap_analysis(annotations)
        print(f"  Total overlaps: {overlaps['total_overlaps']}")
        print(f"  Overlap rate: {overlaps['overlap_rate']*100:.2f}%")
        if overlaps['overlaps']:
            print(f"  Sample overlaps:")
            for overlap in overlaps['overlaps'][:5]:
                print(f"    - {overlap['ann1_label']}: '{overlap['ann1_text']}' overlaps with {overlap['ann2_label']}: '{overlap['ann2_text']}'")
        
        # Message length analysis
        print("\n[STAT] Message Length Analysis:")
        msg_lengths = self.message_length_analysis(annotations)
        for key, value in msg_lengths.items():
            print(f"  {key}: {value:.2f}")
        
        # Quality checks
        print("\n[STAT] Quality Checks:")
        quality = self.annotation_quality_checks(annotations)
        for issue_type, count in quality['issue_counts'].items():
            if count > 0:
                print(f"  {issue_type}: {count}")
        
        # Co-occurrence
        print(f"\n[STAT] Label Co-occurrence (Top 10):")
        cooccurrence = self.label_cooccurrence(annotations)
        for item in cooccurrence[:10]:
            print(f"  {item['label1']} + {item['label2']}: {item['count']} times")
        
        # Generate plots
        self.generate_plots(annotations, output_dir)
        
        # Save full report
        report = {
            'annotation_type': self.annotation_type,
            'file': self.annotation_file,
            'basic_statistics': stats,
            'label_distribution': label_dist,
            'overlap_analysis': {
                'total_overlaps': overlaps['total_overlaps'],
                'overlap_rate': overlaps['overlap_rate']
            },
            'message_length_analysis': msg_lengths,
            'quality_checks': quality['issue_counts'],
            'label_cooccurrence': cooccurrence
        }
        
        report_file = Path(output_dir) / f'{self.annotation_type}_eda_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[OK] Full report saved to: {report_file}")
        
        return report


def compare_annotations(entity_file: str, claim_file: str, output_dir: str = "data/eda"):
    """Compare entity-based and claim-based annotations"""
    print(f"\n{'='*80}")
    print("COMPARISON: Entity vs Claim Annotations")
    print(f"{'='*80}\n")
    
    entity_eda = AnnotationEDA(entity_file, "entity")
    claim_eda = AnnotationEDA(claim_file, "claim")
    
    entity_anns = entity_eda.extract_annotations()
    claim_anns = claim_eda.extract_annotations()
    
    entity_df = pd.DataFrame(entity_anns)
    claim_df = pd.DataFrame(claim_anns)
    
    comparison = {
        'entity_annotations': {
            'total': len(entity_df),
            'unique_labels': entity_df['annotation_label'].nunique(),
            'avg_per_message': len(entity_df) / entity_df['message_id'].nunique(),
            'avg_length': entity_df['annotation_length'].mean()
        },
        'claim_annotations': {
            'total': len(claim_df),
            'unique_labels': claim_df['annotation_label'].nunique(),
            'avg_per_message': len(claim_df) / claim_df['message_id'].nunique(),
            'avg_length': claim_df['annotation_length'].mean()
        }
    }
    
    print("[STAT] Entity Annotations:")
    for key, value in comparison['entity_annotations'].items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n[STAT] Claim Annotations:")
    for key, value in comparison['claim_annotations'].items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save comparison
    comparison_file = Path(output_dir) / 'entity_vs_claim_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n[OK] Comparison saved to: {comparison_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive EDA for annotations")
    parser.add_argument("--entity", help="Path to entity annotations JSON")
    parser.add_argument("--claim", help="Path to claim annotations JSON")
    parser.add_argument("--compare", action="store_true", help="Compare entity and claim annotations")
    parser.add_argument("--output", default="data/eda", help="Output directory for results")
    
    args = parser.parse_args()
    
    if args.entity:
        eda = AnnotationEDA(args.entity, "entity")
        eda.run_full_analysis(args.output)
    
    if args.claim:
        eda = AnnotationEDA(args.claim, "claim")
        eda.run_full_analysis(args.output)
    
    if args.compare and args.entity and args.claim:
        compare_annotations(args.entity, args.claim, args.output)
    
    if not args.entity and not args.claim:
        print("Usage:")
        print("  python scripts/eda_comprehensive.py --entity data/annotations/entity_annotations.json")
        print("  python scripts/eda_comprehensive.py --claim data/annotations/claim_annotations.json")
        print("  python scripts/eda_comprehensive.py --entity <entity_file> --claim <claim_file> --compare")

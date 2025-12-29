#!/usr/bin/env python3
"""
Model Performance Evaluation and Analytics

Analyzes test results from trained NER models and generates:
- Comprehensive performance metrics
- Per-claim-type analysis
- Error analysis and breakdown
- Visualizations (confusion matrix, performance charts)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def normalize_claim_type(claim_type):
    """Normalize rare claim types to OTHER_CLAIM (same as training)"""
    RARE_CLAIMS = ['SECURITY_CLAIM', 'IDENTITY_CLAIM', 'CREDENTIALS_CLAIM', 'LEGAL_CLAIM', 'SOCIAL_CLAIM']
    return 'OTHER_CLAIM' if claim_type in RARE_CLAIMS else claim_type


def load_results(results_file):
    """Load test results from JSON and normalize claim types"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Normalize claim types in ground truth to match training
    for result in results:
        for claim in result['ground_truth']['claims']:
            claim['type'] = normalize_claim_type(claim['type'])
    
    return results


def calculate_claim_type_metrics(results):
    """Calculate precision, recall, F1 per claim type"""
    type_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for entry in results:
        gt_claims = entry['ground_truth']['claims']
        pred_claims = entry['prediction']['claims']
        
        # Track matched predictions
        matched_preds = set()
        matched_gts = set()
        
        # Match predictions to ground truth
        for pred_idx, pred in enumerate(pred_claims):
            matched = False
            for gt_idx, gt in enumerate(gt_claims):
                if gt_idx in matched_gts:
                    continue
                
                # Check overlap
                overlap_start = max(pred['start'], gt['start'])
                overlap_end = min(pred['end'], gt['end'])
                
                if overlap_end > overlap_start:
                    # Has overlap
                    if pred['type'] == gt['type']:
                        type_stats[pred['type']]['tp'] += 1
                        matched_preds.add(pred_idx)
                        matched_gts.add(gt_idx)
                        matched = True
                        break
            
            if not matched:
                type_stats[pred['type']]['fp'] += 1
        
        # Unmatched ground truth are false negatives
        for gt_idx, gt in enumerate(gt_claims):
            if gt_idx not in matched_gts:
                type_stats[gt['type']]['fn'] += 1
    
    # Calculate metrics
    metrics = {}
    for claim_type, stats in type_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[claim_type] = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3),
            'support': tp + fn
        }
    
    return metrics


def analyze_errors(results):
    """Analyze common error patterns"""
    errors = {
        'type_confusion': defaultdict(lambda: defaultdict(int)),
        'boundary_errors': [],
        'false_positives': [],
        'false_negatives': []
    }
    
    for entry in results:
        text = entry['text']
        gt_claims = entry['ground_truth']['claims']
        pred_claims = entry['prediction']['claims']
        
        matched_preds = set()
        matched_gts = set()
        
        # Find matches and errors
        for pred in pred_claims:
            matched = False
            for gt_idx, gt in enumerate(gt_claims):
                if gt_idx in matched_gts:
                    continue
                
                overlap_start = max(pred['start'], gt['start'])
                overlap_end = min(pred['end'], gt['end'])
                
                if overlap_end > overlap_start:
                    if pred['type'] != gt['type']:
                        # Type confusion
                        errors['type_confusion'][gt['type']][pred['type']] += 1
                    else:
                        # Boundary error if not exact match
                        if pred['start'] != gt['start'] or pred['end'] != gt['end']:
                            errors['boundary_errors'].append({
                                'text': text,
                                'gt_text': gt['text'],
                                'pred_text': pred['text'],
                                'type': pred['type']
                            })
                    matched_gts.add(gt_idx)
                    matched = True
                    break
            
            if not matched:
                # False positive
                errors['false_positives'].append({
                    'text': text[:100],
                    'claim': pred['text'],
                    'type': pred['type'],
                    'confidence': pred['confidence']
                })
        
        # False negatives
        for gt_idx, gt in enumerate(gt_claims):
            if gt_idx not in matched_gts:
                errors['false_negatives'].append({
                    'text': text[:100],
                    'claim': gt['text'],
                    'type': gt['type']
                })
    
    return errors


def generate_statistics(results):
    """Generate overall statistics"""
    stats = {
        'total_messages': len(results),
        'total_gt_claims': sum(len(r['ground_truth']['claims']) for r in results),
        'total_pred_claims': sum(len(r['prediction']['claims']) for r in results),
        'messages_with_claims_gt': sum(1 for r in results if len(r['ground_truth']['claims']) > 0),
        'messages_with_claims_pred': sum(1 for r in results if len(r['prediction']['claims']) > 0),
        'perfect_matches': sum(1 for r in results if r['evaluation']['claim_count_match']),
    }
    
    # Calculate overall metrics
    # Check if matched_claims exists (new format) or calculate it (old format)
    if 'matched_claims' in results[0]['evaluation']:
        total_matched = sum(r['evaluation']['matched_claims'] for r in results)
    else:
        # Old format - recalculate matched claims
        total_matched = 0
        for r in results:
            gt_claims = r['ground_truth']['claims']
            pred_claims = r['prediction']['claims']
            for pred in pred_claims:
                for gt in gt_claims:
                    overlap_start = max(pred['start'], gt['start'])
                    overlap_end = min(pred['end'], gt['end'])
                    if overlap_end > overlap_start and pred['type'] == gt['type']:
                        total_matched += 1
                        break
    
    precision = total_matched / stats['total_pred_claims'] if stats['total_pred_claims'] > 0 else 0
    recall = total_matched / stats['total_gt_claims'] if stats['total_gt_claims'] > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    stats['overall_precision'] = round(precision, 3)
    stats['overall_recall'] = round(recall, 3)
    stats['overall_f1'] = round(f1, 3)
    stats['avg_claims_per_message_gt'] = round(stats['total_gt_claims'] / stats['total_messages'], 2)
    stats['avg_claims_per_message_pred'] = round(stats['total_pred_claims'] / stats['total_messages'], 2)
    
    return stats


def plot_confusion_matrix(type_metrics, output_dir):
    """Plot confusion matrix for claim types"""
    claim_types = sorted(type_metrics.keys())
    
    # Create confusion-like matrix (simplified)
    data = []
    for ct in claim_types:
        data.append([
            type_metrics[ct]['true_positives'],
            type_metrics[ct]['false_positives'],
            type_metrics[ct]['false_negatives']
        ])
    
    df = pd.DataFrame(data, index=claim_types, columns=['TP', 'FP', 'FN'])
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
    plt.title('Claim Type Performance Matrix')
    plt.ylabel('Claim Type')
    plt.xlabel('Metric')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")


def plot_performance_metrics(type_metrics, output_dir):
    """Plot precision, recall, F1 for each claim type"""
    claim_types = sorted(type_metrics.keys())
    
    precision = [type_metrics[ct]['precision'] for ct in claim_types]
    recall = [type_metrics[ct]['recall'] for ct in claim_types]
    f1 = [type_metrics[ct]['f1_score'] for ct in claim_types]
    
    x = np.arange(len(claim_types))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    ax.bar(x, recall, width, label='Recall', color='#3498db')
    ax.bar(x + width, f1, width, label='F1 Score', color='#e74c3c')
    
    ax.set_xlabel('Claim Type')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Claim Type')
    ax.set_xticks(x)
    ax.set_xticklabels(claim_types, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'performance_by_type.png'}")


def plot_support_distribution(type_metrics, output_dir):
    """Plot support (number of instances) for each claim type"""
    claim_types = sorted(type_metrics.keys())
    support = [type_metrics[ct]['support'] for ct in claim_types]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(claim_types, support, color='#9b59b6')
    plt.xlabel('Claim Type')
    plt.ylabel('Number of Instances')
    plt.title('Claim Type Distribution in Test Set')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'support_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'support_distribution.png'}")


def plot_confidence_distribution(results, output_dir):
    """Plot confidence score distribution"""
    confidences = []
    for entry in results:
        for claim in entry['prediction']['claims']:
            confidences.append(claim['confidence'])
    
    if not confidences:
        print("No predictions to plot confidence distribution")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'confidence_distribution.png'}")


def generate_report(stats, type_metrics, errors, output_dir):
    """Generate comprehensive text report"""
    report_path = output_dir / 'evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total test messages: {stats['total_messages']}\n")
        f.write(f"Total ground truth claims: {stats['total_gt_claims']}\n")
        f.write(f"Total predicted claims: {stats['total_pred_claims']}\n")
        f.write(f"Perfect matches (exact count): {stats['perfect_matches']} ({stats['perfect_matches']/stats['total_messages']*100:.1f}%)\n")
        f.write(f"\nMessages with claims (GT): {stats['messages_with_claims_gt']}\n")
        f.write(f"Messages with claims (Pred): {stats['messages_with_claims_pred']}\n")
        f.write(f"Avg claims per message (GT): {stats['avg_claims_per_message_gt']}\n")
        f.write(f"Avg claims per message (Pred): {stats['avg_claims_per_message_pred']}\n")
        f.write("\n")
        
        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Precision: {stats['overall_precision']:.3f}\n")
        f.write(f"Recall:    {stats['overall_recall']:.3f}\n")
        f.write(f"F1 Score:  {stats['overall_f1']:.3f}\n")
        f.write("\n")
        
        # Per-type metrics
        f.write("PER-TYPE METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Type':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}\n")
        f.write("-"*80 + "\n")
        
        for claim_type in sorted(type_metrics.keys()):
            m = type_metrics[claim_type]
            f.write(f"{claim_type:<25} {m['precision']:<12.3f} {m['recall']:<12.3f} "
                   f"{m['f1_score']:<12.3f} {m['support']:<10}\n")
        f.write("\n")
        
        # Error analysis
        f.write("ERROR ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total false positives: {len(errors['false_positives'])}\n")
        f.write(f"Total false negatives: {len(errors['false_negatives'])}\n")
        f.write(f"Total boundary errors: {len(errors['boundary_errors'])}\n")
        f.write("\n")
        
        # Type confusion
        if errors['type_confusion']:
            f.write("TYPE CONFUSION MATRIX\n")
            f.write("-"*80 + "\n")
            f.write("(GT Type -> Predicted Type : Count)\n\n")
            for gt_type, pred_types in sorted(errors['type_confusion'].items()):
                for pred_type, count in sorted(pred_types.items(), key=lambda x: -x[1]):
                    f.write(f"  {gt_type} -> {pred_type}: {count}\n")
            f.write("\n")
        
        # Sample errors
        f.write("SAMPLE FALSE POSITIVES (First 10)\n")
        f.write("-"*80 + "\n")
        for i, fp in enumerate(errors['false_positives'][:10], 1):
            f.write(f"{i}. Type: {fp['type']}, Confidence: {fp['confidence']:.3f}\n")
            f.write(f"   Claim: \"{fp['claim']}\"\n")
            f.write(f"   Context: \"{fp['text']}...\"\n\n")
        
        f.write("SAMPLE FALSE NEGATIVES (First 10)\n")
        f.write("-"*80 + "\n")
        for i, fn in enumerate(errors['false_negatives'][:10], 1):
            f.write(f"{i}. Type: {fn['type']}\n")
            f.write(f"   Missed claim: \"{fn['claim']}\"\n")
            f.write(f"   Context: \"{fn['text']}...\"\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance from test results')
    parser.add_argument('results_file', type=str, help='Path to test_results_detailed.json')
    parser.add_argument('--output-dir', type=str, default='evaluation_output',
                       help='Output directory for reports and visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*80)
    
    # Load results
    print(f"\nLoading results from: {args.results_file}")
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} test examples")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    stats = generate_statistics(results)
    type_metrics = calculate_claim_type_metrics(results)
    errors = analyze_errors(results)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Overall Precision: {stats['overall_precision']:.3f}")
    print(f"Overall Recall:    {stats['overall_recall']:.3f}")
    print(f"Overall F1 Score:  {stats['overall_f1']:.3f}")
    print(f"\nTotal Claims (GT): {stats['total_gt_claims']}")
    print(f"Total Claims (Pred): {stats['total_pred_claims']}")
    print(f"Perfect Matches: {stats['perfect_matches']}/{stats['total_messages']} ({stats['perfect_matches']/stats['total_messages']*100:.1f}%)")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(type_metrics, output_dir)
    plot_performance_metrics(type_metrics, output_dir)
    plot_support_distribution(type_metrics, output_dir)
    plot_confidence_distribution(results, output_dir)
    
    # Generate report
    print("\nGenerating detailed report...")
    generate_report(stats, type_metrics, errors, output_dir)
    
    # Save metrics as JSON
    metrics_output = {
        'statistics': stats,
        'per_type_metrics': type_metrics,
        'error_counts': {
            'false_positives': len(errors['false_positives']),
            'false_negatives': len(errors['false_negatives']),
            'boundary_errors': len(errors['boundary_errors']),
            'type_confusions': sum(sum(v.values()) for v in errors['type_confusion'].values())
        }
    }
    
    metrics_path = output_dir / 'metrics_summary.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"Saved: {metrics_path}")
    
    print("\n" + "="*80)
    print(f"Evaluation complete. Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

"""
Advanced EDA Visualizations for SMS Claim Extraction

This script generates deep-dive visualizations including:
- Word clouds for entities/claims
- Frequency distributions
- Co-occurrence networks
- Message pattern analysis
- Character-level statistics
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
import re
from itertools import combinations

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("[WARNING] wordcloud not installed. Word cloud visualizations will be skipped.")
    print("          Install with: pip install wordcloud")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class AdvancedAnnotationViz:
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
        self.annotations = self._extract_annotations()
    
    def _extract_annotations(self):
        """Extract annotations with full context"""
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
            
            msg_anns = []
            for ann in results:
                if ann.get('type') == 'labels':
                    value = ann.get('value', {})
                    msg_anns.append({
                        'text': value.get('text', ''),
                        'label': value.get('labels', ['UNKNOWN'])[0],
                        'start': value.get('start', 0),
                        'end': value.get('end', 0),
                    })
            
            annotations.append({
                'message_id': msg_id,
                'message_text': text,
                'message_label': label,
                'annotations': msg_anns,
                'num_annotations': len(msg_anns)
            })
        
        return annotations
    
    def generate_wordcloud(self, output_dir: str, by_label: bool = True):
        """Generate word clouds for annotations"""
        if not WORDCLOUD_AVAILABLE:
            print("Skipping word clouds (wordcloud not installed)")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if by_label:
            # Word cloud per label
            label_texts = defaultdict(list)
            for msg in self.annotations:
                for ann in msg['annotations']:
                    label_texts[ann['label']].append(ann['text'].lower())
            
            # Generate one word cloud per label
            for label, texts in label_texts.items():
                if len(texts) < 5:  # Skip labels with too few samples
                    continue
                
                # Combine all texts
                combined_text = ' '.join(texts)
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=1200, 
                    height=600,
                    background_color='white',
                    colormap='viridis',
                    max_words=100,
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate(combined_text)
                
                plt.figure(figsize=(14, 7))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'{self.annotation_type.capitalize()}: {label} Word Cloud', 
                         fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_path / f'{self.annotation_type}_wordcloud_{label}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Generated word cloud for {label}")
        
        # Overall word cloud
        all_texts = []
        for msg in self.annotations:
            for ann in msg['annotations']:
                all_texts.append(ann['text'].lower())
        
        if all_texts:
            combined_text = ' '.join(all_texts)
            wordcloud = WordCloud(
                width=1600, 
                height=800,
                background_color='white',
                colormap='plasma',
                max_words=150,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(combined_text)
            
            plt.figure(figsize=(16, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'All {self.annotation_type.capitalize()} Annotations - Word Cloud', 
                     fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / f'{self.annotation_type}_wordcloud_all.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Generated overall word cloud")
    
    def analyze_top_terms(self, output_dir: str, top_n: int = 30):
        """Analyze most frequent terms by label"""
        output_path = Path(output_dir)
        
        label_terms = defaultdict(Counter)
        
        for msg in self.annotations:
            for ann in msg['annotations']:
                label_terms[ann['label']][ann['text'].lower()] += 1
        
        # Plot top terms for each label
        for label, term_counts in label_terms.items():
            top_terms = term_counts.most_common(top_n)
            
            if len(top_terms) < 3:  # Skip if too few
                continue
            
            terms, counts = zip(*top_terms)
            
            plt.figure(figsize=(12, max(8, len(terms) * 0.3)))
            plt.barh(range(len(terms)), counts, color='steelblue')
            plt.yticks(range(len(terms)), terms)
            plt.xlabel('Frequency', fontsize=12)
            plt.title(f'Top {len(terms)} Most Frequent {label} Terms', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_path / f'{self.annotation_type}_top_terms_{label}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Generated top terms for {label}")
    
    def analyze_character_patterns(self, output_dir: str):
        """Analyze character-level patterns"""
        output_path = Path(output_dir)
        
        patterns = {
            'has_numbers': [],
            'has_special_chars': [],
            'all_caps': [],
            'mixed_case': [],
            'length': []
        }
        
        for msg in self.annotations:
            for ann in msg['annotations']:
                text = ann['text']
                patterns['has_numbers'].append(bool(re.search(r'\d', text)))
                patterns['has_special_chars'].append(bool(re.search(r'[^a-zA-Z0-9\s]', text)))
                patterns['all_caps'].append(text.isupper() and text.isalpha())
                patterns['mixed_case'].append(any(c.isupper() for c in text) and any(c.islower() for c in text))
                patterns['length'].append(len(text))
        
        # Plot patterns
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Pattern frequencies
        pattern_names = ['Has Numbers', 'Has Special Chars', 'All Caps', 'Mixed Case']
        pattern_keys = ['has_numbers', 'has_special_chars', 'all_caps', 'mixed_case']
        
        for idx, (name, key) in enumerate(zip(pattern_names, pattern_keys)):
            ax = axes[idx // 2, idx % 2]
            counts = Counter(patterns[key])
            labels = ['False', 'True']
            values = [counts[False], counts[True]]
            
            ax.bar(labels, values, color=['coral', 'lightgreen'])
            ax.set_ylabel('Count')
            ax.set_title(name, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        # Length distribution
        ax = axes[0, 2]
        ax.hist(patterns['length'], bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel('Annotation Length (characters)')
        ax.set_ylabel('Frequency')
        ax.set_title('Length Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Length by label
        ax = axes[1, 2]
        label_lengths = defaultdict(list)
        for msg in self.annotations:
            for ann in msg['annotations']:
                label_lengths[ann['label']].append(len(ann['text']))
        
        ax.boxplot([lengths for lengths in label_lengths.values()], 
                   labels=list(label_lengths.keys()), 
                   vert=True)
        ax.set_ylabel('Length (characters)')
        ax.set_title('Length Distribution by Label', fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path / f'{self.annotation_type}_character_patterns.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Generated character pattern analysis")
    
    def analyze_cooccurrence_network(self, output_dir: str, min_cooccurrence: int = 5):
        """Analyze and visualize label co-occurrence as network"""
        output_path = Path(output_dir)
        
        # Build co-occurrence matrix
        labels = set()
        for msg in self.annotations:
            for ann in msg['annotations']:
                labels.add(ann['label'])
        
        labels = sorted(list(labels))
        n_labels = len(labels)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        cooccur_matrix = np.zeros((n_labels, n_labels))
        
        for msg in self.annotations:
            msg_labels = [ann['label'] for ann in msg['annotations']]
            for label1, label2 in combinations(set(msg_labels), 2):
                idx1 = label_to_idx[label1]
                idx2 = label_to_idx[label2]
                cooccur_matrix[idx1, idx2] += 1
                cooccur_matrix[idx2, idx1] += 1
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = cooccur_matrix < min_cooccurrence
        sns.heatmap(cooccur_matrix, 
                   mask=mask,
                   xticklabels=labels, 
                   yticklabels=labels,
                   annot=True, 
                   fmt='.0f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Co-occurrence Count'})
        plt.title(f'{self.annotation_type.capitalize()} Label Co-occurrence Matrix\n(Min {min_cooccurrence} occurrences)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / f'{self.annotation_type}_cooccurrence_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Generated co-occurrence heatmap")
    
    def analyze_message_patterns(self, output_dir: str):
        """Analyze patterns in messages"""
        output_path = Path(output_dir)
        
        # Collect message statistics
        msg_lengths = []
        num_annotations = []
        annotation_density = []  # annotations per 100 chars
        
        for msg in self.annotations:
            msg_len = len(msg['message_text'])
            num_ann = msg['num_annotations']
            
            msg_lengths.append(msg_len)
            num_annotations.append(num_ann)
            annotation_density.append((num_ann / msg_len * 100) if msg_len > 0 else 0)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Message length distribution
        axes[0, 0].hist(msg_lengths, bins=30, color='lightblue', edgecolor='black')
        axes[0, 0].set_xlabel('Message Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Message Length Distribution', fontweight='bold')
        axes[0, 0].axvline(np.mean(msg_lengths), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(msg_lengths):.0f}')
        axes[0, 0].legend()
        
        # Annotations per message
        axes[0, 1].hist(num_annotations, bins=range(0, max(num_annotations)+2), 
                       color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('Number of Annotations')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Annotations per Message', fontweight='bold')
        axes[0, 1].axvline(np.mean(num_annotations), color='red', linestyle='--',
                          label=f'Mean: {np.mean(num_annotations):.1f}')
        axes[0, 1].legend()
        
        # Annotation density
        axes[1, 0].hist(annotation_density, bins=30, color='lightgreen', edgecolor='black')
        axes[1, 0].set_xlabel('Annotation Density (per 100 chars)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Annotation Density Distribution', fontweight='bold')
        
        # Scatter: length vs annotations
        axes[1, 1].scatter(msg_lengths, num_annotations, alpha=0.5, color='purple')
        axes[1, 1].set_xlabel('Message Length (characters)')
        axes[1, 1].set_ylabel('Number of Annotations')
        axes[1, 1].set_title('Message Length vs Annotations', fontweight='bold')
        
        # Add correlation
        corr = np.corrcoef(msg_lengths, num_annotations)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                       transform=axes[1, 1].transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path / f'{self.annotation_type}_message_patterns.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Generated message pattern analysis")
    
    def analyze_ngrams(self, output_dir: str, n: int = 2, top_k: int = 20):
        """Analyze n-grams in annotations"""
        output_path = Path(output_dir)
        
        from collections import Counter
        
        # Collect n-grams by label
        label_ngrams = defaultdict(Counter)
        
        for msg in self.annotations:
            for ann in msg['annotations']:
                text = ann['text'].lower()
                words = re.findall(r'\b\w+\b', text)
                
                if len(words) >= n:
                    ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
                    label_ngrams[ann['label']].update(ngrams)
        
        # Plot top n-grams for each label
        for label, ngram_counts in label_ngrams.items():
            top_ngrams = ngram_counts.most_common(top_k)
            
            if len(top_ngrams) < 3:
                continue
            
            ngrams, counts = zip(*top_ngrams)
            
            plt.figure(figsize=(12, max(8, len(ngrams) * 0.3)))
            plt.barh(range(len(ngrams)), counts, color='teal')
            plt.yticks(range(len(ngrams)), ngrams)
            plt.xlabel('Frequency', fontsize=12)
            plt.title(f'Top {len(ngrams)} {n}-grams in {label}', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_path / f'{self.annotation_type}_{n}grams_{label}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Generated {n}-gram analysis for {label}")
    
    def generate_all_visualizations(self, output_dir: str = "data/eda/viz"):
        """Generate all visualizations"""
        print(f"\nGenerating advanced visualizations for {self.annotation_type} annotations...")
        print("=" * 80)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n[1/7] Generating word clouds...")
        if WORDCLOUD_AVAILABLE:
            self.generate_wordcloud(output_dir)
        else:
            print("  Skipped (wordcloud not installed)")
        
        print("\n[2/7] Analyzing top terms...")
        self.analyze_top_terms(output_dir, top_n=25)
        
        print("\n[3/7] Analyzing character patterns...")
        self.analyze_character_patterns(output_dir)
        
        print("\n[4/7] Analyzing co-occurrence network...")
        self.analyze_cooccurrence_network(output_dir, min_cooccurrence=3)
        
        print("\n[5/7] Analyzing message patterns...")
        self.analyze_message_patterns(output_dir)
        
        print("\n[6/7] Analyzing bigrams...")
        self.analyze_ngrams(output_dir, n=2, top_k=20)
        
        print("\n[7/7] Analyzing trigrams...")
        self.analyze_ngrams(output_dir, n=3, top_k=15)
        
        print("\n" + "=" * 80)
        print(f"[OK] All visualizations saved to: {output_dir}/")
        print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced EDA visualizations")
    parser.add_argument("--entity", help="Path to entity annotations JSON")
    parser.add_argument("--claim", help="Path to claim annotations JSON")
    parser.add_argument("--output", default="data/eda/viz", help="Output directory")
    
    args = parser.parse_args()
    
    if args.entity:
        print("\n" + "="*80)
        print("ENTITY ANNOTATIONS - ADVANCED VISUALIZATIONS")
        print("="*80)
        viz = AdvancedAnnotationViz(args.entity, "entity")
        viz.generate_all_visualizations(f"{args.output}/entity")
    
    if args.claim:
        print("\n" + "="*80)
        print("CLAIM ANNOTATIONS - ADVANCED VISUALIZATIONS")
        print("="*80)
        viz = AdvancedAnnotationViz(args.claim, "claim")
        viz.generate_all_visualizations(f"{args.output}/claim")
    
    if not args.entity and not args.claim:
        print("Usage:")
        print("  python scripts/eda_visualizations.py --entity data/annotations/entity_annotations.json")
        print("  python scripts/eda_visualizations.py --claim data/annotations/claim_annotations.json")
        print("  python scripts/eda_visualizations.py --entity <file> --claim <file>")


if __name__ == "__main__":
    main()

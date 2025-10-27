import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "annotations" / "annotated_complete.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = []
    entities = []
    
    for item in data:
        msg_id = item['data']['message_id']
        text = item['data']['text']
        
        msg_entities = []
        for annotation in item.get('annotations', []):
            for result in annotation.get('result', []):
                if result.get('type') == 'labels':
                    value = result['value']
                    entity_info = {
                        'message_id': msg_id,
                        'label': value['labels'][0],
                        'text': value['text'],
                        'start': value['start'],
                        'end': value['end'],
                        'length': len(value['text'])
                    }
                    entities.append(entity_info)
                    msg_entities.append(value['labels'][0])
        
        messages.append({
            'message_id': msg_id,
            'text': text,
            'length': len(text),
            'word_count': len(text.split()),
            'entity_count': len(msg_entities),
            'entities': msg_entities,
            'unique_entity_types': len(set(msg_entities))
        })
    
    return pd.DataFrame(messages), pd.DataFrame(entities)

def analyze_entity_distribution(entities_df):
    plt.figure(figsize=(12, 6))
    
    entity_counts = entities_df['label'].value_counts()
    colors = sns.color_palette("husl", len(entity_counts))
    
    plt.subplot(1, 2, 1)
    entity_counts.plot(kind='bar', color=colors)
    plt.title('Entity Type Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Entity Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.pie(entity_counts.values, labels=entity_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Entity Type Percentage', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'entity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return entity_counts

def analyze_message_length(messages_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(messages_df['length'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Message Length Distribution (Characters)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Character Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(messages_df['length'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].axvline(messages_df['length'].median(), color='green', linestyle='--', linewidth=2, label='Median')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].hist(messages_df['word_count'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Message Length Distribution (Words)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(messages_df['word_count'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].axvline(messages_df['word_count'].median(), color='green', linestyle='--', linewidth=2, label='Median')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].hist(messages_df['entity_count'], bins=range(0, messages_df['entity_count'].max()+2), 
                    color='mediumseagreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Entities per Message', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Entities')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(messages_df['entity_count'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].hist(messages_df['unique_entity_types'], bins=range(0, messages_df['unique_entity_types'].max()+2),
                    color='mediumpurple', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Unique Entity Types per Message', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Unique Types')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(messages_df['unique_entity_types'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'message_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_entity_length(entities_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    entity_types = entities_df['label'].unique()
    length_by_type = [entities_df[entities_df['label'] == et]['length'].values for et in entity_types]
    
    bp = axes[0].boxplot(length_by_type, labels=entity_types, patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(entity_types))):
        patch.set_facecolor(color)
    axes[0].set_title('Entity Length Distribution by Type', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Entity Type')
    axes[0].set_ylabel('Character Length')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    avg_lengths = entities_df.groupby('label')['length'].mean().sort_values(ascending=False)
    colors = sns.color_palette("husl", len(avg_lengths))
    avg_lengths.plot(kind='barh', ax=axes[1], color=colors)
    axes[1].set_title('Average Entity Length by Type', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Average Character Length')
    axes[1].set_ylabel('Entity Type')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'entity_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_entity_cooccurrence(messages_df):
    entity_pairs = defaultdict(int)
    
    for entities in messages_df['entities']:
        unique_entities = list(set(entities))
        for i in range(len(unique_entities)):
            for j in range(i+1, len(unique_entities)):
                pair = tuple(sorted([unique_entities[i], unique_entities[j]]))
                entity_pairs[pair] += 1
    
    if entity_pairs:
        top_pairs = sorted(entity_pairs.items(), key=lambda x: -x[1])[:15]
        
        plt.figure(figsize=(14, 8))
        pairs_labels = [f"{p[0]} + {p[1]}" for p, _ in top_pairs]
        pairs_counts = [c for _, c in top_pairs]
        
        colors = sns.color_palette("viridis", len(pairs_labels))
        plt.barh(range(len(pairs_labels)), pairs_counts, color=colors)
        plt.yticks(range(len(pairs_labels)), pairs_labels, fontsize=10)
        plt.xlabel('Co-occurrence Count', fontsize=12)
        plt.title('Top 15 Entity Type Co-occurrences', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'entity_cooccurrence.png', dpi=300, bbox_inches='tight')
        plt.close()

def analyze_entity_patterns(entities_df):
    top_entities = {}
    for entity_type in entities_df['label'].unique():
        type_entities = entities_df[entities_df['label'] == entity_type]['text'].value_counts().head(10)
        top_entities[entity_type] = type_entities
    
    n_types = len(top_entities)
    n_cols = 3
    n_rows = (n_types + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten() if n_types > 1 else [axes]
    
    for idx, (entity_type, top_items) in enumerate(sorted(top_entities.items())):
        if idx < len(axes):
            if len(top_items) > 0:
                colors = sns.color_palette("husl", len(top_items))
                top_items.plot(kind='barh', ax=axes[idx], color=colors)
                axes[idx].set_title(f'Top 10 {entity_type} Entities', fontsize=11, fontweight='bold')
                axes[idx].set_xlabel('Frequency', fontsize=10)
                axes[idx].set_ylabel('')
                axes[idx].tick_params(axis='y', labelsize=8)
                axes[idx].grid(axis='x', alpha=0.3)
    
    for idx in range(len(top_entities), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_entities_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_stats(messages_df, entities_df):
    stats = {
        'dataset_overview': {
            'total_messages': len(messages_df),
            'total_entities': len(entities_df),
            'avg_entities_per_message': entities_df.groupby('message_id').size().mean(),
            'entity_types': len(entities_df['label'].unique())
        },
        'message_statistics': {
            'avg_char_length': messages_df['length'].mean(),
            'median_char_length': messages_df['length'].median(),
            'min_char_length': messages_df['length'].min(),
            'max_char_length': messages_df['length'].max(),
            'avg_word_count': messages_df['word_count'].mean(),
            'median_word_count': messages_df['word_count'].median()
        },
        'entity_statistics': {
            'avg_entity_length': entities_df['length'].mean(),
            'median_entity_length': entities_df['length'].median(),
            'min_entity_length': entities_df['length'].min(),
            'max_entity_length': entities_df['length'].max()
        },
        'entity_distribution': entities_df['label'].value_counts().to_dict(),
        'entity_percentages': (entities_df['label'].value_counts() / len(entities_df) * 100).round(2).to_dict()
    }
    
    with open(OUTPUT_DIR / 'summary_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def print_summary(stats):
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS - ANNOTATION SUMMARY")
    print("="*70)
    
    print("\nDATASET OVERVIEW:")
    print(f"  Total Messages: {stats['dataset_overview']['total_messages']}")
    print(f"  Total Entities: {stats['dataset_overview']['total_entities']}")
    print(f"  Avg Entities/Message: {stats['dataset_overview']['avg_entities_per_message']:.2f}")
    print(f"  Entity Types: {stats['dataset_overview']['entity_types']}")
    
    print("\nMESSAGE STATISTICS:")
    print(f"  Avg Length: {stats['message_statistics']['avg_char_length']:.1f} chars")
    print(f"  Median Length: {stats['message_statistics']['median_char_length']:.1f} chars")
    print(f"  Range: {stats['message_statistics']['min_char_length']} - {stats['message_statistics']['max_char_length']} chars")
    print(f"  Avg Words: {stats['message_statistics']['avg_word_count']:.1f}")
    print(f"  Median Words: {stats['message_statistics']['median_word_count']:.1f}")
    
    print("\nENTITY STATISTICS:")
    print(f"  Avg Length: {stats['entity_statistics']['avg_entity_length']:.1f} chars")
    print(f"  Median Length: {stats['entity_statistics']['median_entity_length']:.1f} chars")
    print(f"  Range: {stats['entity_statistics']['min_entity_length']} - {stats['entity_statistics']['max_entity_length']} chars")
    
    print("\nENTITY TYPE DISTRIBUTION:")
    for entity_type, count in sorted(stats['entity_distribution'].items(), key=lambda x: -x[1]):
        pct = stats['entity_percentages'][entity_type]
        print(f"  {entity_type:20s}: {count:4d} ({pct:5.2f}%)")
    
    print("\n" + "="*70)
    print(f"\nVisualizations saved to: {OUTPUT_DIR}/")
    print("  - entity_distribution.png")
    print("  - message_statistics.png")
    print("  - entity_length_analysis.png")
    print("  - entity_cooccurrence.png")
    print("  - top_entities_by_type.png")
    print(f"\nSummary statistics saved to: {OUTPUT_DIR}/summary_statistics.json")
    print("="*70 + "\n")

def main():
    print("\nLoading annotation data...")
    messages_df, entities_df = load_data()
    
    print(f"Loaded {len(messages_df)} messages with {len(entities_df)} entities")
    
    print("\nGenerating visualizations...")
    print("  - Entity distribution analysis...")
    analyze_entity_distribution(entities_df)
    
    print("  - Message statistics analysis...")
    analyze_message_length(messages_df)
    
    print("  - Entity length analysis...")
    analyze_entity_length(entities_df)
    
    print("  - Entity co-occurrence analysis...")
    analyze_entity_cooccurrence(messages_df)
    
    print("  - Top entities by type analysis...")
    analyze_entity_patterns(entities_df)
    
    print("\nGenerating summary statistics...")
    stats = generate_summary_stats(messages_df, entities_df)
    
    print_summary(stats)

if __name__ == "__main__":
    main()

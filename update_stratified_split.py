#!/usr/bin/env python3
"""
Update all notebooks to use STRATIFIED split (balanced ham/smish in train and test)
"""

import json
from pathlib import Path

def update_to_stratified_split(notebook_path):
    """Update split to be stratified by ham/smish"""
    print(f"Updating {notebook_path.name}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Update split code for approaches 1, 2, 3 (with examples)
            if '# Split data' in source and 'train_examples, test_examples = train_test_split(examples' in source:
                new_source = """# Split data with stratification (balanced ham/smish)
# First, determine which examples are HAM vs SMISH
example_labels = []
for ex in examples:
    # SMISH = has claims, HAM = no claims
    is_smish = len(ex.get('claim_spans', [])) > 0 or len(ex.get('entity_spans', [])) > 0
    example_labels.append('SMISH' if is_smish else 'HAM')

# Stratified split to maintain ham/smish balance
train_examples, test_examples = train_test_split(
    examples, 
    test_size=0.20, 
    random_state=42,
    stratify=example_labels
)

print(f"Dataset split:")
print(f"  Train: {len(train_examples)} examples")
# No validation set - using test set for evaluation during training
print(f"  Test:  {len(test_examples)} examples")

# Count ham/smish distribution
from collections import Counter
train_labels = []
test_labels = []
for ex in train_examples:
    is_smish = len(ex.get('claim_spans', [])) > 0 or len(ex.get('entity_spans', [])) > 0
    train_labels.append('SMISH' if is_smish else 'HAM')
for ex in test_examples:
    is_smish = len(ex.get('claim_spans', [])) > 0 or len(ex.get('entity_spans', [])) > 0
    test_labels.append('SMISH' if is_smish else 'HAM')

train_dist = Counter(train_labels)
test_dist = Counter(test_labels)
print(f"\\nTrain distribution:")
print(f"  HAM:   {train_dist['HAM']} ({train_dist['HAM']/len(train_examples)*100:.1f}%)")
print(f"  SMISH: {train_dist['SMISH']} ({train_dist['SMISH']/len(train_examples)*100:.1f}%)")
print(f"\\nTest distribution:")
print(f"  HAM:   {test_dist['HAM']} ({test_dist['HAM']/len(test_examples)*100:.1f}%)")
print(f"  SMISH: {test_dist['SMISH']} ({test_dist['SMISH']/len(test_examples)*100:.1f}%)")

# Count labels
all_labels = []
for ex in train_examples:
    all_labels.extend(ex['labels'])"""
                
                # Keep the rest of the original cell
                if 'label_counts = Counter(all_labels)' in source:
                    rest = source[source.index('label_counts = Counter(all_labels)'):]
                    new_source = new_source + '\n\n' + rest
                
                cell['source'] = new_source.split('\n')
            
            # Update split code for approach 4 (classification)
            elif '# Split data' in source and 'train_texts, test_texts, train_labels, test_labels = train_test_split' in source:
                # Already has stratify parameter, just verify it's there
                if 'stratify=labels' not in source:
                    new_source = source.replace(
                        'train_texts, test_texts, train_labels, test_labels = train_test_split(\n    texts, labels, test_size=0.20, random_state=42\n)',
                        'train_texts, test_texts, train_labels, test_labels = train_test_split(\n    texts, labels, test_size=0.20, random_state=42, stratify=labels\n)'
                    )
                    cell['source'] = new_source.split('\n')
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"  Updated to stratified split")

def main():
    notebooks = [
        'approach1_entity_first_ner.ipynb',
        'approach2_claim_phrase_ner.ipynb',
        'approach3_hybrid_claim_llm.ipynb',
        'approach4_contrastive_classification.ipynb'
    ]
    
    base_dir = Path(__file__).parent
    
    print("="*60)
    print("UPDATING TO STRATIFIED SPLIT")
    print("="*60)
    print("Changes:")
    print("  - Balance HAM/SMISH in train and test sets")
    print("  - Use stratify parameter in train_test_split")
    print("  - Show distribution statistics")
    print("="*60)
    print()
    
    for notebook_file in notebooks:
        notebook_path = base_dir / notebook_file
        if notebook_path.exists():
            update_to_stratified_split(notebook_path)
        else:
            print(f"  Skipped {notebook_file} (not found)")
    
    print()
    print("="*60)
    print("STRATIFIED SPLIT ENABLED!")
    print("="*60)
    print("Train and test sets now have balanced HAM/SMISH distribution")
    print("="*60)

if __name__ == '__main__':
    main()

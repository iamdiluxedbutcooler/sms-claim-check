#!/usr/bin/env python3
"""
Update all notebooks to use 80-20 train-test split (no validation set)
"""

import json
from pathlib import Path

def update_split_in_notebook(notebook_path):
    """Update data split to 80-20"""
    print(f"Updating {notebook_path.name}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Update the split code
            if 'train_test_split' in source and 'test_size=0.15' in source:
                # Replace split logic
                new_source = source.replace(
                    'train_examples, test_examples = train_test_split(examples, test_size=0.15, random_state=42)\ntrain_examples, val_examples = train_test_split(train_examples, test_size=0.176, random_state=42)  # 0.15/0.85',
                    'train_examples, test_examples = train_test_split(examples, test_size=0.20, random_state=42)'
                ).replace(
                    'train_examples, test_examples = train_test_split(examples, test_size=0.15, random_state=42)\ntrain_examples, val_examples = train_test_split(train_examples, test_size=0.176, random_state=42)',
                    'train_examples, test_examples = train_test_split(examples, test_size=0.20, random_state=42)'
                ).replace(
                    'print(f"  Val:   {len(val_examples)} examples")',
                    '# No validation set - using test set for evaluation during training'
                ).replace(
                    'val_dataset',
                    'test_dataset'
                ).replace(
                    'val_examples',
                    'test_examples'
                ).replace(
                    'val_tokenized',
                    'test_tokenized'
                ).replace(
                    'val_encodings',
                    'test_encodings'
                ).replace(
                    'val_texts',
                    'test_texts'
                ).replace(
                    'val_labels',
                    'test_labels'
                )
                
                # For classification approach
                new_source = new_source.replace(
                    'train_texts, test_texts, train_labels, test_labels = train_test_split(\n    texts, labels, test_size=0.15, random_state=42, stratify=labels\n)\n\ntrain_texts, val_texts, train_labels, val_labels = train_test_split(\n    train_texts, train_labels, test_size=0.176, random_state=42, stratify=train_labels\n)',
                    'train_texts, test_texts, train_labels, test_labels = train_test_split(\n    texts, labels, test_size=0.20, random_state=42, stratify=labels\n)'
                )
                
                cell['source'] = new_source.split('\n')
            
            # Update tokenization calls
            if 'val_tokenized = tokenize_and_align_labels(val_examples)' in source:
                new_source = source.replace(
                    'val_tokenized = tokenize_and_align_labels(val_examples)\ntest_tokenized = tokenize_and_align_labels(test_examples)',
                    'test_tokenized = tokenize_and_align_labels(test_examples)'
                ).replace(
                    'val_encodings = tokenizer(val_texts',
                    'test_encodings = tokenizer(test_texts'
                )
                cell['source'] = new_source.split('\n')
            
            # Update dataset creation
            if 'val_dataset = NERDataset' in source or 'val_dataset = SMSDataset' in source:
                new_source = source.replace(
                    'val_dataset = NERDataset(val_tokenized)',
                    '# Using test set for evaluation during training'
                ).replace(
                    'val_dataset = SMSDataset(val_encodings, val_labels)',
                    '# Using test set for evaluation during training'
                )
                cell['source'] = new_source.split('\n')
            
            # Update trainer initialization
            if 'eval_dataset=val_dataset' in source:
                new_source = source.replace(
                    'eval_dataset=val_dataset',
                    'eval_dataset=test_dataset  # Using test set for evaluation'
                )
                cell['source'] = new_source.split('\n')
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"  Updated to 80-20 split")

def main():
    notebooks = [
        'approach1_entity_first_ner.ipynb',
        'approach2_claim_phrase_ner.ipynb',
        'approach3_hybrid_claim_llm.ipynb',
        'approach4_contrastive_classification.ipynb'
    ]
    
    base_dir = Path(__file__).parent
    
    print("="*60)
    print("UPDATING TO 80-20 TRAIN-TEST SPLIT")
    print("="*60)
    print("Changes:")
    print("  - Remove validation set")
    print("  - 80% train, 20% test")
    print("  - Use test set for evaluation during training")
    print("="*60)
    print()
    
    for notebook_file in notebooks:
        notebook_path = base_dir / notebook_file
        if notebook_path.exists():
            update_split_in_notebook(notebook_path)
        else:
            print(f"  Skipped {notebook_file} (not found)")
    
    print()
    print("="*60)
    print("SPLIT UPDATED!")
    print("="*60)
    print("New split: 80% train (1600) / 20% test (400)")
    print("Test set used for evaluation during training")
    print("="*60)

if __name__ == '__main__':
    main()

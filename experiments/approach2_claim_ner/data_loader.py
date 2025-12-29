"""
Data loader for Claim-Phrase NER
Converts claim_annotations_2000.json to token classification format
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

@dataclass
class ClaimSpan:
    """Represents a claim span in text"""
    text: str
    start: int
    end: int
    label: str

@dataclass
class NERExample:
    """NER training example"""
    text: str
    tokens: List[str]
    labels: List[str]
    claim_spans: List[ClaimSpan]

class ClaimNERDataset(Dataset):
    """PyTorch Dataset for Claim NER"""
    
    def __init__(self, examples: List[NERExample], tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Define label mappings
        self.claim_types = [
            'IDENTITY_CLAIM', 'DELIVERY_CLAIM', 'FINANCIAL_CLAIM',
            'ACCOUNT_CLAIM', 'URGENCY_CLAIM', 'ACTION_CLAIM',
            'VERIFICATION_CLAIM', 'SECURITY_CLAIM', 'REWARD_CLAIM',
            'LEGAL_CLAIM', 'SOCIAL_CLAIM', 'CREDENTIALS_CLAIM'
        ]
        
        # Create BIO labels
        self.labels = ['O']  # Outside
        for claim_type in self.claim_types:
            self.labels.append(f'B-{claim_type}')
            self.labels.append(f'I-{claim_type}')
        
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize with offsets for subword alignment
        encoding = self.tokenizer(
            example.text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Align labels with subword tokens
        labels = self._align_labels_with_tokens(
            example.text,
            encoding.offset_mapping[0],
            example.labels
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _align_labels_with_tokens(self, text, offset_mapping, word_labels):
        """Align word-level labels with subword tokens"""
        labels = []
        
        # Get character-level label mapping
        char_labels = ['O'] * len(text)
        word_idx = 0
        char_idx = 0
        
        # Split text into words and assign labels
        words = text.split()
        for word, label in zip(words, word_labels):
            # Find word position in text
            word_start = text.find(word, char_idx)
            if word_start != -1:
                word_end = word_start + len(word)
                for i in range(word_start, word_end):
                    char_labels[i] = label
                char_idx = word_end
        
        # Align with subword tokens
        for offset in offset_mapping:
            start, end = offset
            
            if start == 0 and end == 0:
                # Special token (CLS, SEP, PAD)
                labels.append(-100)
            else:
                # Use label of first character in span
                if start < len(char_labels):
                    labels.append(self.label2id.get(char_labels[start], 0))
                else:
                    labels.append(0)  # O label
        
        return labels

def convert_annotations_to_ner(annotation_file: Path) -> List[NERExample]:
    """
    Convert claim annotations to NER examples with BIO tagging
    
    Args:
        annotation_file: Path to claim_annotations_2000.json
        
    Returns:
        List of NERExample objects
    """
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    
    for entry in data:
        text = entry['data']['text']
        
        # Skip if no annotations
        if not entry.get('annotations') or len(entry['annotations']) == 0:
            continue
        
        annotations = entry['annotations'][0]
        
        # Skip if no results
        if 'result' not in annotations or len(annotations['result']) == 0:
            # Add as all-O example (ham message)
            tokens = text.split()
            labels = ['O'] * len(tokens)
            examples.append(NERExample(
                text=text,
                tokens=tokens,
                labels=labels,
                claim_spans=[]
            ))
            continue
        
        # Extract claim spans
        claim_spans = []
        for result in annotations['result']:
            value = result.get('value', {})
            labels_list = value.get('labels', [])
            
            if labels_list:
                claim_span = ClaimSpan(
                    text=value.get('text', ''),
                    start=value.get('start', 0),
                    end=value.get('end', 0),
                    label=labels_list[0]  # Take first label
                )
                claim_spans.append(claim_span)
        
        # Sort spans by start position
        claim_spans.sort(key=lambda x: x.start)
        
        # Convert to BIO format
        tokens, labels = _text_to_bio(text, claim_spans)
        
        examples.append(NERExample(
            text=text,
            tokens=tokens,
            labels=labels,
            claim_spans=claim_spans
        ))
    
    return examples

def _text_to_bio(text: str, claim_spans: List[ClaimSpan]) -> Tuple[List[str], List[str]]:
    """
    Convert text and claim spans to BIO format
    
    This is a simplified version that works at word level.
    For better accuracy, you might want character-level alignment.
    """
    tokens = text.split()
    labels = ['O'] * len(tokens)
    
    # Track character position
    char_pos = 0
    
    for token_idx, token in enumerate(tokens):
        # Find token position in text
        token_start = text.find(token, char_pos)
        token_end = token_start + len(token)
        char_pos = token_end
        
        # Check if token overlaps with any claim span
        for span in claim_spans:
            # Token overlaps with span
            if not (token_end <= span.start or token_start >= span.end):
                # Determine if B- or I- tag
                # If token starts the span or is first token in span, use B-
                if token_start <= span.start < token_end:
                    labels[token_idx] = f'B-{span.label}'
                else:
                    labels[token_idx] = f'I-{span.label}'
                break  # Only assign one label per token
    
    return tokens, labels

def load_and_split_data(
    annotation_file: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[List[NERExample], List[NERExample], List[NERExample]]:
    """
    Load data and split into train/val/test sets
    
    Args:
        annotation_file: Path to claim_annotations_2000.json
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        (train_examples, val_examples, test_examples)
    """
    examples = convert_annotations_to_ner(annotation_file)
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        examples,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    # Second split: train vs val
    val_proportion = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_proportion,
        random_state=random_state,
        shuffle=True
    )
    
    return train, val, test

def create_dataloaders(
    train_examples: List[NERExample],
    val_examples: List[NERExample],
    test_examples: List[NERExample],
    tokenizer,
    batch_size: int = 16,
    max_length: int = 128
):
    """Create PyTorch DataLoaders for train/val/test"""
    from torch.utils.data import DataLoader
    
    train_dataset = ClaimNERDataset(train_examples, tokenizer, max_length)
    val_dataset = ClaimNERDataset(val_examples, tokenizer, max_length)
    test_dataset = ClaimNERDataset(test_examples, tokenizer, max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader, train_dataset.label2id, train_dataset.id2label

if __name__ == '__main__':
    # Test data loading
    from transformers import AutoTokenizer
    
    print("Testing Claim NER Data Loader...")
    print("="*60)
    
    # Load data
    data_path = Path(__file__).parent.parent.parent / 'data' / 'annotations' / 'claim_annotations_2000.json'
    train, val, test = load_and_split_data(data_path)
    
    print(f"✅ Data loaded:")
    print(f"   Train: {len(train)} examples")
    print(f"   Val:   {len(val)} examples")
    print(f"   Test:  {len(test)} examples")
    
    # Show example
    print(f"\n📝 Example:")
    example = train[0]
    print(f"   Text: {example.text[:100]}...")
    print(f"   Tokens: {example.tokens[:10]}...")
    print(f"   Labels: {example.labels[:10]}...")
    print(f"   Claims: {len(example.claim_spans)} claim spans")
    
    # Test tokenization
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    dataset = ClaimNERDataset(train[:10], tokenizer)
    
    print(f"\n🔤 Tokenization test:")
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Number of labels: {len(dataset.labels)}")
    print(f"   Labels: {dataset.labels}")
    
    sample = dataset[0]
    print(f"\n   Sample batch:")
    print(f"   - input_ids shape: {sample['input_ids'].shape}")
    print(f"   - attention_mask shape: {sample['attention_mask'].shape}")
    print(f"   - labels shape: {sample['labels'].shape}")
    
    print(f"\n✅ Data loader test passed!")

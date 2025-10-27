import json
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import evaluate
from collections import Counter
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "annotations" / "annotated_complete.json"
OUTPUT_DIR = PROJECT_ROOT / "models" / "ner"

ENTITY_LABELS = [
    "O",
    "B-BRAND", "I-BRAND",
    "B-PHONE", "I-PHONE",
    "B-URL", "I-URL",
    "B-ORDER_ID", "I-ORDER_ID",
    "B-AMOUNT", "I-AMOUNT",
    "B-DATE", "I-DATE",
    "B-DEADLINE", "I-DEADLINE",
    "B-ACTION_REQUIRED", "I-ACTION_REQUIRED",
    "B-EMAIL", "I-EMAIL"
]

LABEL2ID = {label: idx for idx, label in enumerate(ENTITY_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

def load_annotations():
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = []
    for item in data:
        msg_id = item['data']['message_id']
        text = item['data']['text']
        
        entities = []
        for annotation in item.get('annotations', []):
            for result in annotation.get('result', []):
                if result.get('type') == 'labels':
                    value = result['value']
                    entities.append({
                        'start': value['start'],
                        'end': value['end'],
                        'label': value['labels'][0]
                    })
        
        messages.append({
            'id': msg_id,
            'text': text,
            'entities': sorted(entities, key=lambda x: x['start'])
        })
    
    return messages

def text_to_bio_tokens(text, entities):
    tokens = []
    labels = []
    
    current_pos = 0
    for char in text:
        if char.strip():
            tokens.append(char)
            current_pos_in_text = len(''.join(tokens)) - 1
            
            label = "O"
            for ent in entities:
                if ent['start'] <= current_pos_in_text < ent['end']:
                    if current_pos_in_text == ent['start']:
                        label = f"B-{ent['label']}"
                    else:
                        label = f"I-{ent['label']}"
                    break
            
            labels.append(label)
        current_pos += 1
    
    words = text.split()
    word_labels = []
    
    word_start = 0
    for word in words:
        word_end = word_start + len(word)
        
        word_label = "O"
        for ent in entities:
            if ent['start'] <= word_start < ent['end']:
                if word_start == ent['start'] or all(
                    labels[i] == "O" for i in range(max(0, word_start-1), word_start) if i < len(labels)
                ):
                    word_label = f"B-{ent['label']}"
                else:
                    word_label = f"I-{ent['label']}"
                break
            elif word_start < ent['end'] and word_end > ent['start']:
                overlap_start = max(word_start, ent['start'])
                if overlap_start == ent['start']:
                    word_label = f"B-{ent['label']}"
                else:
                    word_label = f"I-{ent['label']}"
                break
        
        word_labels.append(word_label)
        word_start = text.find(' ', word_end) + 1
        if word_start == 0:
            break
    
    return words, word_labels

def prepare_dataset(messages, tokenizer):
    examples = []
    
    for msg in messages:
        words, labels = text_to_bio_tokens(msg['text'], msg['entities'])
        examples.append({
            'id': msg['id'],
            'tokens': words,
            'ner_tags': [LABEL2ID.get(label, 0) for label in labels]
        })
    
    return examples

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def compute_metrics(eval_pred, metric):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SMS PHISHING NER MODEL TRAINING")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    print("\nLoading annotations...")
    messages = load_annotations()
    print(f"Loaded {len(messages)} annotated messages")
    
    entity_counts = Counter()
    for msg in messages:
        for ent in msg['entities']:
            entity_counts[ent['label']] += 1
    
    print("\nEntity distribution:")
    for label, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:20s}: {count:4d}")
    
    print("\nSplitting dataset (67% train / 17% val / 16% test)...")
    train_msgs, temp_msgs = train_test_split(messages, test_size=0.33, random_state=args.seed)
    val_msgs, test_msgs = train_test_split(temp_msgs, test_size=0.48, random_state=args.seed)
    
    print(f"  Train: {len(train_msgs)} messages")
    print(f"  Val:   {len(val_msgs)} messages")
    print(f"  Test:  {len(test_msgs)} messages")
    
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print("Preparing datasets...")
    train_examples = prepare_dataset(train_msgs, tokenizer)
    val_examples = prepare_dataset(val_msgs, tokenizer)
    test_examples = prepare_dataset(test_msgs, tokenizer)
    
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    test_dataset = Dataset.from_list(test_examples)
    
    print("Tokenizing and aligning labels...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    print(f"\nLoading model: {args.model}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    metric = evaluate.load("seqeval")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=3,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, metric),
    )
    
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")
    
    trainer.train()
    
    print("\n" + "="*70)
    print("Evaluating on test set...")
    print("="*70 + "\n")
    
    test_results = trainer.evaluate(test_dataset)
    
    print("\nTest Results:")
    print(f"  Precision: {test_results['eval_precision']:.4f}")
    print(f"  Recall:    {test_results['eval_recall']:.4f}")
    print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
    print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
    
    print(f"\nSaving model to {OUTPUT_DIR}/final_model")
    trainer.save_model(str(OUTPUT_DIR / "final_model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final_model"))
    
    with open(OUTPUT_DIR / "label_mapping.json", 'w') as f:
        json.dump({
            'label2id': LABEL2ID,
            'id2label': ID2LABEL
        }, f, indent=2)
    
    with open(OUTPUT_DIR / "test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

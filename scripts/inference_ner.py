import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "ner" / "final_model"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    
    with open(PROJECT_ROOT / "models" / "ner" / "label_mapping.json", 'r') as f:
        label_mapping = json.load(f)
    
    return model, tokenizer, label_mapping['id2label']

def predict_entities(text, model, tokenizer, id2label):
    model.eval()
    
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**tokens)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    word_ids = tokens.word_ids()
    words = text.split()
    
    entities = []
    current_entity = None
    
    for idx, (word_id, pred) in enumerate(zip(word_ids, predictions[0].tolist())):
        if word_id is None:
            continue
        
        label = id2label[str(pred)]
        
        if label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        elif label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]
            current_entity = {
                'type': entity_type,
                'text': words[word_id],
                'start_word': word_id,
                'end_word': word_id
            }
        elif label.startswith("I-"):
            if current_entity and current_entity['type'] == label[2:]:
                current_entity['text'] += ' ' + words[word_id]
                current_entity['end_word'] = word_id
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='SMS text to analyze')
    parser.add_argument('--file', type=str, help='File with SMS messages (one per line)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SMS PHISHING NER - INFERENCE")
    print("="*70)
    
    print("\nLoading model...")
    model, tokenizer, id2label = load_model()
    print("Model loaded successfully!")
    
    if args.text:
        messages = [args.text]
    elif args.file:
        with open(args.file, 'r') as f:
            messages = [line.strip() for line in f if line.strip()]
    else:
        print("\nInteractive mode - Enter SMS messages (empty line to quit):")
        messages = []
        while True:
            msg = input("\nSMS> ")
            if not msg:
                break
            messages.append(msg)
    
    print("\n" + "="*70)
    print("PREDICTIONS")
    print("="*70)
    
    for i, text in enumerate(messages, 1):
        print(f"\n[{i}] Message: {text}")
        
        entities = predict_entities(text, model, tokenizer, id2label)
        
        if entities:
            print(f"    Entities found: {len(entities)}")
            for ent in entities:
                print(f"      - {ent['type']:15s}: {ent['text']}")
        else:
            print("    No entities detected")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()

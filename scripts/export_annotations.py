import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

import sys
sys.path.append(str(PROJECT_ROOT / "config"))
from entity_schema import get_tag_list

def load_label_studio_export(filepath: Path) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_annotations(data: List[Dict]) -> List[Dict]:
    annotated_messages = []
    
    for item in data:
        message_data = item.get('data', {})
        text = message_data.get('text', '')
        message_id = message_data.get('message_id', '')
        label = message_data.get('label', '')
        
        entities = []
        
        for annotation in item.get('annotations', []):
            for result in annotation.get('result', []):
                if result.get('type') == 'labels':
                    value = result.get('value', {})
                    entities.append({
                        'label': value.get('labels', [''])[0],
                        'text': value.get('text'),
                        'start': value.get('start'),
                        'end': value.get('end'),
                    })
        
        entities.sort(key=lambda x: x['start'])
        
        annotated_messages.append({
            'message_id': message_id,
            'text': text,
            'label': label,
            'entities': entities
        })
    
    return annotated_messages

def text_to_tokens(text: str) -> List[str]:
    return text.split()

def convert_to_bio(messages: List[Dict]) -> List[Dict]:
    bio_data = []
    
    for msg in messages:
        text = msg['text']
        entities = msg['entities']
        
        tokens = text_to_tokens(text)
        tags = ['O'] * len(tokens)
        
        char_to_token = {}
        current_pos = 0
        for i, token in enumerate(tokens):
            token_start = text.find(token, current_pos)
            token_end = token_start + len(token)
            for char_pos in range(token_start, token_end):
                char_to_token[char_pos] = i
            current_pos = token_end
        
        for entity in entities:
            entity_label = entity['label']
            start_char = entity['start']
            end_char = entity['end']
            
            start_token = char_to_token.get(start_char)
            end_token = char_to_token.get(end_char - 1)
            
            if start_token is not None and end_token is not None:
                tags[start_token] = f"B-{entity_label}"
                for i in range(start_token + 1, end_token + 1):
                    tags[i] = f"I-{entity_label}"
        
        bio_data.append({
            'message_id': msg['message_id'],
            'tokens': tokens,
            'tags': tags,
            'label': msg['label']
        })
    
    return bio_data

def save_conll_format(bio_data: List[Dict], output_path: Path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for msg in bio_data:
            for token, tag in zip(msg['tokens'], msg['tags']):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")
    
    print(f"Saved CoNLL format to: {output_path}")

def save_json_format(bio_data: List[Dict], output_path: Path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bio_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved JSON format to: {output_path}")

def print_stats(bio_data: List[Dict]):
    total_tokens = sum(len(msg['tokens']) for msg in bio_data)
    total_entities = sum(sum(1 for tag in msg['tags'] if tag.startswith('B-')) for msg in bio_data)
    
    entity_counts = defaultdict(int)
    for msg in bio_data:
        for tag in msg['tags']:
            if tag.startswith('B-'):
                entity_type = tag[2:]
                entity_counts[entity_type] += 1
    
    print("\n" + "="*60)
    print("EXPORT STATISTICS")
    print("="*60)
    print(f"\nOverview:")
    print(f"  Total messages: {len(bio_data)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total entities: {total_entities}")
    print(f"  Avg tokens per message: {total_tokens / len(bio_data):.2f}")
    print(f"  Avg entities per message: {total_entities / len(bio_data):.2f}")
    
    print(f"\nEntity Distribution:")
    for entity_type, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_entities * 100) if total_entities > 0 else 0
        print(f"  {entity_type:20s}: {count:4d} ({percentage:5.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Export Label Studio annotations")
    parser.add_argument('--input', type=str)
    parser.add_argument('--format', type=str, choices=['conll', 'json', 'both'], default='both')
    parser.add_argument('--output-prefix', type=str, default='annotations')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("EXPORT LABEL STUDIO ANNOTATIONS")
    print("="*60)
    
    if args.input:
        input_file = Path(args.input)
    else:
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        exports = list(ANNOTATIONS_DIR.glob("*.json"))
        
        if not exports:
            print("\nNo annotation files found in data/annotations/")
            print("\nExport from Label Studio first.")
            return
        
        input_file = max(exports, key=lambda p: p.stat().st_mtime)
        print(f"\nUsing: {input_file.name}")
    
    if not input_file.exists():
        print(f"\nFile not found: {input_file}")
        return
    
    print(f"\nLoading annotations...")
    data = load_label_studio_export(input_file)
    messages = extract_annotations(data)
    print(f"Loaded {len(messages)} annotated messages")
    
    print(f"\nConverting to BIO format...")
    bio_data = convert_to_bio(messages)
    
    print_stats(bio_data)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.format in ['conll', 'both']:
        conll_path = OUTPUT_DIR / f"{args.output_prefix}.conll"
        save_conll_format(bio_data, conll_path)
    
    if args.format in ['json', 'both']:
        json_path = OUTPUT_DIR / f"{args.output_prefix}.json"
        save_json_format(bio_data, json_path)
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print("\nNext: Begin NER model training")
    print()

if __name__ == "__main__":
    main()

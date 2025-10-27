import json
import argparse
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data"

def convert_to_label_studio(messages: List[Dict], output_path: Path):
    label_studio_data = []
    
    for msg in messages:
        label_studio_data.append({
            "data": {
                "text": msg["text"],
                "message_id": msg["message_id"],
                "label": msg["label"]
            }
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(label_studio_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(messages)} messages to Label Studio format")
    print(f"Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert data to Label Studio format")
    parser.add_argument('--input', type=str, default='annotation_set.json')
    parser.add_argument('--output', type=str, default='label_studio_import.json')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CONVERT TO LABEL STUDIO FORMAT")
    print("="*60)
    
    if args.pilot:
        input_file = PROCESSED_DIR / "pilot_set.json"
        output_file = OUTPUT_DIR / "label_studio_pilot.json"
    else:
        input_file = PROCESSED_DIR / args.input
        output_file = OUTPUT_DIR / args.output
    
    if not input_file.exists():
        print(f"\nError: Input file not found: {input_file}")
        print("\nRun: python scripts/prepare_data.py")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        messages = json.load(f)
    
    print(f"\nLoading: {input_file}")
    print(f"Messages: {len(messages)}")
    
    convert_to_label_studio(messages, output_file)
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Install: pip install label-studio")
    print("2. Start: label-studio start")
    print("3. Import config: config/label_studio_config.xml")
    print(f"4. Import data: {output_file}")
    print()

if __name__ == "__main__":
    main()

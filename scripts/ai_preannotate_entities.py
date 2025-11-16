"""
Entity-Based Pre-annotation using GPT-4o Batch API

This script extracts entities (BRAND, PHONE, URL, ORDER_ID, AMOUNT, DATE, DEADLINE, ACTION_REQUIRED)
from SMS messages for Approach 1 (Entity-First NER) and Approach 3 (Hybrid Entity+LLM).

Entities are concrete and well-defined, making them suitable for training NER models.
"""

import pandas as pd
import json
import os
from typing import List, Dict
from openai import OpenAI
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ENTITY_ANNOTATION_PROMPT = """You are an expert annotator for phishing detection research. Extract entities from SMS messages with EXTREME PRECISION.

**Entity Types:**
1. BRAND - Company/organization names (Amazon, PayPal, USPS, IRS, FedEx, DHL, Netflix, Apple, etc.)
2. PHONE - Phone numbers in ANY format (digits, dashes, spaces, parentheses)
3. URL - Web addresses (http://, https://, www., domains like amazon.com, bit.ly links)
4. ORDER_ID - Tracking numbers, order numbers, confirmation codes, transaction IDs
5. AMOUNT - Monetary values WITH currency symbols (£500, $99.99, €50)
6. DATE - Non-urgent temporal references (Jan 15, 2024-01-15, yesterday, last week)
7. DEADLINE - URGENT time references (within 24h, immediately, now, today, ASAP, expires tonight)
8. ACTION_REQUIRED - Imperative verbs (click, call, verify, confirm, update, reply, text, respond)

**Critical Annotation Rules:**
1. Extract MINIMAL SPANS - only the entity itself, no extra words
2. NO trailing punctuation (!,.,?) unless part of entity (e.g., "Pvt. Ltd.")
3. NO leading/trailing whitespace
4. SEPARATE nested entities (e.g., "Call 123-456-7890" → ACTION_REQUIRED: "Call", PHONE: "123-456-7890")
5. Context determines label:
   - "amazon.com" in "Visit amazon.com" → URL
   - "Amazon" in "Amazon package" → BRAND
6. Phone numbers: INCLUDE if ANY digits appear (even short codes like "85023")
7. URLs: INCLUDE domains without http:// (bit.ly, amazon-security.net)
8. ACTION_REQUIRED: Only verbs that demand user action (not "received", "sent", "has")

**Examples:**
Message: "URGENT: Your Amazon package #ABC123 is delayed. Click here: bit.ly/pkg123 or call 1-800-555-0199 within 24h to resolve. Cost: $15.99"

Correct Annotations:
{{
  "entities": [
    {{"text": "URGENT", "start": 0, "end": 6, "label": "DEADLINE"}},
    {{"text": "Amazon", "start": 13, "end": 19, "label": "BRAND"}},
    {{"text": "#ABC123", "start": 28, "end": 35, "label": "ORDER_ID"}},
    {{"text": "Click", "start": 49, "end": 54, "label": "ACTION_REQUIRED"}},
    {{"text": "bit.ly/pkg123", "start": 61, "end": 74, "label": "URL"}},
    {{"text": "call", "start": 78, "end": 82, "label": "ACTION_REQUIRED"}},
    {{"text": "1-800-555-0199", "start": 83, "end": 97, "label": "PHONE"}},
    {{"text": "within 24h", "start": 98, "end": 108, "label": "DEADLINE"}},
    {{"text": "resolve", "start": 112, "end": 119, "label": "ACTION_REQUIRED"}},
    {{"text": "$15.99", "start": 127, "end": 133, "label": "AMOUNT"}}
  ]
}}

**Your Task:**
Message: {message}

Return ONLY valid JSON with character-level start/end positions (0-indexed).
If no entities found, return: {{"entities": []}}

Output format:
{{
  "entities": [
    {{"text": "exact text", "start": int, "end": int, "label": "ENTITY_TYPE"}}
  ]
}}
"""


def create_entity_batch_input(input_csv: str, output_jsonl: str, limit: int = None):
    """Create batch input file for entity annotation"""
    df = pd.read_csv(input_csv)
    
    # Filter to smishing messages only
    df = df[df['LABEL'].str.lower().isin(['smishing', 'smish'])]
    print(f"Filtered to {len(df)} smishing messages")
    
    if limit:
        df = df.head(limit)
    
    batch_requests = []
    
    for idx, row in df.iterrows():
        message = row['TEXT']
        label = row['LABEL']
        
        request = {
            "custom_id": f"entity_msg_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",  # Using GPT-4o for better quality
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert entity extraction system for NLP research. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": ENTITY_ANNOTATION_PROMPT.format(message=message)
                    }
                ],
                "temperature": 0,
                "max_tokens": 1500,
                "response_format": {"type": "json_object"}
            }
        }
        
        batch_requests.append({
            "request": request,
            "message": message,
            "label": label,
            "msg_id": idx
        })
    
    # Write batch input file
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in batch_requests:
            f.write(json.dumps(item['request']) + '\n')
    
    # Write metadata
    metadata_file = output_jsonl.replace('.jsonl', '_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump([{
            "msg_id": item['msg_id'], 
            "message": item['message'], 
            "label": item['label']
        } for item in batch_requests], f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Created entity batch input file: {output_jsonl}")
    print(f"[OK] Created metadata file: {metadata_file}")
    print(f"[STAT] Total requests: {len(batch_requests)}")
    
    return output_jsonl, metadata_file


def submit_batch(client: OpenAI, input_file_path: str):
    """Submit batch job to OpenAI"""
    print(f"\n[UPLOAD] Uploading batch file: {input_file_path}")
    
    with open(input_file_path, 'rb') as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
    
    print(f"[OK] File uploaded: {batch_input_file.id}")
    
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "SMS phishing entity extraction - GPT-4o"}
    )
    
    print(f"\n[SUCCESS] Batch created successfully!")
    print(f"[INFO] Batch ID: {batch.id}")
    print(f"[STAT] Status: {batch.status}")
    print(f"\n[TIP] Check status with:")
    print(f"   python scripts/ai_preannotate_entities.py --check-status {batch.id}")
    
    return batch.id


def check_batch_status(client: OpenAI, batch_id: str):
    """Check batch job status"""
    batch = client.batches.retrieve(batch_id)
    
    print(f"\n[INFO] Batch ID: {batch.id}")
    print(f"[STAT] Status: {batch.status}")
    print(f"[DATE] Created: {batch.created_at}")
    print(f"[PROGRESS] Progress: {batch.request_counts.completed}/{batch.request_counts.total} completed")
    print(f"[ERROR] Failed: {batch.request_counts.failed}")
    
    if batch.status == "completed":
        print(f"\n[OK] Batch completed!")
        print(f"[DATA] Output file ID: {batch.output_file_id}")
        print(f"\n[TIP] Download results with:")
        print(f"   python scripts/ai_preannotate_entities.py --download {batch.id}")
    elif batch.status == "failed":
        print(f"\n[ERROR] Batch failed!")
        print(f"[DATA] Error file ID: {batch.error_file_id}")
    else:
        print(f"\n[WAIT] Batch still processing. Check again later.")
    
    return batch


def download_and_convert_results(client: OpenAI, batch_id: str, metadata_file: str, output_json: str):
    """Download batch results and convert to Label Studio format"""
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        print(f"[ERROR] Batch not completed yet. Status: {batch.status}")
        return
    
    print("\n[DOWNLOAD] Downloading results...")
    result_content = client.files.content(batch.output_file_id)
    result_lines = result_content.text.strip().split('\n')
    
    # Parse results
    results = {}
    errors = []
    
    for line in result_lines:
        result = json.loads(line)
        custom_id = result['custom_id']
        msg_idx = int(custom_id.split('_')[-1])
        
        try:
            response_content = result['response']['body']['choices'][0]['message']['content']
            entities = json.loads(response_content).get('entities', [])
            results[msg_idx] = entities
        except Exception as e:
            print(f"[WARNING]  Error processing {custom_id}: {e}")
            results[msg_idx] = []
            errors.append({"msg_id": msg_idx, "error": str(e)})
    
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Convert to Label Studio format
    label_studio_data = []
    
    for item in metadata:
        msg_id = item['msg_id']
        message = item['message']
        label = item['label']
        entities = results.get(msg_id, [])
        
        annotations = []
        for ent in entities:
            if not all(k in ent for k in ["start", "end", "text", "label"]):
                print(f"[WARNING]  Skipping malformed entity in msg {msg_id}: {ent}")
                continue
            
            annotations.append({
                "value": {
                    "start": ent["start"],
                    "end": ent["end"],
                    "text": ent["text"],
                    "labels": [ent["label"]]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "prediction"
            })
        
        label_studio_data.append({
            "data": {
                "text": message,
                "message_id": msg_id,
                "label": label
            },
            "predictions": [{
                "model_version": "gpt4o-entity-batch-v1",
                "result": annotations
            }]
        })
    
    # Save Label Studio format
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(label_studio_data, f, indent=2, ensure_ascii=False)
    
    # Save errors if any
    if errors:
        error_file = output_json.replace('.json', '_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2)
        print(f"[WARNING]  Saved {len(errors)} errors to: {error_file}")
    
    print(f"\n[OK] Conversion complete!")
    print(f"[DATA] Output file: {output_json}")
    print(f"[STAT] Total messages: {len(label_studio_data)}")
    print(f"[PROGRESS] Success rate: {(len(results) - len(errors))/len(results)*100:.1f}%")
    print(f"\n[TIP] Next steps:")
    print(f"   1. Import {output_json} into Label Studio")
    print(f"   2. Review and correct annotations")
    print(f"   3. Export final annotations for training")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Entity-based pre-annotation using GPT-4o Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create and submit batch job
  python scripts/ai_preannotate_entities.py --submit
  
  # Check batch status
  python scripts/ai_preannotate_entities.py --check-status batch_xxx
  
  # Download results
  python scripts/ai_preannotate_entities.py --download batch_xxx
        """
    )
    
    parser.add_argument("--input", default="data/raw/mendeley.csv", 
                        help="Input CSV file with SMS messages")
    parser.add_argument("--limit", type=int, 
                        help="Limit number of messages (for testing)")
    parser.add_argument("--submit", action="store_true", 
                        help="Create and submit batch job")
    parser.add_argument("--check-status", metavar="BATCH_ID",
                        help="Check status of batch job")
    parser.add_argument("--download", metavar="BATCH_ID",
                        help="Download and convert results")
    parser.add_argument("--metadata", default="data/annotations/entity_batch_metadata.json",
                        help="Metadata file path")
    parser.add_argument("--output", default="data/annotations/entity_annotations.json",
                        help="Output Label Studio JSON file")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] Error: OPENAI_API_KEY not found in environment")
        print("[TIP] Set it in .env file or export OPENAI_API_KEY=your_key")
        exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Create output directory
    os.makedirs("data/annotations", exist_ok=True)
    
    if args.submit:
        batch_input_file = "data/annotations/entity_batch_input.jsonl"
        create_entity_batch_input(args.input, batch_input_file, args.limit)
        batch_id = submit_batch(client, batch_input_file)
        
        # Save batch ID for reference
        with open("data/annotations/entity_batch_id.txt", "w") as f:
            f.write(batch_id)
    
    elif args.check_status:
        check_batch_status(client, args.check_status)
    
    elif args.download:
        download_and_convert_results(client, args.download, args.metadata, args.output)
    
    else:
        parser.print_help()

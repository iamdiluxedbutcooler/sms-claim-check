import pandas as pd
import json
import os
from typing import List, Dict
from openai import OpenAI
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ENTITY_TYPES = [
    "BRAND", "PHONE", "URL", "ORDER_ID", 
    "AMOUNT", "DATE", "DEADLINE", "ACTION_REQUIRED"
]

ANNOTATION_PROMPT = """You are annotating SMS messages for phishing detection research. Extract entities from the text following these rules:

Entity Types:
- BRAND: Company/organization names (Amazon, PayPal, IRS, USPS, etc.)
- PHONE: Phone numbers in any format
- URL: Web addresses, shortened links, domains
- ORDER_ID: Order numbers, tracking IDs, confirmation codes
- AMOUNT: Monetary values with currency symbols
- DATE: Temporal references without urgency
- DEADLINE: Time references with urgency (within 24h, immediately, now, etc.)
- ACTION_REQUIRED: Imperative verbs (click, call, verify, confirm, etc.)

Rules:
1. Annotate minimal spans only
2. Exclude trailing punctuation
3. Only mark verifiable entities
4. Separate nested entities
5. Context determines label (amazon.com = URL if directive, BRAND if reference)

Output JSON format example:
{{
  "entities": [
    {{"text": "Amazon", "start": 5, "end": 11, "label": "BRAND"}},
    {{"text": "Click here", "start": 20, "end": 30, "label": "ACTION_REQUIRED"}}
  ]
}}

Message to annotate:
{message}

Return only valid JSON with character-level start/end positions. If no entities found, return: {{"entities": []}}"""


def create_batch_input_file(input_csv: str, output_jsonl: str, limit: int = None):
    df = pd.read_csv(input_csv)
    
    df = df[df['label'].str.lower().isin(['smishing', 'smish'])]
    print(f"Filtered to {len(df)} smishing messages")
    
    if limit:
        df = df.head(limit)
    
    batch_requests = []
    
    for idx, row in df.iterrows():
        message = row['text']
        label = row['label']
        
        request = {
            "custom_id": f"msg_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at extracting named entities from SMS messages for NLP research. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": ANNOTATION_PROMPT.format(message=message)
                    }
                ],
                "temperature": 0,
                "max_tokens": 1000,
                "response_format": {"type": "json_object"}
            }
        }
        
        batch_requests.append({
            "request": request,
            "message": message,
            "label": label,
            "msg_id": idx
        })
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in batch_requests:
            f.write(json.dumps(item['request']) + '\n')
    
    metadata_file = output_jsonl.replace('.jsonl', '_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump([{"msg_id": item['msg_id'], "message": item['message'], "label": item['label']} 
                   for item in batch_requests], f, indent=2, ensure_ascii=False)
    
    print(f"Created batch input file: {output_jsonl}")
    print(f"Created metadata file: {metadata_file}")
    print(f"Total requests: {len(batch_requests)}")
    
    return output_jsonl, metadata_file


def submit_batch(client: OpenAI, input_file_path: str):
    print(f"Uploading batch file: {input_file_path}")
    
    with open(input_file_path, 'rb') as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
    
    print(f"File uploaded: {batch_input_file.id}")
    
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "SMS phishing entity extraction"}
    )
    
    print(f"\nBatch created successfully!")
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"\nSave this batch ID to check status later:")
    print(f"python scripts/ai_preannotate.py --check-status {batch.id}")
    
    return batch.id


def check_batch_status(client: OpenAI, batch_id: str):
    batch = client.batches.retrieve(batch_id)
    
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created at: {batch.created_at}")
    print(f"Total requests: {batch.request_counts.total}")
    print(f"Completed: {batch.request_counts.completed}")
    print(f"Failed: {batch.request_counts.failed}")
    
    if batch.status == "completed":
        print(f"\nOutput file ID: {batch.output_file_id}")
        print(f"\nDownload results with:")
        print(f"python scripts/ai_preannotate.py --download {batch.id}")
    elif batch.status == "failed":
        print(f"\nBatch failed. Error file ID: {batch.error_file_id}")
    else:
        print(f"\nBatch still processing. Check again in a few minutes.")
    
    return batch


def download_and_convert_results(client: OpenAI, batch_id: str, metadata_file: str, output_json: str):
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        print(f"Batch not completed yet. Status: {batch.status}")
        return
    
    print("Downloading results...")
    result_content = client.files.content(batch.output_file_id)
    result_lines = result_content.text.strip().split('\n')
    
    results = {}
    for line in result_lines:
        result = json.loads(line)
        custom_id = result['custom_id']
        msg_idx = int(custom_id.split('_')[1])
        
        try:
            response_content = result['response']['body']['choices'][0]['message']['content']
            entities = json.loads(response_content).get('entities', [])
            results[msg_idx] = entities
        except Exception as e:
            print(f"Error processing {custom_id}: {e}")
            results[msg_idx] = []
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    label_studio_data = []
    
    for item in metadata:
        msg_id = item['msg_id']
        message = item['message']
        label = item['label']
        entities = results.get(msg_id, [])
        
        annotations = []
        for ent in entities:
            if not all(k in ent for k in ["start", "end", "text", "label"]):
                print(f"Warning: Skipping malformed entity in msg {msg_id}: {ent}")
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
                "type": "labels"
            })
        
        label_studio_data.append({
            "data": {
                "text": message,
                "message_id": msg_id,
                "label": label
            },
            "predictions": [{
                "model_version": "gpt4-mini-batch-v1",
                "result": annotations
            }]
        })
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(label_studio_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nConversion complete!")
    print(f"Label Studio import file: {output_json}")
    print(f"Total tasks: {len(label_studio_data)}")
    print(f"\nNext steps:")
    print(f"1. Import {output_json} into Label Studio")
    print(f"2. Import config from config/label_studio_config.xml")
    print(f"3. Review and correct AI predictions")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-assisted pre-annotation using OpenAI Batch API")
    parser.add_argument("--input", default="data/processed/train.csv", help="Input CSV file")
    parser.add_argument("--limit", type=int, help="Limit number of messages to process")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--submit", action="store_true", help="Create and submit batch job")
    parser.add_argument("--check-status", help="Check status of batch job by ID")
    parser.add_argument("--download", help="Download and convert results by batch ID")
    parser.add_argument("--metadata", default="data/annotations/batch_metadata.json", help="Metadata file")
    parser.add_argument("--output", default="data/annotations/preannotated.json", help="Final Label Studio JSON")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key required: use --api-key or set OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    os.makedirs("data/annotations", exist_ok=True)
    
    if args.submit:
        batch_input_file = "data/annotations/batch_input.jsonl"
        create_batch_input_file(args.input, batch_input_file, args.limit)
        submit_batch(client, batch_input_file)
    
    elif args.check_status:
        check_batch_status(client, args.check_status)
    
    elif args.download:
        download_and_convert_results(client, args.download, args.metadata, args.output)
    
    else:
        print("Please specify an action:")
        print("  --submit              Create and submit batch job")
        print("  --check-status <id>   Check batch job status")
        print("  --download <id>       Download and convert results")
        print("\nExample workflow:")
        print("  1. python scripts/ai_preannotate.py --submit --limit 100")
        print("  2. python scripts/ai_preannotate.py --check-status batch_xxx")
        print("  3. python scripts/ai_preannotate.py --download batch_xxx")

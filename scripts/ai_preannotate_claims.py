"""
Claim-Based Pre-annotation using GPT-4o Batch API

This script extracts atomic, verifiable claims from SMS messages for:
- Approach 2 (Claim-Phrase NER) 
- Approach 3 (Hybrid Claim+LLM)

Claims are semantic units that can be verified against authoritative sources.
This captures the INTENT and ASSERTIONS in the message, not just entities.
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

CLAIM_ANNOTATION_PROMPT = """You are an expert at extracting verifiable claims from SMS messages for phishing detection research.

**What is a Claim?**
A claim is an ATOMIC, VERIFIABLE assertion that can be fact-checked against authoritative sources.
Claims represent what the message is ASSERTING or ASKING the user to believe/do.

**Claim Types:**

1. **IDENTITY_CLAIM** - Who sent this / Who is this about
   Examples: 
   - "This is from Amazon"
   - "Your PayPal account"
   - "IRS notice"
   - "FedEx delivery notification"

2. **DELIVERY_CLAIM** - Package/delivery related assertions
   Examples:
   - "Your package is delayed"
   - "Delivery failed due to incomplete address"
   - "Item #ABC123 is waiting for pickup"
   - "Shipment is held at customs"

3. **FINANCIAL_CLAIM** - Money/payment related assertions
   Examples:
   - "You owe $50.99"
   - "Refund of £305.96 is pending"
   - "Your account has suspicious activity"
   - "Payment failed for order #12345"
   - "You've won $1000"

4. **ACCOUNT_CLAIM** - Account status assertions
   Examples:
   - "Your account is suspended"
   - "Login credentials expired"
   - "Unusual activity detected"
   - "Account requires verification"

5. **URGENCY_CLAIM** - Time-sensitive assertions
   Examples:
   - "Act within 24 hours"
   - "Expires today"
   - "Immediate action required"
   - "Limited time offer"

6. **ACTION_CLAIM** - What the user must do
   Examples:
   - "Click this link to verify"
   - "Call 1-800-555-0199 immediately"
   - "Reply with your details"
   - "Update your payment information"

**Annotation Rules:**
1. Extract MINIMAL claim phrases - just enough to convey the assertion
2. Claims should be ATOMIC - one verifiable fact per claim
3. Claims should overlap with actual message text (extract spans, don't paraphrase)
4. Include enough context to be verifiable (e.g., "Amazon package delayed" not just "package delayed")
5. DO NOT include URLs, phone numbers, or amounts in claim text (those are entities)
6. Focus on WHAT IS BEING CLAIMED, not the words used

**Example:**

Message: "URGENT: Your Amazon package #ABC123 is delayed. Click here: bit.ly/pkg123 or call 1-800-555-0199 within 24h to resolve. Cost: $15.99"

Correct Claims:
{{
  "claims": [
    {{
      "text": "Your Amazon package #ABC123 is delayed",
      "start": 8,
      "end": 46,
      "label": "DELIVERY_CLAIM",
      "verifiable_components": {{
        "subject": "Amazon package #ABC123",
        "assertion": "is delayed",
        "verification_needed": ["Does tracking #ABC123 exist?", "Is it actually delayed?", "Is it associated with recipient?"]
      }}
    }},
    {{
      "text": "within 24h to resolve",
      "start": 98,
      "end": 119,
      "label": "URGENCY_CLAIM",
      "verifiable_components": {{
        "assertion": "Must act within 24 hours",
        "verification_needed": ["Is this timeframe legitimate for the claimed sender?"]
      }}
    }},
    {{
      "text": "Click here",
      "start": 48,
      "end": 58,
      "label": "ACTION_CLAIM",
      "verifiable_components": {{
        "action_required": "click link",
        "verification_needed": ["Is link from legitimate domain?", "Does Amazon send links in SMS?"]
      }}
    }},
    {{
      "text": "call 1-800-555-0199",
      "start": 78,
      "end": 97,
      "label": "ACTION_CLAIM",
      "verifiable_components": {{
        "action_required": "call phone number",
        "verification_needed": ["Is this Amazon's official number?"]
      }}
    }},
    {{
      "text": "Cost: $15.99",
      "start": 121,
      "end": 133,
      "label": "FINANCIAL_CLAIM",
      "verifiable_components": {{
        "assertion": "Payment of $15.99 required",
        "verification_needed": ["Is this charge legitimate?", "Does user have pending payment?"]
      }}
    }}
  ]
}}

**Your Task:**
Extract ALL verifiable claims from this SMS message. For each claim:
1. Identify the claim type
2. Extract the minimal text span
3. Provide start/end positions (0-indexed characters)
4. List what would need to be verified

Message: {message}

Return ONLY valid JSON.
If no claims found, return: {{"claims": []}}

Output format:
{{
  "claims": [
    {{
      "text": "exact claim text from message",
      "start": int,
      "end": int,
      "label": "CLAIM_TYPE",
      "verifiable_components": {{
        "assertion": "what is being claimed",
        "verification_needed": ["list", "of", "checks"]
      }}
    }}
  ]
}}
"""


def create_claim_batch_input(input_csv: str, output_jsonl: str, limit: int = None):
    """Create batch input file for claim annotation"""
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
            "custom_id": f"claim_msg_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",  # Using GPT-4o for better reasoning
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at identifying verifiable claims in text for fact-checking systems. Always return valid JSON with detailed verification requirements."
                    },
                    {
                        "role": "user",
                        "content": CLAIM_ANNOTATION_PROMPT.format(message=message)
                    }
                ],
                "temperature": 0,
                "max_tokens": 2000,
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
    
    print(f"\n[OK] Created claim batch input file: {output_jsonl}")
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
        metadata={"description": "SMS phishing claim extraction - GPT-4o"}
    )
    
    print(f"\n[SUCCESS] Batch created successfully!")
    print(f"[INFO] Batch ID: {batch.id}")
    print(f"[STAT] Status: {batch.status}")
    print(f"\n[TIP] Check status with:")
    print(f"   python scripts/ai_preannotate_claims.py --check-status {batch.id}")
    
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
        print(f"   python scripts/ai_preannotate_claims.py --download {batch.id}")
    elif batch.status == "failed":
        print(f"\n[ERROR] Batch failed!")
        print(f"[DATA] Error file ID: {batch.error_file_id}")
    else:
        print(f"\n[WAIT] Batch still processing. Check again later.")
    
    return batch


def download_and_convert_results(client: OpenAI, batch_id: str, metadata_file: str, output_json: str):
    """Download batch results and convert to training format"""
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
            claims_data = json.loads(response_content)
            results[msg_idx] = claims_data.get('claims', [])
        except Exception as e:
            print(f"[WARNING]  Error processing {custom_id}: {e}")
            results[msg_idx] = []
            errors.append({"msg_id": msg_idx, "error": str(e)})
    
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Convert to Label Studio format for claim annotation
    label_studio_data = []
    
    for item in metadata:
        msg_id = item['msg_id']
        message = item['message']
        label = item['label']
        claims = results.get(msg_id, [])
        
        annotations = []
        for claim in claims:
            if not all(k in claim for k in ["start", "end", "text", "label"]):
                print(f"[WARNING]  Skipping malformed claim in msg {msg_id}: {claim}")
                continue
            
            # Store verifiable components as metadata
            meta = claim.get('verifiable_components', {})
            
            annotations.append({
                "value": {
                    "start": claim["start"],
                    "end": claim["end"],
                    "text": claim["text"],
                    "labels": [claim["label"]]
                },
                "from_name": "claim_label",
                "to_name": "text",
                "type": "labels",
                "origin": "prediction",
                "meta": meta
            })
        
        label_studio_data.append({
            "data": {
                "text": message,
                "message_id": msg_id,
                "label": label
            },
            "predictions": [{
                "model_version": "gpt4o-claim-batch-v1",
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
    
    # Generate statistics
    claim_type_counts = {}
    total_claims = 0
    
    for msg_id, claims in results.items():
        for claim in claims:
            claim_type = claim.get('label', 'UNKNOWN')
            claim_type_counts[claim_type] = claim_type_counts.get(claim_type, 0) + 1
            total_claims += 1
    
    stats = {
        "total_messages": len(label_studio_data),
        "total_claims": total_claims,
        "avg_claims_per_message": total_claims / len(label_studio_data) if label_studio_data else 0,
        "claim_type_distribution": claim_type_counts,
        "success_rate": (len(results) - len(errors))/len(results)*100 if results else 0
    }
    
    stats_file = output_json.replace('.json', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n[OK] Conversion complete!")
    print(f"[DATA] Output file: {output_json}")
    print(f"[STAT] Statistics: {stats_file}")
    print(f"[PROGRESS] Total messages: {stats['total_messages']}")
    print(f"[PROGRESS] Total claims: {stats['total_claims']}")
    print(f"[PROGRESS] Avg claims/message: {stats['avg_claims_per_message']:.2f}")
    print(f"[PROGRESS] Success rate: {stats['success_rate']:.1f}%")
    print(f"\n[STAT] Claim Type Distribution:")
    for claim_type, count in sorted(claim_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {claim_type}: {count}")
    print(f"\n[TIP] Next steps:")
    print(f"   1. Import {output_json} into Label Studio")
    print(f"   2. Review and correct claim annotations")
    print(f"   3. Export final annotations for training")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Claim-based pre-annotation using GPT-4o Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create and submit batch job
  python scripts/ai_preannotate_claims.py --submit
  
  # Check batch status
  python scripts/ai_preannotate_claims.py --check-status batch_xxx
  
  # Download results
  python scripts/ai_preannotate_claims.py --download batch_xxx
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
    parser.add_argument("--metadata", default="data/annotations/claim_batch_metadata.json",
                        help="Metadata file path")
    parser.add_argument("--output", default="data/annotations/claim_annotations.json",
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
        batch_input_file = "data/annotations/claim_batch_input.jsonl"
        create_claim_batch_input(args.input, batch_input_file, args.limit)
        batch_id = submit_batch(client, batch_input_file)
        
        # Save batch ID for reference
        with open("data/annotations/claim_batch_id.txt", "w") as f:
            f.write(batch_id)
    
    elif args.check_status:
        check_batch_status(client, args.check_status)
    
    elif args.download:
        download_and_convert_results(client, args.download, args.metadata, args.output)
    
    else:
        parser.print_help()

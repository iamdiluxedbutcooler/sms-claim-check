#!/usr/bin/env python3
"""
Label ALL 2000 messages using OpenAI Batch API (50% cheaper, 24hr processing)

Usage:
    # Step 1: Create batch
    OPENAI_API_KEY='your-key' python 2_claim_parsing_agent/label_all_batch.py create-batch
    
    # Step 2: Check status (run periodically)
    OPENAI_API_KEY='your-key' python 2_claim_parsing_agent/label_all_batch.py check-status
    
    # Step 3: Download results (after 24 hours)
    OPENAI_API_KEY='your-key' python 2_claim_parsing_agent/label_all_batch.py download-results
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
from openai import OpenAI

# Import modules
config_module = importlib.import_module("2_claim_parsing_agent.config")
data_loader_module = importlib.import_module("2_claim_parsing_agent.data_loader")
schemas_module = importlib.import_module("2_claim_parsing_agent.schemas")

ParsingConfig = config_module.ParsingConfig
format_schema_for_prompt = schemas_module.format_schema_for_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_batch_file():
    """Create JSONL file with all batch requests"""
    logger.info("=" * 80)
    logger.info("STEP 1: CREATE BATCH FILE")
    logger.info("=" * 80)
    
    # Load ALL messages and claims
    logger.info("Loading all messages and claims...")
    all_messages = data_loader_module.load_all_messages()
    all_claims = data_loader_module.load_claim_spans()
    
    logger.info(f"Total: {len(all_messages)} messages, {len(all_claims)} claims")
    
    # Group claims by message
    claims_by_message = defaultdict(list)
    for claim in all_claims:
        claims_by_message[claim.message_id].append(claim)
    
    # Create message ID to message map
    message_map = {m.message_id: m for m in all_messages}
    
    # Create batch requests
    batch_requests = []
    request_id = 0
    
    for message_id, claims in claims_by_message.items():
        message = message_map.get(message_id)
        if not message or not claims:
            continue
        
        # Build schemas text
        schemas_text = "\n\n".join([format_schema_for_prompt(c.claim_type) for c in claims])
        
        # Build claims list
        claims_list = []
        for idx, claim in enumerate(claims):
            claims_list.append(
                f"Claim {idx + 1}:\n"
                f"  - Type: {claim.claim_type}\n"
                f"  - Text: \"{claim.text}\"\n"
                f"  - Position: [{claim.start}:{claim.end}]"
            )
        claims_text = "\n\n".join(claims_list)
        
        # Create prompt
        prompt = f"""You are a claim parsing expert. Given an SMS message and extracted claim spans, 
parse each claim into a canonical form (a clear, standalone statement) and extract structured slot values.

SMS Message:
"{message.text}"

Extracted Claims:
{claims_text}

Slot Schemas:
{schemas_text}

For EACH claim, provide:
1. canonical_form: A clear, standalone statement of what is being claimed
2. slots: A dictionary of slot values extracted from the claim text and message context

Return ONLY a valid JSON array of objects, each with this structure:
{{
  "claim_id": <claim number 1-indexed>,
  "canonical_form": "<canonical statement>",
  "slots": {{"<slot_name>": "<value>", ...}}
}}

Rules:
- Only include slots that are explicitly present or can be inferred from the message
- Use null for slots that cannot be determined
- Keep slot values concise and normalized
- Ensure the canonical form is a complete, understandable statement

Return the JSON array now:"""
        
        # Create batch request
        batch_request = {
            "custom_id": f"request-{request_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2000,
                "temperature": 0.0,
            }
        }
        
        batch_requests.append({
            "request": batch_request,
            "message_id": message_id,
            "message_label": message.label,
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "claim_type": c.claim_type,
                    "text": c.text,
                    "start": c.start,
                    "end": c.end,
                }
                for c in claims
            ]
        })
        
        request_id += 1
    
    logger.info(f"Created {len(batch_requests)} batch requests")
    
    # Save batch file (JSONL format for OpenAI)
    batch_file_path = Path("data/batch_requests.jsonl")
    batch_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(batch_file_path, "w") as f:
        for item in batch_requests:
            f.write(json.dumps(item["request"]) + "\n")
    
    logger.info(f"✓ Saved batch requests to {batch_file_path}")
    
    # Save metadata
    metadata_path = Path("data/batch_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(batch_requests, f, indent=2)
    
    logger.info(f"✓ Saved metadata to {metadata_path}")
    
    # Estimate
    estimated_cost = len(all_claims) * 0.005  # 50% cheaper
    logger.info("")
    logger.info(f"Estimated cost: ~${estimated_cost:.2f} (50% cheaper than regular API)")
    logger.info(f"Processing time: ~24 hours")
    
    return batch_file_path


def submit_batch(batch_file_path: Path):
    """Submit batch to OpenAI"""
    logger.info("=" * 80)
    logger.info("STEP 2: SUBMIT BATCH TO OPENAI")
    logger.info("=" * 80)
    
    config = ParsingConfig()
    if not config.openai_api_key:
        logger.error("OPENAI_API_KEY not set!")
        sys.exit(1)
    
    client = OpenAI(api_key=config.openai_api_key)
    
    # Upload file
    logger.info(f"Uploading {batch_file_path}...")
    with open(batch_file_path, "rb") as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )
    
    logger.info(f"✓ File uploaded: {batch_input_file.id}")
    
    # Create batch
    logger.info("Creating batch...")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Claim parsing for all 2000 messages"
        }
    )
    
    logger.info(f"✓ Batch created: {batch.id}")
    logger.info(f"Status: {batch.status}")
    
    # Save batch ID
    batch_info_path = Path("data/batch_info.json")
    with open(batch_info_path, "w") as f:
        json.dump({
            "batch_id": batch.id,
            "input_file_id": batch_input_file.id,
            "status": batch.status,
            "created_at": batch.created_at,
        }, f, indent=2)
    
    logger.info(f"✓ Saved batch info to {batch_info_path}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("BATCH SUBMITTED!")
    logger.info("=" * 80)
    logger.info("Check status with:")
    logger.info(f"  python {__file__} check-status")
    logger.info("")
    logger.info("This will take ~24 hours to process.")
    logger.info("=" * 80)


def check_batch_status():
    """Check status of submitted batch"""
    logger.info("=" * 80)
    logger.info("CHECK BATCH STATUS")
    logger.info("=" * 80)
    
    config = ParsingConfig()
    if not config.openai_api_key:
        logger.error("OPENAI_API_KEY not set!")
        sys.exit(1)
    
    # Load batch info
    batch_info_path = Path("data/batch_info.json")
    if not batch_info_path.exists():
        logger.error("No batch info found! Run 'create-batch' first.")
        sys.exit(1)
    
    with open(batch_info_path, "r") as f:
        batch_info = json.load(f)
    
    batch_id = batch_info["batch_id"]
    
    client = OpenAI(api_key=config.openai_api_key)
    batch = client.batches.retrieve(batch_id)
    
    logger.info(f"Batch ID: {batch.id}")
    logger.info(f"Status: {batch.status}")
    logger.info(f"Total requests: {batch.request_counts.total}")
    logger.info(f"Completed: {batch.request_counts.completed}")
    logger.info(f"Failed: {batch.request_counts.failed}")
    
    if batch.status == "completed":
        logger.info("")
        logger.info("✓ Batch completed! Download results with:")
        logger.info(f"  python {__file__} download-results")
    elif batch.status == "failed":
        logger.error("✗ Batch failed!")
        logger.error(f"Errors: {batch.errors}")
    else:
        logger.info("")
        logger.info("Batch still processing. Check again later.")
        
        # Estimate time remaining
        if batch.request_counts.completed > 0:
            progress = batch.request_counts.completed / batch.request_counts.total
            logger.info(f"Progress: {progress:.1%}")


def download_batch_results():
    """Download and process batch results"""
    logger.info("=" * 80)
    logger.info("DOWNLOAD BATCH RESULTS")
    logger.info("=" * 80)
    
    config = ParsingConfig()
    if not config.openai_api_key:
        logger.error("OPENAI_API_KEY not set!")
        sys.exit(1)
    
    # Load batch info
    batch_info_path = Path("data/batch_info.json")
    with open(batch_info_path, "r") as f:
        batch_info = json.load(f)
    
    batch_id = batch_info["batch_id"]
    
    client = OpenAI(api_key=config.openai_api_key)
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        logger.error(f"Batch not completed yet! Status: {batch.status}")
        sys.exit(1)
    
    # Download results
    logger.info("Downloading results...")
    result_file_id = batch.output_file_id
    result = client.files.content(result_file_id)
    
    # Save raw results
    raw_results_path = Path("data/batch_results_raw.jsonl")
    with open(raw_results_path, "wb") as f:
        f.write(result.content)
    
    logger.info(f"✓ Downloaded raw results to {raw_results_path}")
    
    # Load metadata
    metadata_path = Path("data/batch_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Map custom_id to metadata
    metadata_map = {item["request"]["custom_id"]: item for item in metadata}
    
    # Parse results
    logger.info("Parsing results...")
    all_parsed = []
    
    with open(raw_results_path, "r") as f:
        for line in f:
            result_item = json.loads(line)
            custom_id = result_item["custom_id"]
            
            if result_item.get("error"):
                logger.warning(f"Error in {custom_id}: {result_item['error']}")
                continue
            
            # Get metadata
            meta = metadata_map.get(custom_id)
            if not meta:
                continue
            
            # Extract GPT response
            response = result_item["response"]["body"]["choices"][0]["message"]["content"]
            
            # Parse JSON
            try:
                # Clean JSON
                content = response.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                parsed_data = json.loads(content)
                
                if not isinstance(parsed_data, list):
                    continue
                
                # Process each claim
                for item in parsed_data:
                    claim_idx = item.get("claim_id", 0) - 1
                    
                    if claim_idx < 0 or claim_idx >= len(meta["claims"]):
                        continue
                    
                    original_claim = meta["claims"][claim_idx]
                    
                    all_parsed.append({
                        "message_id": meta["message_id"],
                        "message_label": meta["message_label"],
                        "claim_id": original_claim["claim_id"],
                        "claim_type": original_claim["claim_type"],
                        "canonical_form": item.get("canonical_form", ""),
                        "slots": item.get("slots", {}),
                    })
            
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON for {custom_id}")
                continue
    
    logger.info(f"✓ Parsed {len(all_parsed)} claims")
    
    # Save final results
    output_path = Path("data/all_gpt_labels.json")
    with open(output_path, "w") as f:
        json.dump(all_parsed, f, indent=2)
    
    logger.info(f"✓ Saved all labels to {output_path}")
    
    # Statistics
    claim_type_counts = defaultdict(int)
    for item in all_parsed:
        claim_type_counts[item["claim_type"]] += 1
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS DOWNLOADED!")
    logger.info("=" * 80)
    logger.info(f"Total claims parsed: {len(all_parsed)}")
    logger.info("")
    logger.info("Claim type distribution:")
    for claim_type, count in sorted(claim_type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {claim_type:25s}: {count:4d} claims")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("NEXT: Split into train/test and train T5 parser")
    logger.info("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch API labeling")
    parser.add_argument("command", choices=["create-batch", "check-status", "download-results"])
    
    args = parser.parse_args()
    
    if args.command == "create-batch":
        batch_file_path = create_batch_file()
        submit_batch(batch_file_path)
    elif args.command == "check-status":
        check_batch_status()
    elif args.command == "download-results":
        download_batch_results()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Label ALL 2000 messages with GPT (no sampling, no splits)

Usage:
    OPENAI_API_KEY='your-key' python 2_claim_parsing_agent/label_all_with_gpt.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib

# Import modules
config_module = importlib.import_module("2_claim_parsing_agent.config")
data_loader_module = importlib.import_module("2_claim_parsing_agent.data_loader")
parser_llm_module = importlib.import_module("2_claim_parsing_agent.parser_llm")

ParsingConfig = config_module.ParsingConfig
GPTClaimParser = parser_llm_module.GPTClaimParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("LABEL ALL 2000 MESSAGES WITH GPT")
    logger.info("=" * 80)
    
    config = ParsingConfig()
    
    if not config.openai_api_key:
        logger.error("OPENAI_API_KEY not set in environment!")
        logger.error("Please run: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Load ALL messages and claims
    logger.info("Loading all messages and claims...")
    all_messages = data_loader_module.load_all_messages()
    all_claims = data_loader_module.load_claim_spans()
    
    logger.info(f"Total: {len(all_messages)} messages, {len(all_claims)} claims")
    
    # Count by label
    ham_count = sum(1 for m in all_messages if m.label == "ham")
    smish_count = sum(1 for m in all_messages if m.label == "smish")
    logger.info(f"Distribution: {ham_count} ham, {smish_count} smish")
    
    # Group claims by message
    claims_by_message = defaultdict(list)
    for claim in all_claims:
        claims_by_message[claim.message_id].append(claim)
    
    # Filter to only messages with claims
    messages_with_claims = [m for m in all_messages if m.message_id in claims_by_message]
    logger.info(f"Messages with claims: {len(messages_with_claims)}")
    
    # Estimate
    avg_claims_per_msg = len(all_claims) / len(messages_with_claims)
    estimated_time = len(messages_with_claims) * 0.5 / 60
    estimated_cost = len(all_claims) * 0.01
    
    logger.info(f"Estimated time: ~{estimated_time:.0f} minutes")
    logger.info(f"Estimated cost: ~${estimated_cost:.2f}")
    
    logger.info("")
    logger.info("Starting GPT labeling...")
    logger.info("Note: This will take 15-25 minutes. Go get coffee! ☕")
    logger.info("")
    
    # Initialize parser
    parser = GPTClaimParser(
        api_key=config.openai_api_key,
        model=config.openai_model,
        max_tokens=config.openai_max_tokens,
        temperature=config.openai_temperature,
    )
    
    # Process all messages
    results = parser.parse_batch(
        messages=messages_with_claims,
        claims_by_message=claims_by_message,
        sleep_between_calls=0.5,
    )
    
    # Flatten results
    all_parsed = []
    for parsed_list in results.values():
        all_parsed.extend(parsed_list)
    
    logger.info(f"Generated {len(all_parsed)} parsed claims")
    
    # Save to JSON
    output_data = [
        {
            "message_id": pc.message_id,
            "message_label": pc.message_label,
            "claim_id": pc.claim_id,
            "claim_type": pc.claim_type,
            "canonical_form": pc.canonical_form,
            "slots": pc.slots,
        }
        for pc in all_parsed
    ]
    
    output_path = Path("data/all_gpt_labels.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✓ Saved all GPT labels to {output_path}")
    
    # Print statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("LABELING COMPLETE!")
    logger.info("=" * 80)
    
    claim_type_counts = defaultdict(int)
    for pc in all_parsed:
        claim_type_counts[pc.claim_type] += 1
    
    logger.info("\nClaim type distribution:")
    for claim_type, count in sorted(claim_type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {claim_type:25s}: {count:4d} claims")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("NEXT STEPS:")
    logger.info("=" * 80)
    logger.info("1. Split into train/test:")
    logger.info("   - Randomly select 400 messages for TEST (manual review)")
    logger.info("   - Use remaining 1600 for TRAIN (as-is)")
    logger.info("")
    logger.info("2. Manual review test set:")
    logger.info("   - Review canonical_form and slots")
    logger.info("   - Fix errors")
    logger.info("   - Save as data/test_gold_labels.json")
    logger.info("")
    logger.info("3. Train T5 parser:")
    logger.info("   python 2_claim_parsing_agent/hybrid_labeling_workflow.py train-t5 \\")
    logger.info("     --train-data data/train_silver_labels.json \\")
    logger.info("     --val-data data/test_gold_labels.json")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

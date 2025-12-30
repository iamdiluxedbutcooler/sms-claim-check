#!/usr/bin/env python3
"""
Hybrid Gold/Silver Labeling Workflow for Claim Parsing

This script automates the creation of:
1. Gold test set (400 messages) - GPT labels that need manual review
2. Silver training set (1600 messages) - GPT labels used as-is

Usage:
    # Step 1: Generate GPT labels for test set
    python 2_claim_parsing_agent/hybrid_labeling_workflow.py generate-test --output data/test_gpt_labels.json
    
    # Step 2: Manually review and save as data/test_gold_labels.json
    
    # Step 3: Generate GPT labels for training set
    python 2_claim_parsing_agent/hybrid_labeling_workflow.py generate-train --output data/train_silver_labels.json
    
    # Step 4: Train T5 parser
    python 2_claim_parsing_agent/hybrid_labeling_workflow.py train-t5 \
        --train-data data/train_silver_labels.json \
        --val-data data/test_gold_labels.json
"""

from __future__ import annotations

import argparse
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
gold_sampling_module = importlib.import_module("2_claim_parsing_agent.gold_sampling")
parser_llm_module = importlib.import_module("2_claim_parsing_agent.parser_llm")
parser_t5_module = importlib.import_module("2_claim_parsing_agent.parser_t5")
train_t5_module = importlib.import_module("2_claim_parsing_agent.train_t5_parser")

ParsingConfig = config_module.ParsingConfig
GPTClaimParser = parser_llm_module.GPTClaimParser
prepare_training_data = parser_t5_module.prepare_training_data
train_t5_parser = train_t5_module.train_t5_parser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_train_test_split():
    """Create 80/20 train/test split and return message lists"""
    from sklearn.model_selection import train_test_split
    
    logger.info("Loading all messages...")
    messages = data_loader_module.load_all_messages()
    
    # Split by label to maintain balance
    ham = [m for m in messages if m.label == "ham"]
    smish = [m for m in messages if m.label == "smish" or m.label != "ham"]
    
    logger.info(f"Total: {len(messages)} messages ({len(ham)} ham, {len(smish)} smish)")
    
    # 80/20 split for each class
    ham_train, ham_test = train_test_split(ham, test_size=0.2, random_state=42)
    smish_train, smish_test = train_test_split(smish, test_size=0.2, random_state=42)
    
    train_messages = ham_train + smish_train
    test_messages = ham_test + smish_test
    
    logger.info(f"Train: {len(train_messages)} messages ({len(ham_train)} ham, {len(smish_train)} smish)")
    logger.info(f"Test: {len(test_messages)} messages ({len(ham_test)} ham, {len(smish_test)} smish)")
    
    return train_messages, test_messages


def generate_test_labels(output_path: str, target_messages: int = 400):
    """
    Step 1: Generate GPT labels for test set with stratified sampling
    These will need manual review to become gold labels
    """
    logger.info("=" * 80)
    logger.info("STEP 1: GENERATE GPT LABELS FOR TEST SET (FOR MANUAL REVIEW)")
    logger.info("=" * 80)
    
    config = ParsingConfig()
    
    if not config.openai_api_key:
        logger.error("OPENAI_API_KEY not set in environment!")
        logger.error("Please run: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Get train/test split
    _, test_messages = create_train_test_split()
    all_claims = data_loader_module.load_claim_spans()
    test_message_ids = {m.message_id for m in test_messages}
    test_claims = [c for c in all_claims if c.message_id in test_message_ids]
    
    logger.info(f"Test set: {len(test_messages)} messages, {len(test_claims)} claims")
    
    # Use gold sampling to select representative subset
    logger.info(f"Selecting {target_messages} representative messages with rare claim coverage...")
    selected_ids = gold_sampling_module.select_messages_for_parsing_gold_set(
        messages=test_messages,
        claims=test_claims,
        target_num_messages=target_messages,
        min_per_rare_type=config.min_per_rare_type,
        random_seed=42,
    )
    
    selected_messages = [m for m in test_messages if m.message_id in selected_ids]
    selected_claims = [c for c in test_claims if c.message_id in selected_ids]
    
    logger.info(f"Selected {len(selected_messages)} messages with {len(selected_claims)} claims")
    
    # Group claims by message
    claims_by_message = defaultdict(list)
    for claim in selected_claims:
        claims_by_message[claim.message_id].append(claim)
    
    # Generate GPT labels
    logger.info("Generating GPT labels (this may take a few minutes)...")
    parser = GPTClaimParser(
        api_key=config.openai_api_key,
        model=config.openai_model,
        max_tokens=config.openai_max_tokens,
        temperature=config.openai_temperature,
    )
    
    results = parser.parse_batch(
        messages=selected_messages,
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
            "needs_review": True,  # Mark for manual review
        }
        for pc in all_parsed
    ]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✓ Saved GPT labels to {output_path}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("NEXT STEP: MANUAL REVIEW")
    logger.info("=" * 80)
    logger.info(f"1. Open {output_path}")
    logger.info("2. Review each claim's canonical_form and slots")
    logger.info("3. Fix any errors or missing information")
    logger.info("4. Remove 'needs_review': True field")
    logger.info("5. Save as data/test_gold_labels.json")
    logger.info(f"6. Then run: python {__file__} generate-train")
    logger.info("=" * 80)
    
    # Print statistics
    claim_type_counts = defaultdict(int)
    for pc in all_parsed:
        claim_type_counts[pc.claim_type] += 1
    
    logger.info("\nClaim type distribution in test set:")
    for claim_type, count in sorted(claim_type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {claim_type:25s}: {count:3d} claims")


def generate_train_labels(output_path: str):
    """
    Step 3: Generate GPT labels for training set
    These are used as-is (silver labels, no manual review)
    """
    logger.info("=" * 80)
    logger.info("STEP 3: GENERATE GPT LABELS FOR TRAINING SET (SILVER LABELS)")
    logger.info("=" * 80)
    
    config = ParsingConfig()
    
    if not config.openai_api_key:
        logger.error("OPENAI_API_KEY not set in environment!")
        sys.exit(1)
    
    # Get train/test split
    train_messages, _ = create_train_test_split()
    all_claims = data_loader_module.load_claim_spans()
    train_message_ids = {m.message_id for m in train_messages}
    train_claims = [c for c in all_claims if c.message_id in train_message_ids]
    
    logger.info(f"Training set: {len(train_messages)} messages, {len(train_claims)} claims")
    
    # Group claims by message
    claims_by_message = defaultdict(list)
    for claim in train_claims:
        claims_by_message[claim.message_id].append(claim)
    
    # Filter to only messages with claims
    messages_with_claims = [m for m in train_messages if m.message_id in claims_by_message]
    logger.info(f"Processing {len(messages_with_claims)} messages that have claims...")
    
    # Generate GPT labels
    logger.info("Generating GPT labels (this will take 10-15 minutes)...")
    logger.info("Note: These are SILVER labels (not manually reviewed)")
    
    parser = GPTClaimParser(
        api_key=config.openai_api_key,
        model=config.openai_model,
        max_tokens=config.openai_max_tokens,
        temperature=config.openai_temperature,
    )
    
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
            "silver_label": True,  # Mark as silver (not gold)
        }
        for pc in all_parsed
    ]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✓ Saved silver training labels to {output_path}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("NEXT STEP: TRAIN T5 PARSER")
    logger.info("=" * 80)
    logger.info(f"python {__file__} train-t5 \\")
    logger.info(f"  --train-data {output_path} \\")
    logger.info(f"  --val-data data/test_gold_labels.json")
    logger.info("=" * 80)


def train_t5(train_data_path: str, val_data_path: str, output_dir: str = "models/claim_parsing_t5"):
    """
    Step 4: Train T5 parser on silver training data, validate on gold test data
    """
    logger.info("=" * 80)
    logger.info("STEP 4: TRAIN T5 PARSER")
    logger.info("=" * 80)
    
    # Load gold labels
    with open(val_data_path, "r") as f:
        val_gold_data = json.load(f)
    
    logger.info(f"Loaded {len(val_gold_data)} gold validation labels from {val_data_path}")
    
    # Load silver training labels
    with open(train_data_path, "r") as f:
        train_silver_data = json.load(f)
    
    logger.info(f"Loaded {len(train_silver_data)} silver training labels from {train_data_path}")
    
    # Convert to ParsedClaim objects
    models_module = importlib.import_module("2_claim_parsing_agent.models")
    ParsedClaim = models_module.ParsedClaim
    
    val_parsed_claims = [ParsedClaim(**item) for item in val_gold_data]
    train_parsed_claims = [ParsedClaim(**item) for item in train_silver_data]
    
    # Load messages and claims
    messages = data_loader_module.load_all_messages()
    claims = data_loader_module.load_claim_spans()
    
    # Prepare training examples
    logger.info("Preparing training examples...")
    train_examples = prepare_training_data(messages, claims, train_parsed_claims)
    val_examples = prepare_training_data(messages, claims, val_parsed_claims)
    
    logger.info(f"Prepared {len(train_examples)} training examples")
    logger.info(f"Prepared {len(val_examples)} validation examples")
    
    # Train
    logger.info("Starting T5 training...")
    train_t5_parser(
        train_examples=train_examples,
        val_examples=val_examples,
        model_name="t5-base",
        output_dir=output_dir,
        learning_rate=5e-5,
        batch_size=8,
        epochs=3,
        max_input_length=256,
        max_output_length=256,
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {output_dir}")
    logger.info("")
    logger.info("NEXT STEP: EVALUATE BOTH PARSERS")
    logger.info("=" * 80)
    logger.info("# Evaluate GPT parser:")
    logger.info("python scripts/run_parsing_experiment.py \\")
    logger.info("  --parser-type gpt \\")
    logger.info("  --gold-labels-path data/test_gold_labels.json")
    logger.info("")
    logger.info("# Evaluate T5 parser:")
    logger.info("python scripts/run_parsing_experiment.py \\")
    logger.info("  --parser-type t5 \\")
    logger.info(f"  --t5-model-path {output_dir} \\")
    logger.info("  --gold-labels-path data/test_gold_labels.json")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid gold/silver labeling workflow for claim parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate test set labels (need manual review)
  python %(prog)s generate-test --output data/test_gpt_labels.json
  
  # After manual review, generate training set labels
  python %(prog)s generate-train --output data/train_silver_labels.json
  
  # Train T5 parser
  python %(prog)s train-t5 \\
    --train-data data/train_silver_labels.json \\
    --val-data data/test_gold_labels.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # generate-test command
    test_parser = subparsers.add_parser("generate-test", help="Generate GPT labels for test set (need review)")
    test_parser.add_argument("--output", default="data/test_gpt_labels.json", help="Output path")
    test_parser.add_argument("--target-messages", type=int, default=400, help="Target number of test messages")
    
    # generate-train command
    train_parser = subparsers.add_parser("generate-train", help="Generate GPT labels for training set (silver)")
    train_parser.add_argument("--output", default="data/train_silver_labels.json", help="Output path")
    
    # train-t5 command
    t5_parser = subparsers.add_parser("train-t5", help="Train T5 parser")
    t5_parser.add_argument("--train-data", required=True, help="Path to silver training labels")
    t5_parser.add_argument("--val-data", required=True, help="Path to gold validation labels")
    t5_parser.add_argument("--output-dir", default="models/claim_parsing_t5", help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == "generate-test":
        generate_test_labels(args.output, args.target_messages)
    elif args.command == "generate-train":
        generate_train_labels(args.output)
    elif args.command == "train-t5":
        train_t5(args.train_data, args.val_data, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import importlib.util

agent2_path = project_root / "2_claim_parsing_agent"
spec = importlib.util.spec_from_file_location("claim_parsing_agent", agent2_path / "__init__.py")
claim_parsing_agent = importlib.util.module_from_spec(spec)

from openai import OpenAI

config_module = importlib.import_module("2_claim_parsing_agent.config")
data_loader_module = importlib.import_module("2_claim_parsing_agent.data_loader")
evaluation_module = importlib.import_module("2_claim_parsing_agent.evaluation")
gold_sampling_module = importlib.import_module("2_claim_parsing_agent.gold_sampling")
models_module = importlib.import_module("2_claim_parsing_agent.models")
parser_llm_module = importlib.import_module("2_claim_parsing_agent.parser_llm")
parser_t5_module = importlib.import_module("2_claim_parsing_agent.parser_t5")

ParsingConfig = config_module.ParsingConfig
get_claims_for_messages = data_loader_module.get_claims_for_messages
get_messages_by_ids = data_loader_module.get_messages_by_ids
get_messages_by_split = data_loader_module.get_messages_by_split
load_claim_spans = data_loader_module.load_claim_spans
evaluate_parsing = evaluation_module.evaluate_parsing
print_evaluation_summary = evaluation_module.print_evaluation_summary
select_messages_for_parsing_gold_set = gold_sampling_module.select_messages_for_parsing_gold_set
ParsedClaim = models_module.ParsedClaim
GPTClaimParser = parser_llm_module.GPTClaimParser
T5ClaimParser = parser_t5_module.T5ClaimParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_gold_labels(gold_labels_path: str) -> list[ParsedClaim]:
    with open(gold_labels_path, "r") as f:
        data = json.load(f)
    
    parsed_claims = []
    for item in data:
        parsed_claims.append(
            ParsedClaim(
                message_id=item["message_id"],
                message_label=item.get("message_label"),
                claim_id=item["claim_id"],
                claim_type=item["claim_type"],
                canonical_form=item["canonical_form"],
                slots=item.get("slots", {}),
            )
        )
    
    logger.info(f"Loaded {len(parsed_claims)} gold parsed claims from {gold_labels_path}")
    return parsed_claims


def main():
    parser = argparse.ArgumentParser(description="Run claim parsing experiments")
    parser.add_argument(
        "--parser-type",
        choices=["gpt", "t5"],
        required=True,
        help="Parser type to use"
    )
    parser.add_argument(
        "--gold-labels-path",
        type=str,
        help="Path to gold parsing dataset JSON (if available for evaluation)"
    )
    parser.add_argument(
        "--target-num-messages",
        type=int,
        default=300,
        help="Target number of messages for gold set sampling"
    )
    parser.add_argument(
        "--t5-model-path",
        type=str,
        help="Path to trained T5 model (required for t5 parser)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/parsing_experiments",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to JSON file"
    )
    
    args = parser.parse_args()
    
    config = ParsingConfig()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading messages and claims from test split...")
    test_messages = get_messages_by_split("test")
    all_claims = load_claim_spans()
    test_claims = [c for c in all_claims if c.message_id in {m.message_id for m in test_messages}]
    
    logger.info(f"Loaded {len(test_messages)} test messages with {len(test_claims)} claims")
    
    logger.info("Selecting parsing gold set...")
    selected_message_ids = select_messages_for_parsing_gold_set(
        messages=test_messages,
        claims=test_claims,
        target_num_messages=args.target_num_messages,
        min_per_rare_type=config.min_per_rare_type,
    )
    
    selected_messages = get_messages_by_ids(selected_message_ids)
    selected_claims = get_claims_for_messages(selected_message_ids)
    
    logger.info(f"Selected {len(selected_messages)} messages with {len(selected_claims)} claims")
    
    claims_by_message = defaultdict(list)
    for claim in selected_claims:
        claims_by_message[claim.message_id].append(claim)
    
    logger.info(f"Initializing {args.parser_type} parser...")
    
    if args.parser_type == "gpt":
        if not config.openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            sys.exit(1)
        
        parser = GPTClaimParser(
            api_key=config.openai_api_key,
            model=config.openai_model,
            max_tokens=config.openai_max_tokens,
            temperature=config.openai_temperature,
        )
        
        logger.info("Running GPT-based parsing...")
        predictions_by_message = parser.parse_batch(
            messages=selected_messages,
            claims_by_message=claims_by_message,
        )
    
    elif args.parser_type == "t5":
        if not args.t5_model_path:
            logger.error("--t5-model-path is required for t5 parser")
            sys.exit(1)
        
        parser = T5ClaimParser(
            model_path=args.t5_model_path,
            max_input_length=config.t5_max_input_length,
            max_output_length=config.t5_max_output_length,
        )
        
        logger.info("Running T5-based parsing...")
        predictions_by_message = {}
        for message in selected_messages:
            msg_claims = claims_by_message[message.message_id]
            parsed = parser.parse_message(message, msg_claims)
            predictions_by_message[message.message_id] = parsed
    
    all_predictions = []
    for preds in predictions_by_message.values():
        all_predictions.extend(preds)
    
    logger.info(f"Generated {len(all_predictions)} parsed claim predictions")
    
    if args.save_predictions:
        predictions_file = output_dir / f"predictions_{args.parser_type}.json"
        predictions_data = [
            {
                "message_id": pc.message_id,
                "message_label": pc.message_label,
                "claim_id": pc.claim_id,
                "claim_type": pc.claim_type,
                "canonical_form": pc.canonical_form,
                "slots": pc.slots,
            }
            for pc in all_predictions
        ]
        
        with open(predictions_file, "w") as f:
            json.dump(predictions_data, f, indent=2)
        
        logger.info(f"Saved predictions to {predictions_file}")
    
    if args.gold_labels_path:
        logger.info("Loading gold labels for evaluation...")
        gold_labels = load_gold_labels(args.gold_labels_path)
        
        logger.info("Evaluating parsing performance...")
        metrics = evaluate_parsing(gold_labels, all_predictions)
        
        print_evaluation_summary(metrics)
        
        metrics_file = output_dir / f"metrics_{args.parser_type}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_file}")
    else:
        logger.info("No gold labels provided - skipping evaluation")
        logger.info("To evaluate, provide --gold-labels-path with gold ParsedClaim labels")
    
    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()

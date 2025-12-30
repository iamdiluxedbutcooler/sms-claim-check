from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib

config_module = importlib.import_module("2_claim_parsing_agent.config")
data_loader_module = importlib.import_module("2_claim_parsing_agent.data_loader")
evaluation_module = importlib.import_module("2_claim_parsing_agent.evaluation")
models_module = importlib.import_module("2_claim_parsing_agent.models")

ParsingConfig = config_module.ParsingConfig
ParsedClaim = models_module.ParsedClaim

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_test_data(test_path: str):
    messages = data_loader_module.load_all_messages()
    claims = data_loader_module.load_claim_spans()
    
    with open(test_path, "r") as f:
        gold_parsed = [ParsedClaim(**item) for item in json.load(f)]
    
    test_message_ids = {p.message_id for p in gold_parsed}
    test_messages = [m for m in messages if m.message_id in test_message_ids]
    test_claims = [c for c in claims if c.message_id in test_message_ids]
    
    claims_by_message = defaultdict(list)
    for claim in test_claims:
        claims_by_message[claim.message_id].append(claim)
    
    return test_messages, claims_by_message, gold_parsed


def run_zero_shot_gpt(test_messages, claims_by_message, config):
    logger.info("Running Zero-Shot GPT-4o...")
    parser_llm = importlib.import_module("2_claim_parsing_agent.parser_llm")
    GPTClaimParser = parser_llm.GPTClaimParser
    
    parser = GPTClaimParser(
        api_key=config.openai_api_key,
        model=config.openai_model,
        max_tokens=config.openai_max_tokens,
        temperature=config.openai_temperature,
    )
    
    start_time = time.time()
    results = parser.parse_batch(test_messages, claims_by_message, sleep_between_calls=0.5)
    elapsed = time.time() - start_time
    
    predictions = [pred for preds in results.values() for pred in preds]
    
    return predictions, elapsed


def run_few_shot_gpt(test_messages, claims_by_message, config, train_path):
    logger.info("Running Few-Shot GPT-4o...")
    parser_fewshot = importlib.import_module("2_claim_parsing_agent.parser_fewshot")
    FewShotGPTParser = parser_fewshot.FewShotGPTParser
    
    with open(train_path, "r") as f:
        training_data = [ParsedClaim(**item) for item in json.load(f)]
    
    parser = FewShotGPTParser(
        api_key=config.openai_api_key,
        model=config.openai_model,
        max_tokens=config.openai_max_tokens,
        temperature=config.openai_temperature,
        num_examples=3,
        training_data=training_data,
    )
    
    start_time = time.time()
    results = {}
    for message in test_messages:
        claims = claims_by_message.get(message.message_id, [])
        if claims:
            results[message.message_id] = parser.parse(message, claims)
            time.sleep(0.5)
    elapsed = time.time() - start_time
    
    predictions = [pred for preds in results.values() for pred in preds]
    
    return predictions, elapsed


def run_vanilla_t5(test_messages, claims_by_message, model_path):
    logger.info("Running Vanilla T5...")
    parser_t5 = importlib.import_module("2_claim_parsing_agent.parser_t5")
    T5ClaimParser = parser_t5.T5ClaimParser
    
    parser = T5ClaimParser(model_path=model_path)
    
    start_time = time.time()
    results = parser.parse_batch(test_messages, claims_by_message)
    elapsed = time.time() - start_time
    
    predictions = [pred for preds in results.values() for pred in preds]
    
    return predictions, elapsed


def run_enhanced_t5(test_messages, claims_by_message, model_path):
    logger.info("Running Enhanced T5...")
    parser_enhanced = importlib.import_module("2_claim_parsing_agent.parser_enhanced_t5")
    EnhancedT5Parser = parser_enhanced.EnhancedT5Parser
    
    parser = EnhancedT5Parser(model_path=model_path)
    
    start_time = time.time()
    results = parser.parse_batch(test_messages, claims_by_message)
    elapsed = time.time() - start_time
    
    predictions = [pred for preds in results.values() for pred in preds]
    
    return predictions, elapsed


def evaluate_predictions(predictions, gold_parsed, approach_name):
    logger.info(f"\nEvaluating {approach_name}...")
    
    metrics = evaluation_module.evaluate_parsing(predictions, gold_parsed)
    
    logger.info(f"Results for {approach_name}:")
    logger.info(f"  Slot F1: {metrics['slot_f1_overall']:.3f}")
    logger.info(f"  Slot Precision: {metrics['slot_precision_overall']:.3f}")
    logger.info(f"  Slot Recall: {metrics['slot_recall_overall']:.3f}")
    logger.info(f"  Canonical Accuracy: {metrics['canonical_accuracy']:.3f}")
    
    logger.info(f"\nPer-Claim-Type F1:")
    for claim_type, f1 in sorted(metrics['slot_f1_by_type'].items(), key=lambda x: -x[1]):
        logger.info(f"  {claim_type:25s}: {f1:.3f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run all 4 parsing experiments")
    parser.add_argument("--test-data", required=True, help="Path to test JSON")
    parser.add_argument("--train-data", required=True, help="Path to train JSON")
    parser.add_argument("--vanilla-t5-model", help="Path to vanilla T5 model")
    parser.add_argument("--enhanced-t5-model", help="Path to enhanced T5 model")
    parser.add_argument("--output-dir", default="output/experiments", help="Output directory")
    parser.add_argument("--skip-gpt", action="store_true", help="Skip GPT experiments")
    parser.add_argument("--skip-t5", action="store_true", help="Skip T5 experiments")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = ParsingConfig()
    
    logger.info("Loading test data...")
    test_messages, claims_by_message, gold_parsed = load_test_data(args.test_data)
    logger.info(f"Loaded {len(test_messages)} test messages, {len(gold_parsed)} gold claims")
    
    all_results = {}
    
    if not args.skip_gpt:
        if config.openai_api_key:
            predictions, elapsed = run_zero_shot_gpt(test_messages, claims_by_message, config)
            metrics = evaluate_predictions(predictions, gold_parsed, "Zero-Shot GPT-4o")
            metrics['elapsed_time'] = elapsed
            metrics['speed_per_message'] = elapsed / len(test_messages)
            all_results['zero_shot_gpt'] = metrics
            
            with open(output_dir / "predictions_zero_shot_gpt.json", "w") as f:
                json.dump([p.__dict__ for p in predictions], f, indent=2)
            
            predictions, elapsed = run_few_shot_gpt(test_messages, claims_by_message, config, args.train_data)
            metrics = evaluate_predictions(predictions, gold_parsed, "Few-Shot GPT-4o")
            metrics['elapsed_time'] = elapsed
            metrics['speed_per_message'] = elapsed / len(test_messages)
            all_results['few_shot_gpt'] = metrics
            
            with open(output_dir / "predictions_few_shot_gpt.json", "w") as f:
                json.dump([p.__dict__ for p in predictions], f, indent=2)
        else:
            logger.warning("OPENAI_API_KEY not set, skipping GPT experiments")
    
    if not args.skip_t5:
        if args.vanilla_t5_model and Path(args.vanilla_t5_model).exists():
            predictions, elapsed = run_vanilla_t5(test_messages, claims_by_message, args.vanilla_t5_model)
            metrics = evaluate_predictions(predictions, gold_parsed, "Vanilla T5")
            metrics['elapsed_time'] = elapsed
            metrics['speed_per_message'] = elapsed / len(test_messages)
            all_results['vanilla_t5'] = metrics
            
            with open(output_dir / "predictions_vanilla_t5.json", "w") as f:
                json.dump([p.__dict__ for p in predictions], f, indent=2)
        else:
            logger.warning(f"Vanilla T5 model not found at {args.vanilla_t5_model}")
        
        if args.enhanced_t5_model and Path(args.enhanced_t5_model).exists():
            predictions, elapsed = run_enhanced_t5(test_messages, claims_by_message, args.enhanced_t5_model)
            metrics = evaluate_predictions(predictions, gold_parsed, "Enhanced T5")
            metrics['elapsed_time'] = elapsed
            metrics['speed_per_message'] = elapsed / len(test_messages)
            all_results['enhanced_t5'] = metrics
            
            with open(output_dir / "predictions_enhanced_t5.json", "w") as f:
                json.dump([p.__dict__ for p in predictions], f, indent=2)
        else:
            logger.warning(f"Enhanced T5 model not found at {args.enhanced_t5_model}")
    
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nAll results saved to {output_dir}")
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 80)
    for approach, metrics in all_results.items():
        logger.info(f"\n{approach.upper()}:")
        logger.info(f"  Slot F1: {metrics['slot_f1_overall']:.3f}")
        logger.info(f"  Speed: {metrics['speed_per_message']:.2f}s per message")


if __name__ == "__main__":
    main()

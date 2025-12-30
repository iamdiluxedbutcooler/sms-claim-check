from __future__ import annotations

import argparse
import logging
from pathlib import Path

from agents.common.models import ParsedClaim
from agents.claim_parsing_agent.data_loader import AnnotationLoader
from agents.claim_parsing_agent.gold_sampling import sample_gold_set, SamplingConfig
from agents.claim_parsing_agent.parser_llm import GPTClaimParser
from agents.claim_parsing_agent.parser_t5 import T5ParserConfig, T5ClaimParser
from agents.claim_parsing_agent.evaluation import evaluate_parsing

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(project_root: Path, parser_type: str) -> dict[str, float]:
    loader = AnnotationLoader(project_root)
    annotations = loader.load()
    messages = list(loader.iter_messages(annotations))
    sampling = sample_gold_set(project_root, SamplingConfig())
    claims_map = loader.claims_by_message(annotations)

    if parser_type == "gpt":
        parser = GPTClaimParser()
    elif parser_type == "t5":
        parser = T5ClaimParser(T5ParserConfig())
    else:
        raise ValueError(f"Unknown parser_type: {parser_type}")

    preds: list[ParsedClaim] = []
    for message_id in sampling.message_ids:
        message_tuple = next((m for m in messages if m[0] == message_id), None)
        if not message_tuple:
            continue
        _, label, text = message_tuple
        spans = claims_map.get(message_id, [])
        preds.extend(parser.parse_message(message_id, label, text, spans))

    gold: list[ParsedClaim] = []  # placeholder until human labels are available
    metrics = evaluate_parsing(gold, preds)
    logger.info("Parsing metrics: %s", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parsing experiment pipeline")
    parser.add_argument("--parser_type", type=str, choices=["gpt", "t5"], default="gpt")
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run_pipeline(args.project_root, args.parser_type)


if __name__ == "__main__":
    main()


import argparse
import json
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import EntityNERModel, ClaimNERModel, HybridNERLLMModel, ContrastiveModel

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ModelInference:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.approach = self.config.get('approach', 'entity_ner')
        
        logger.info(f"Loading model: {self.approach}")
        logger.info(f"From: {self.model_dir}")
        
        self.model = self._load_model()
    
    def _load_model(self):
        if self.approach == 'entity_ner':
            model = EntityNERModel(self.config)
        elif self.approach == 'claim_ner':
            model = ClaimNERModel(self.config)
        elif self.approach == 'hybrid_llm':
            model = HybridNERLLMModel(self.config)
        elif self.approach == 'contrastive':
            model = ContrastiveModel(self.config)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
        
        model.load(self.model_dir)
        
        logger.info("Model loaded successfully\n")
        return model
    
    def predict(self, text: str):
        logger.info(f"Input: {text}\n")
        
        result = self.model.predict(text)
        
        logger.info("Results:")
        logger.info("-" * 80)
        
        if self.approach in ['entity_ner', 'claim_ner']:
            if 'entities' in result:
                logger.info("\nExtracted Entities:")
                for entity in result['entities']:
                    logger.info(f"  - {entity['text']:<30} [{entity['label']}]")
            
            if 'claims' in result:
                logger.info("\nExtracted Claims:")
                for claim in result['claims']:
                    logger.info(f"  - {claim}")
        
        elif self.approach == 'hybrid_llm':
            logger.info("\nStructured Output:")
            logger.info(json.dumps(result, indent=2))
        
        elif self.approach == 'contrastive':
            logger.info(f"\nSimilarity Score: {result.get('similarity', 0):.4f}")
            logger.info(f"Is Phishing: {result.get('is_phishing', False)}")
        
        logger.info("-" * 80 + "\n")
        return result
    
    def predict_batch(self, texts):
        results = []
        for i, text in enumerate(texts, 1):
            logger.info(f"\n>>> Message {i}/{len(texts)} <<<\n")
            result = self.predict(text)
            results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained SMS phishing models"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model directory (e.g., experiments/entity_ner_roberta/final_model)'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Single SMS text to analyze'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='File with SMS messages (one per line)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode - enter messages manually'
    )
    
    args = parser.parse_args()
    
    inference = ModelInference(args.model)
    
    if args.text:
        inference.predict(args.text)
    
    elif args.file:
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(texts)} messages from {args.file}\n")
        inference.predict_batch(texts)
    
    elif args.interactive:
        logger.info("Interactive mode - Enter SMS messages (Ctrl+C to exit)")
        logger.info("=" * 80 + "\n")
        
        try:
            while True:
                text = input("\nEnter SMS text: ").strip()
                if text:
                    inference.predict(text)
        except KeyboardInterrupt:
            logger.info("\n\nExiting...")
    
    else:
        logger.info("Running on example messages...\n")
        
        examples = [
            "Your Amazon order #12345 requires verification. Click here: bit.ly/amzn123 or call 1-800-FAKE within 24 hours to avoid suspension.",
            "Your package from USPS is ready for delivery. Confirm your address at usps-track.info",
            "URGENT: Your PayPal account will be locked tonight unless you verify now at paypal-secure.ru",
        ]
        
        inference.predict_batch(examples)


if __name__ == "__main__":
    main()

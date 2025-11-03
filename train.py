
import argparse
import logging
import yaml
from pathlib import Path
import json
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import AnnotationLoader, EntityNERPreprocessor, ClaimNERPreprocessor, ContrastivePreprocessor
from src.models import EntityNERModel, ClaimNERModel, HybridNERLLMModel, ContrastiveModel

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.project_root = Path(__file__).parent
        
        self.output_dir = self.project_root / self.config['output_config']['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
        
        logger.info("="*70)
        logger.info(f"Model Training: {self.config['name']}")
        logger.info("="*70)
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Output: {self.output_dir}")
    
    def _load_config(self) -> dict:
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_data(self):
        logger.info("\n" + "="*70)
        logger.info("Loading Data")
        logger.info("="*70)
        
        data_config = self.config['data_config']
        annotations_file = self.project_root / data_config['annotations_file']
        
        logger.info(f"Loading annotations from: {annotations_file}")
        loader = AnnotationLoader(annotations_file)
        messages = loader.load()
        logger.info(f"Loaded {len(messages)} messages")
        
        logger.info("Splitting data...")
        train_msgs, val_msgs, test_msgs = loader.split_data(
            messages,
            train_ratio=data_config['train_ratio'],
            val_ratio=data_config['val_ratio'],
            test_ratio=data_config['test_ratio'],
            seed=data_config['seed']
        )
        
        logger.info(f"  Train: {len(train_msgs)} messages")
        logger.info(f"  Val: {len(val_msgs)} messages")
        logger.info(f"  Test: {len(test_msgs)} messages")
        
        approach = self.config['approach']
        logger.info(f"\nPreprocessing data for approach: {approach}")
        
        if approach == 'entity_ner':
            preprocessor = EntityNERPreprocessor(
                entity_labels=self.config['model_config'].get('entity_labels', [])
            )
            train_examples = preprocessor.prepare_examples(train_msgs)
            val_examples = preprocessor.prepare_examples(val_msgs)
            test_examples = preprocessor.prepare_examples(test_msgs)
        
        elif approach == 'claim_ner':
            preprocessor = ClaimNERPreprocessor()
            train_examples = preprocessor.prepare_examples(train_msgs)
            val_examples = preprocessor.prepare_examples(val_msgs)
            test_examples = preprocessor.prepare_examples(test_msgs)
        
        elif approach == 'hybrid_llm':
            preprocessor = EntityNERPreprocessor(
                entity_labels=self.config['model_config']['ner_config'].get('entity_labels', [])
            )
            train_examples = preprocessor.prepare_examples(train_msgs)
            val_examples = preprocessor.prepare_examples(val_msgs)
            test_examples = preprocessor.prepare_examples(test_msgs)
        
        elif approach == 'contrastive':
            preprocessor = ContrastivePreprocessor()
            train_examples = preprocessor.prepare_examples(train_msgs)
            val_examples = preprocessor.prepare_examples(val_msgs)
            test_examples = preprocessor.prepare_examples(test_msgs)
        
        else:
            raise ValueError(f"Unknown approach: {approach}")
        
        logger.info(f"Preprocessing complete:")
        logger.info(f"  Train examples: {len(train_examples)}")
        logger.info(f"  Val examples: {len(val_examples)}")
        logger.info(f"  Test examples: {len(test_examples)}")
        
        splits_info = {
            'train_ids': [ex['id'] for ex in train_examples],
            'val_ids': [ex['id'] for ex in val_examples],
            'test_ids': [ex['id'] for ex in test_examples],
        }
        
        with open(self.output_dir / 'data_splits.json', 'w') as f:
            json.dump(splits_info, f, indent=2)
        
        return train_examples, val_examples, test_examples
    
    def create_model(self):
        logger.info("\n" + "="*70)
        logger.info("Creating Model")
        logger.info("="*70)
        
        approach = self.config['approach']
        model_config = self.config['model_config']
        
        logger.info(f"Approach: {approach}")
        logger.info(f"Model config: {model_config.get('model_name', 'default')}")
        
        if approach == 'entity_ner':
            model = EntityNERModel(model_config)
        elif approach == 'claim_ner':
            model = ClaimNERModel(model_config)
        elif approach == 'hybrid_llm':
            model = HybridNERLLMModel(model_config)
        elif approach == 'contrastive':
            model = ContrastiveModel(model_config)
        else:
            raise ValueError(f"Unknown approach: {approach}")
        
        logger.info(f"Model created: {model.get_name()}")
        return model
    
    def train(self):
        logger.info("\nStarting training pipeline...")
        
        train_examples, val_examples, test_examples = self.load_data()
        
        model = self.create_model()
        
        logger.info("\n" + "="*70)
        logger.info("Training Model")
        logger.info("="*70)
        
        train_metrics = model.train(
            train_examples,
            val_examples,
            self.output_dir
        )
        
        logger.info("\nTraining complete!")
        logger.info(f"Final training metrics: {train_metrics}")
        
        logger.info("\n" + "="*70)
        logger.info("Evaluating on Test Set")
        logger.info("="*70)
        
        test_metrics = model.evaluate(test_examples)
        
        logger.info(f"\nTest metrics: {test_metrics}")
        
        final_model_dir = self.output_dir / "final_model"
        model.save(final_model_dir)
        
        logger.info(f"\nModel saved to: {final_model_dir}")
        
        all_metrics = {
            'config': self.config,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        logger.info("\n" + "="*70)
        logger.info("Training Complete!")
        logger.info("="*70)
        logger.info(f"Results saved to: {self.output_dir / 'results.json'}")
        
        return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train SMS phishing claim extraction models"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (e.g., configs/entity_ner.yaml)'
    )
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()

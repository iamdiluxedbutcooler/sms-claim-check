
import argparse
import logging
import yaml
from pathlib import Path
import json
import sys
from datetime import datetime
import numpy as np

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


class KFoldModelTrainer:
    """
    K-Fold Cross-Validation trainer that keeps test set completely blind.
    
    Training process:
    1. Load 459 training samples and 103 test samples
    2. Create 5 folds from the 459 training samples (~367 train / ~92 val per fold)
    3. Train 5 models (one per fold) and average results
    4. Test set (103 samples) is NEVER used until final evaluation
    """
    
    def __init__(self, config_path: Path, n_folds: int = 5):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.n_folds = n_folds
        self.project_root = Path(__file__).parent
        
        self.output_dir = self.project_root / self.config['output_config']['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.output_dir / f"kfold_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
        
        logger.info("="*70)
        logger.info(f"K-Fold Cross-Validation Training: {self.config['name']}")
        logger.info("="*70)
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Number of Folds: {self.n_folds}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"\n⚠️  Test set will remain BLIND until final evaluation")
    
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
        
        logger.info("\nSplitting into train/test (no validation - using K-fold CV)...")
        train_msgs, _, test_msgs = loader.split_data(
            messages,
            seed=data_config['seed']
        )
        
        logger.info(f"\n✓ Train: {len(train_msgs)} messages (will be split into {self.n_folds} folds)")
        logger.info(f"✓ Test:  {len(test_msgs)} messages (BLIND - for final evaluation only)")
        
        # Generate K-fold splits
        logger.info(f"\nGenerating {self.n_folds}-Fold CV splits...")
        folds = loader.get_kfold_splits(train_msgs, n_splits=self.n_folds, seed=data_config['seed'])
        
        return folds, test_msgs, loader
    
    def preprocess_fold(self, train_msgs, val_msgs):
        """Preprocess a single fold of train/val data"""
        approach = self.config['approach']
        
        if approach == 'entity_ner':
            preprocessor = EntityNERPreprocessor(
                entity_labels=self.config['model_config'].get('entity_labels', [])
            )
        elif approach == 'claim_ner':
            preprocessor = ClaimNERPreprocessor()
        elif approach == 'hybrid_llm':
            preprocessor = EntityNERPreprocessor(
                entity_labels=self.config['model_config']['ner_config'].get('entity_labels', [])
            )
        elif approach == 'contrastive':
            preprocessor = ContrastivePreprocessor()
        else:
            raise ValueError(f"Unknown approach: {approach}")
        
        train_examples = preprocessor.prepare_examples(train_msgs)
        val_examples = preprocessor.prepare_examples(val_msgs)
        
        return train_examples, val_examples, preprocessor
    
    def create_model(self):
        """Create a new model instance"""
        approach = self.config['approach']
        model_config = self.config['model_config']
        
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
        
        return model
    
    def train_fold(self, fold_idx, train_examples, val_examples, fold_output_dir):
        """Train a single fold"""
        logger.info("\n" + "="*70)
        logger.info(f"Training Fold {fold_idx + 1}/{self.n_folds}")
        logger.info("="*70)
        logger.info(f"Train examples: {len(train_examples)}")
        logger.info(f"Val examples:   {len(val_examples)}")
        
        model = self.create_model()
        
        # Train the model
        train_metrics = model.train(
            train_examples,
            val_examples,
            fold_output_dir
        )
        
        logger.info(f"\n✓ Fold {fold_idx + 1} training complete!")
        logger.info(f"  Validation metrics: {train_metrics}")
        
        # Save the trained model
        model_dir = fold_output_dir / "model"
        model.save(model_dir)
        
        return model, train_metrics
    
    def train(self):
        logger.info("\nStarting K-Fold Cross-Validation training pipeline...")
        
        # Load data and get K-fold splits
        folds, test_msgs, loader = self.load_data()
        
        # Track metrics across folds
        fold_metrics = []
        fold_models = []
        
        # Train each fold
        for fold_idx, (train_msgs, val_msgs) in enumerate(folds):
            fold_output_dir = self.output_dir / f"fold_{fold_idx + 1}"
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Preprocess this fold's data
            train_examples, val_examples, preprocessor = self.preprocess_fold(train_msgs, val_msgs)
            
            # Train on this fold
            model, fold_val_metrics = self.train_fold(
                fold_idx, 
                train_examples, 
                val_examples,
                fold_output_dir
            )
            
            fold_metrics.append(fold_val_metrics)
            fold_models.append((model, fold_output_dir))
            
            # Save fold info
            fold_info = {
                'fold': fold_idx + 1,
                'train_size': len(train_examples),
                'val_size': len(val_examples),
                'validation_metrics': fold_val_metrics,
            }
            
            with open(fold_output_dir / 'fold_results.json', 'w') as f:
                json.dump(fold_info, f, indent=2)
        
        # Calculate average validation metrics across folds
        logger.info("\n" + "="*70)
        logger.info("Cross-Validation Results (Averaged Across Folds)")
        logger.info("="*70)
        
        avg_metrics = self._average_metrics(fold_metrics)
        logger.info(f"\nAverage validation metrics:")
        for key, value in avg_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Now evaluate on BLIND test set
        logger.info("\n" + "="*70)
        logger.info("Final Evaluation on BLIND Test Set")
        logger.info("="*70)
        logger.info("⚠️  This is the first time models see the test data!")
        
        # Preprocess test data
        approach = self.config['approach']
        if approach == 'entity_ner':
            preprocessor = EntityNERPreprocessor(
                entity_labels=self.config['model_config'].get('entity_labels', [])
            )
        elif approach == 'claim_ner':
            preprocessor = ClaimNERPreprocessor()
        elif approach == 'hybrid_llm':
            preprocessor = EntityNERPreprocessor(
                entity_labels=self.config['model_config']['ner_config'].get('entity_labels', [])
            )
        elif approach == 'contrastive':
            preprocessor = ContrastivePreprocessor()
        
        test_examples = preprocessor.prepare_examples(test_msgs)
        logger.info(f"Test examples: {len(test_examples)}")
        
        # Evaluate each fold's model on test set
        test_metrics_per_fold = []
        for fold_idx, (model, fold_dir) in enumerate(fold_models):
            logger.info(f"\nEvaluating Fold {fold_idx + 1} model on test set...")
            fold_test_metrics = model.evaluate(test_examples)
            test_metrics_per_fold.append(fold_test_metrics)
            logger.info(f"  Fold {fold_idx + 1} test metrics: {fold_test_metrics}")
        
        # Average test metrics
        avg_test_metrics = self._average_metrics(test_metrics_per_fold)
        
        logger.info("\n" + "="*70)
        logger.info("FINAL RESULTS")
        logger.info("="*70)
        logger.info(f"\nAverage TEST metrics (across {self.n_folds} folds):")
        for key, value in avg_test_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Save final results
        final_results = {
            'config': self.config,
            'n_folds': self.n_folds,
            'cv_validation_metrics': {
                'per_fold': fold_metrics,
                'average': avg_metrics,
            },
            'test_metrics': {
                'per_fold': test_metrics_per_fold,
                'average': avg_test_metrics,
            },
            'timestamp': datetime.now().isoformat(),
        }
        
        results_file = self.output_dir / 'kfold_results.json'
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info("\n" + "="*70)
        logger.info("K-Fold Cross-Validation Complete!")
        logger.info("="*70)
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Individual fold models saved in: {self.output_dir}/fold_*")
        
        return avg_test_metrics
    
    def _average_metrics(self, metrics_list):
        """Average metrics across folds"""
        if not metrics_list:
            return {}
        
        # Collect all keys
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        avg_metrics = {}
        for key in all_keys:
            values = [m.get(key) for m in metrics_list if key in m and isinstance(m.get(key), (int, float))]
            if values:
                avg_metrics[f'{key}_mean'] = float(np.mean(values))
                avg_metrics[f'{key}_std'] = float(np.std(values))
        
        return avg_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train SMS phishing models with K-Fold Cross-Validation"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (e.g., configs/entity_ner.yaml)'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    
    args = parser.parse_args()
    
    trainer = KFoldModelTrainer(args.config, n_folds=args.n_folds)
    trainer.train()


if __name__ == "__main__":
    main()

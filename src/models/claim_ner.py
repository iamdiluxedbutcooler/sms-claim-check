
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import evaluate

from .base import BaseModel

logger = logging.getLogger(__name__)


class ClaimNERModel(BaseModel):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.claim_labels = ["O", "B-CLAIM", "I-CLAIM"]
        self.label2id = {label: idx for idx, label in enumerate(self.claim_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        self.model_name = config.get('model_name', 'distilbert-base-uncased')
        self.max_length = config.get('max_length', 128)
        
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.warmup_steps = config.get('warmup_steps', 500)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.seed = config.get('seed', 42)
    
    def train(
        self,
        train_examples: List[Dict],
        val_examples: List[Dict],
        output_dir: Path
    ) -> Dict[str, float]:
        logger.info(f"Training {self.get_name()} with {len(train_examples)} examples")
        
        # Add add_prefix_space=True for RoBERTa models
        tokenizer_kwargs = {}
        if 'roberta' in self.model_name.lower():
            tokenizer_kwargs['add_prefix_space'] = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        train_dataset = self._prepare_dataset(train_examples)
        val_dataset = self._prepare_dataset(val_examples)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            logging_dir=str(output_dir / "logs"),
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=3,
            seed=self.seed,
            report_to="none",
        )
        
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        metric = evaluate.load("seqeval")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=lambda x: self._compute_metrics(x, metric),
        )
        
        trainer.train()
        
        final_metrics = trainer.evaluate()
        logger.info(f"Training complete. Final metrics: {final_metrics}")
        
        self.save(output_dir / "final_model")
        
        return final_metrics
    
    def _prepare_dataset(self, examples: List[Dict]) -> Dataset:
        dataset = Dataset.from_list(examples)
        
        tokenized_dataset = dataset.map(
            self._tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def _tokenize_and_align_labels(self, examples: Dict) -> Dict:
        tokenized_inputs = self.tokenizer(
            examples['tokens'],
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )
        
        labels = []
        # Use 'claim_tags' from preprocessor
        label_key = 'claim_tags' if 'claim_tags' in examples else 'labels'
        for i, label_list in enumerate(examples[label_key]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id[label_list[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    def _compute_metrics(self, eval_pred, metric) -> Dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def evaluate(self, test_examples: List[Dict]) -> Dict[str, float]:
        logger.info(f"Evaluating on {len(test_examples)} test examples")
        
        test_dataset = self._prepare_dataset(test_examples)
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        metric = evaluate.load("seqeval")
        
        training_args = TrainingArguments(
            output_dir="./temp",
            per_device_eval_batch_size=self.batch_size,
            report_to="none",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            compute_metrics=lambda x: self._compute_metrics(x, metric),
        )
        
        results = trainer.evaluate(test_dataset)
        
        logger.info(f"Test results: {results}")
        return results
    
    def predict(self, text: str) -> Dict[str, Any]:
        tokens = text.split()
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        word_ids = inputs.word_ids(batch_index=0)
        predicted_labels = []
        
        previous_word_idx = None
        for word_idx, pred_id in zip(word_ids, predictions[0].tolist()):
            if word_idx is not None and word_idx != previous_word_idx:
                predicted_labels.append(self.id2label[pred_id])
            previous_word_idx = word_idx
        
        claims = self._extract_claims(tokens, predicted_labels)
        
        return {
            'text': text,
            'tokens': tokens,
            'labels': predicted_labels,
            'claims': claims
        }
    
    def _extract_claims(self, tokens: List[str], labels: List[str]) -> List[Dict]:
        claims = []
        current_claim = []
        
        for token, label in zip(tokens, labels):
            if label == 'B-CLAIM':
                if current_claim:
                    claims.append(' '.join(current_claim))
                current_claim = [token]
            elif label == 'I-CLAIM' and current_claim:
                current_claim.append(token)
            else:
                if current_claim:
                    claims.append(' '.join(current_claim))
                    current_claim = []
        
        if current_claim:
            claims.append(' '.join(current_claim))
        
        return claims
    
    def save(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        with open(output_dir / "label_mapping.json", 'w') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label,
                'claim_labels': self.claim_labels
            }, f, indent=2)
        
        with open(output_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load(self, model_dir: Path):
        model_dir = Path(model_dir)
        
        logger.info(f"Loading model from {model_dir}")
        
        tokenizer_kwargs = {}
        if 'roberta' in str(model_dir).lower():
            tokenizer_kwargs['add_prefix_space'] = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, **tokenizer_kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        
        with open(model_dir / "label_mapping.json", 'r') as f:
            label_data = json.load(f)
            self.label2id = label_data['label2id']
            self.id2label = {int(k): v for k, v in label_data['id2label'].items()}
            self.claim_labels = label_data['claim_labels']
        
        logger.info("Model loaded successfully")

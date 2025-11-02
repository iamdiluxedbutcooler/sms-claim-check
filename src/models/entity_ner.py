
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


class EntityNERModel(BaseModel):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.entity_labels = config.get('entity_labels', self._get_default_labels())
        self.label2id = {label: idx for idx, label in enumerate(self.entity_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        self.model_name = config.get('model_name', 'distilbert-base-uncased')
        self.max_length = config.get('max_length', 128)
        
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.warmup_steps = config.get('warmup_steps', 500)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.seed = config.get('seed', 42)
    
    def _get_default_labels(self) -> List[str]:
        return [
            "O",
            "B-BRAND", "I-BRAND",
            "B-ACCOUNT_TYPE", "I-ACCOUNT_TYPE",
            "B-ACCOUNT_ID", "I-ACCOUNT_ID",
            "B-MONETARY_AMOUNT", "I-MONETARY_AMOUNT",
            "B-URL", "I-URL",
            "B-PHONE_NUMBER", "I-PHONE_NUMBER",
            "B-DEADLINE", "I-DEADLINE",
            "B-ACTION_REQUIRED", "I-ACTION_REQUIRED"
        ]
    
    def train(self, train_examples: List[Dict], val_examples: List[Dict], output_dir: Path):
        logger.info(f"Training {self.get_name()} with {len(train_examples)} examples")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
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
            fp16=torch.cuda.is_available(),
            report_to="none",  # Disable wandb/tensorboard by default
        )
        
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        metric = evaluate.load("seqeval")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda x: self._compute_metrics(x, metric),
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        final_metrics = trainer.evaluate()
        
        logger.info(f"Training complete. Final metrics: {final_metrics}")
        
        return final_metrics
    
    def _prepare_dataset(self, examples: List[Dict]) -> Dataset:
        tokenized_inputs = self.tokenizer(
            examples['tokens'],
            truncation=True,
            is_split_into_words=True,
            padding=False,
            max_length=self.max_length
        )
        
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
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
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.id2label[p.item()] for p in predictions[0]]
        
        entities = []
        current_entity = None
        
        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = {
                    'type': entity_type,
                    'text': token,
                    'tokens': [token]
                }
            elif label.startswith('I-') and current_entity:
                current_entity['text'] += ' ' + token
                current_entity['tokens'].append(token)
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def save(self, output_dir: Path):
        model_dir = Path(model_dir)
        
        logger.info(f"Loading model from {model_dir}")
        
        with open(model_dir / "label_mapping.json", 'r') as f:
            label_data = json.load(f)
            self.label2id = label_data['label2id']
            self.id2label = {int(k): v for k, v in label_data['id2label'].items()}
            self.entity_labels = label_data['entity_labels']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        
        logger.info("Model loaded successfully")

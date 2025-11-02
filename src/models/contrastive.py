
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from .base import BaseModel

logger = logging.getLogger(__name__)


class ContrastiveSMSDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        encoding = self.tokenizer(
            example['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': example['label']
        }


class ContrastiveEncoder(nn.Module):
    def __init__(self, model_name: str, embedding_dim: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, embedding_dim)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        projected = self.projection(pooled)
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = features
        
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature
        )
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss


class ContrastiveModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model_name = config.get('model_name', 'distilbert-base-uncased')
        self.embedding_dim = config.get('embedding_dim', 256)
        self.max_length = config.get('max_length', 128)
        
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.temperature = config.get('temperature', 0.07)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(
        self,
        train_examples: List[Dict],
        val_examples: List[Dict],
        output_dir: Path
    ) -> Dict[str, float]:
        
        logger.info(f"Training {self.get_name()} with {len(train_examples)} examples")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = ContrastiveEncoder(self.model_name, self.embedding_dim)
        self.model.to(self.device)
        
        train_dataset = ContrastiveSMSDataset(train_examples, self.tokenizer, self.max_length)
        val_dataset = ContrastiveSMSDataset(val_examples, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        criterion = SupConLoss(temperature=self.temperature)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        best_val_loss = float('inf')
        metrics_history = []
        
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )
            
            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(output_dir / "best_model")
        
        final_metrics = metrics_history[-1]
        final_metrics['best_val_loss'] = best_val_loss
        
        return final_metrics
    
    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            embeddings = self.model(input_ids, attention_mask)
            
            loss = criterion(embeddings, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                embeddings = self.model(input_ids, attention_mask)
                
                loss = criterion(embeddings, labels)
                total_loss += loss.item()
                
                total += labels.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = 0.0
        
        return avg_loss, accuracy
    
    def evaluate(self, test_examples: List[Dict]) -> Dict[str, float]:
        logger.info(f"Evaluating on {len(test_examples)} test examples")
        
        test_dataset = ContrastiveSMSDataset(test_examples, self.tokenizer, self.max_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        criterion = SupConLoss(temperature=self.temperature)
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                embeddings = self.model(input_ids, attention_mask)
                loss = criterion(embeddings, labels)
                
                total_loss += loss.item()
        
        test_loss = total_loss / len(test_loader)
        
        logger.info(f"Test loss: {test_loss:.4f}")
        
        return {'test_loss': test_loss}
    
    def predict(self, text: str) -> Dict[str, Any]:
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            embedding = self.model(input_ids, attention_mask)
        
        return {
            'text': text,
            'embedding': embedding.cpu().numpy()[0].tolist()
        }
    
    def save(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, output_dir / "model.pt")
        
        self.tokenizer.save_pretrained(output_dir)
        
        with open(output_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load(self, model_dir: Path):
        model_dir = Path(model_dir)
        
        logger.info(f"Loading model from {model_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        self.model = ContrastiveEncoder(self.model_name, self.embedding_dim)
        
        checkpoint = torch.load(model_dir / "model.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        logger.info("Model loaded successfully")

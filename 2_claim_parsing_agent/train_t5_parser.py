from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from tqdm import tqdm

from .config import ParsingConfig
from .parser_t5 import prepare_training_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class T5ParsingDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, str]],
        tokenizer: T5Tokenizer,
        max_input_length: int,
        max_output_length: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        input_encoding = self.tokenizer(
            example["input_text"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        target_encoding = self.tokenizer(
            example["target_text"],
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def train_t5_parser(
    train_examples: list[dict[str, str]],
    val_examples: list[dict[str, str]] | None = None,
    model_name: str = "t5-base",
    output_dir: Path | str = "models/claim_parsing_t5",
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    epochs: int = 3,
    max_input_length: int = 256,
    max_output_length: int = 256,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    
    train_dataset = T5ParsingDataset(
        train_examples, tokenizer, max_input_length, max_output_length
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if val_examples:
        val_dataset = T5ParsingDataset(
            val_examples, tokenizer, max_input_length, max_output_length
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    logger.info(f"Training for {epochs} epochs on {len(train_examples)} examples")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        
        if val_loader:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")
    
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train T5 claim parser")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data JSON")
    parser.add_argument("--val-data", type=str, help="Path to validation data JSON")
    parser.add_argument("--model-name", type=str, default="t5-base", help="Base T5 model name")
    parser.add_argument("--output-dir", type=str, default="models/claim_parsing_t5", help="Output directory")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max-input-length", type=int, default=256, help="Max input length")
    parser.add_argument("--max-output-length", type=int, default=256, help="Max output length")
    
    args = parser.parse_args()
    
    with open(args.train_data, "r") as f:
        train_examples = json.load(f)
    
    val_examples = None
    if args.val_data:
        with open(args.val_data, "r") as f:
            val_examples = json.load(f)
    
    train_t5_parser(
        train_examples=train_examples,
        val_examples=val_examples,
        model_name=args.model_name,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
    )


if __name__ == "__main__":
    main()

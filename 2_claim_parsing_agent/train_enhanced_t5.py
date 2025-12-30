from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

from .parser_enhanced_t5 import (
    EnhancedT5Dataset,
    prepare_enhanced_training_data,
    augment_training_data,
    create_curriculum_splits,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_enhanced_t5(
    train_examples: list[dict],
    val_examples: list[dict],
    model_name: str = "t5-base",
    output_dir: str | Path = "models/enhanced_t5_parser",
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    epochs_per_phase: int = 2,
    max_input_length: int = 256,
    max_output_length: int = 256,
    use_curriculum: bool = True,
    use_augmentation: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    
    if use_augmentation:
        logger.info("Applying data augmentation...")
        train_examples = augment_training_data(train_examples, augmentation_factor=2)
        logger.info(f"Augmented to {len(train_examples)} training examples")
    
    if use_curriculum:
        logger.info("Using curriculum learning...")
        phase1, phase2, phase3 = create_curriculum_splits(train_examples)
        phases = [
            ("Phase 1 (Common types)", phase1),
            ("Phase 2 (Medium types)", phase1 + phase2),
            ("Phase 3 (All types)", train_examples),
        ]
    else:
        phases = [("Standard training", train_examples)]
        epochs_per_phase = 3
    
    val_dataset = EnhancedT5Dataset(val_examples, tokenizer, max_input_length, max_output_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    for phase_name, phase_examples in phases:
        logger.info(f"\n{phase_name}: {len(phase_examples)} examples")
        
        train_dataset = EnhancedT5Dataset(phase_examples, tokenizer, max_input_length, max_output_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs_per_phase):
            model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"{phase_name} Epoch {epoch + 1}")
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"{phase_name} Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")
            
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
                        labels=labels
                    )
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"{phase_name} Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}")
    
    logger.info(f"\nSaving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    import sys
    import argparse
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    import importlib
    data_loader = importlib.import_module("2_claim_parsing_agent.data_loader")
    models = importlib.import_module("2_claim_parsing_agent.models")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True, help="Path to training JSON")
    parser.add_argument("--val-data", required=True, help="Path to validation JSON")
    parser.add_argument("--output-dir", default="models/enhanced_t5_parser")
    parser.add_argument("--model-name", default="t5-base")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs-per-phase", type=int, default=2)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    
    args = parser.parse_args()
    
    with open(args.train_data, "r") as f:
        train_parsed = [models.ParsedClaim(**item) for item in json.load(f)]
    
    with open(args.val_data, "r") as f:
        val_parsed = [models.ParsedClaim(**item) for item in json.load(f)]
    
    messages = data_loader.load_all_messages()
    claims = data_loader.load_claim_spans()
    
    logger.info("Preparing training data...")
    train_examples = prepare_enhanced_training_data(messages, claims, train_parsed)
    val_examples = prepare_enhanced_training_data(messages, claims, val_parsed)
    
    logger.info(f"Training examples: {len(train_examples)}")
    logger.info(f"Validation examples: {len(val_examples)}")
    
    train_enhanced_t5(
        train_examples=train_examples,
        val_examples=val_examples,
        model_name=args.model_name,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs_per_phase=args.epochs_per_phase,
        use_curriculum=not args.no_curriculum,
        use_augmentation=not args.no_augmentation,
    )

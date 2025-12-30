from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .models import MessageRecord, ClaimSpan, ParsedClaim
from .parser_base import ClaimParser
from .schemas import CLAIM_TYPE_SCHEMAS


class EnhancedT5Parser(ClaimParser):
    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_input_length: int = 256,
        max_output_length: int = 256,
    ):
        self.device = device
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def parse(self, message: MessageRecord, claims: list[ClaimSpan]) -> list[ParsedClaim]:
        if not claims:
            return []
        
        results = []
        
        for claim in claims:
            input_text = format_enhanced_input(
                message_text=message.text,
                claim_text=claim.text,
                claim_type=claim.claim_type,
            )
            
            input_ids = self.tokenizer.encode(
                input_text,
                max_length=self.max_input_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=self.max_output_length,
                    num_beams=4,
                    early_stopping=True,
                )
            
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            canonical_form, slots = parse_enhanced_output(output_text)
            
            results.append(ParsedClaim(
                message_id=message.message_id,
                message_label=message.label,
                claim_id=claim.claim_id,
                claim_type=claim.claim_type,
                claim_text=claim.text,
                canonical_form=canonical_form,
                slots=slots,
            ))
        
        return results
    
    def parse_batch(
        self,
        messages: list[MessageRecord],
        claims_by_message: dict[str, list[ClaimSpan]],
        **kwargs
    ) -> dict[str, list[ParsedClaim]]:
        results = {}
        for message in messages:
            claims = claims_by_message.get(message.message_id, [])
            if claims:
                results[message.message_id] = self.parse(message, claims)
        return results


def format_enhanced_input(message_text: str, claim_text: str, claim_type: str) -> str:
    schema = CLAIM_TYPE_SCHEMAS.get(claim_type, {})
    slot_names = list(schema.keys())
    slots_str = ", ".join(slot_names) if slot_names else "none"
    
    return f"[{claim_type}] [SLOTS: {slots_str}] Message: {message_text} | Claim: {claim_text}"


def parse_enhanced_output(output_text: str) -> tuple[str, dict]:
    try:
        if " | " in output_text:
            parts = output_text.split(" | ", 1)
            canonical = parts[0].replace("canonical:", "").strip()
            slots_part = parts[1].replace("slots:", "").strip() if len(parts) > 1 else "{}"
        else:
            canonical = output_text.strip()
            slots_part = "{}"
        
        try:
            slots = json.loads(slots_part)
        except:
            slots = {}
        
        return canonical, slots
    except:
        return output_text.strip(), {}


def prepare_enhanced_training_data(
    messages: list[MessageRecord],
    claims: list[ClaimSpan],
    parsed_claims: list[ParsedClaim],
) -> list[dict]:
    message_map = {m.message_id: m for m in messages}
    claim_map = {(c.message_id, c.claim_id): c for c in claims}
    
    training_examples = []
    
    for parsed in parsed_claims:
        message = message_map.get(parsed.message_id)
        claim = claim_map.get((parsed.message_id, parsed.claim_id))
        
        if not message or not claim:
            continue
        
        input_text = format_enhanced_input(
            message_text=message.text,
            claim_text=claim.text,
            claim_type=claim.claim_type,
        )
        
        output_text = f"canonical: {parsed.canonical_form} | slots: {json.dumps(parsed.slots)}"
        
        training_examples.append({
            "input": input_text,
            "output": output_text,
            "claim_type": parsed.claim_type,
            "message_id": parsed.message_id,
        })
    
    return training_examples


def augment_training_data(examples: list[dict], augmentation_factor: int = 2) -> list[dict]:
    augmented = list(examples)
    
    slot_substitutions = {
        "Amazon": ["PayPal", "eBay", "Walmart", "Target"],
        "PayPal": ["Amazon", "Venmo", "Cash App"],
        "package": ["parcel", "delivery", "shipment", "order"],
        "account": ["profile", "membership", "subscription"],
        "£": ["$", "€", "¥"],
        "urgent": ["immediate", "critical", "important"],
    }
    
    for _ in range(len(examples) * (augmentation_factor - 1)):
        original = random.choice(examples)
        
        augmented_input = original["input"]
        augmented_output = original["output"]
        
        for key, replacements in slot_substitutions.items():
            if key in augmented_input:
                replacement = random.choice(replacements)
                augmented_input = augmented_input.replace(key, replacement)
                augmented_output = augmented_output.replace(key, replacement)
        
        augmented.append({
            "input": augmented_input,
            "output": augmented_output,
            "claim_type": original["claim_type"],
            "message_id": original["message_id"] + "_aug",
        })
    
    return augmented


def create_curriculum_splits(examples: list[dict]) -> tuple[list, list, list]:
    common_types = ["ACTION_CLAIM", "URGENCY_CLAIM", "REWARD_CLAIM"]
    medium_types = ["FINANCIAL_CLAIM", "ACCOUNT_CLAIM", "DELIVERY_CLAIM", "VERIFICATION_CLAIM"]
    rare_types = ["IDENTITY_CLAIM", "SOCIAL_CLAIM", "LEGAL_CLAIM", "SECURITY_CLAIM", "CREDENTIALS_CLAIM"]
    
    phase1 = [ex for ex in examples if ex["claim_type"] in common_types]
    phase2 = [ex for ex in examples if ex["claim_type"] in medium_types]
    phase3 = [ex for ex in examples if ex["claim_type"] in rare_types]
    
    return phase1, phase2, phase3


class EnhancedT5Dataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_input_length: int, max_output_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        input_encoding = self.tokenizer(
            example["input"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            example["output"],
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_encoding.input_ids.flatten(),
            "attention_mask": input_encoding.attention_mask.flatten(),
            "labels": labels.flatten(),
        }

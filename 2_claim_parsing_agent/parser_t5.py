from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .models import ClaimSpan, MessageRecord, ParsedClaim

logger = logging.getLogger(__name__)


class T5ClaimParser:
    def __init__(
        self,
        model_path: str | Path,
        max_input_length: int = 256,
        max_output_length: int = 256,
    ):
        self.model_path = Path(model_path)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        self.tokenizer = T5Tokenizer.from_pretrained(str(self.model_path))
        self.model = T5ForConditionalGeneration.from_pretrained(str(self.model_path))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded T5 parser from {self.model_path} on {self.device}")
    
    def parse_message(
        self,
        message: MessageRecord,
        claim_spans: list[ClaimSpan],
    ) -> list[ParsedClaim]:
        if not claim_spans:
            return []
        
        results = []
        
        for claim in claim_spans:
            input_text = self._format_input(message, claim)
            output_text = self._generate(input_text)
            parsed_claim = self._parse_output(output_text, message, claim)
            results.append(parsed_claim)
        
        return results
    
    def _format_input(self, message: MessageRecord, claim: ClaimSpan) -> str:
        return f"CLAIM_TYPE: {claim.claim_type}; TEXT: {claim.text}; CONTEXT: {message.text}"
    
    def _generate(self, input_text: str) -> str:
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=4,
                early_stopping=True,
            )
        
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text
    
    def _parse_output(
        self,
        output_text: str,
        message: MessageRecord,
        claim: ClaimSpan,
    ) -> ParsedClaim:
        canonical_form = ""
        slots = {}
        
        canonical_match = re.search(r"canonical=([^;]+)", output_text)
        if canonical_match:
            canonical_form = canonical_match.group(1).strip()
        
        slot_pattern = r"(\w+)=([^;]+)"
        for match in re.finditer(slot_pattern, output_text):
            slot_name = match.group(1).strip()
            slot_value = match.group(2).strip()
            
            if slot_name != "canonical":
                if slot_value.lower() in ["null", "none", "n/a"]:
                    slots[slot_name] = None
                else:
                    slots[slot_name] = slot_value
        
        return ParsedClaim(
            message_id=message.message_id,
            message_label=message.label,
            claim_id=claim.claim_id,
            claim_type=claim.claim_type,
            canonical_form=canonical_form,
            slots=slots,
        )


def prepare_training_data(
    messages: list[MessageRecord],
    claims: list[ClaimSpan],
    parsed_claims: list[ParsedClaim],
) -> list[dict[str, str]]:
    parsed_by_key = {}
    for pc in parsed_claims:
        key = (pc.message_id, pc.claim_id)
        parsed_by_key[key] = pc
    
    message_by_id = {m.message_id: m for m in messages}
    
    training_examples = []
    
    for claim in claims:
        key = (claim.message_id, claim.claim_id)
        parsed = parsed_by_key.get(key)
        
        if not parsed:
            continue
        
        message = message_by_id.get(claim.message_id)
        if not message:
            continue
        
        input_text = f"CLAIM_TYPE: {claim.claim_type}; TEXT: {claim.text}; CONTEXT: {message.text}"
        
        output_parts = [f"canonical={parsed.canonical_form}"]
        for slot_name, slot_value in sorted(parsed.slots.items()):
            if slot_value is not None:
                output_parts.append(f"{slot_name}={slot_value}")
        output_text = "; ".join(output_parts)
        
        training_examples.append({
            "input_text": input_text,
            "target_text": output_text,
        })
    
    logger.info(f"Prepared {len(training_examples)} training examples")
    return training_examples

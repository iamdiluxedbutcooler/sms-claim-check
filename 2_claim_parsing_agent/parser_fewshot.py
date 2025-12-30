from __future__ import annotations

import json
from typing import Protocol
from sentence_transformers import SentenceTransformer

from .models import MessageRecord, ClaimSpan, ParsedClaim
from .schemas import format_schema_for_prompt
from .parser_base import ClaimParser


class FewShotGPTParser(ClaimParser):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 2000,
        temperature: float = 0.0,
        num_examples: int = 3,
        training_data: list[ParsedClaim] = None,
    ):
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_examples = num_examples
        
        self.training_data = training_data or []
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if self.training_data:
            self._build_example_index()
    
    def _build_example_index(self):
        self.example_texts = []
        self.example_data = []
        
        for example in self.training_data:
            self.example_texts.append(example.claim_text)
            self.example_data.append(example)
        
        if self.example_texts:
            self.example_embeddings = self.embedding_model.encode(
                self.example_texts,
                convert_to_tensor=False,
                show_progress_bar=False
            )
    
    def _retrieve_examples(self, claim_text: str, claim_type: str, k: int = 3):
        if not hasattr(self, 'example_embeddings'):
            return []
        
        type_examples = [
            (i, ex) for i, ex in enumerate(self.example_data)
            if ex.claim_type == claim_type
        ]
        
        if not type_examples:
            return []
        
        query_emb = self.embedding_model.encode([claim_text], convert_to_tensor=False)[0]
        
        type_indices = [i for i, _ in type_examples]
        type_embeddings = [self.example_embeddings[i] for i in type_indices]
        
        import numpy as np
        similarities = np.dot(type_embeddings, query_emb)
        
        top_k_local = min(k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k_local:][::-1]
        
        return [self.example_data[type_indices[i]] for i in top_indices]
    
    def parse(self, message: MessageRecord, claims: list[ClaimSpan]) -> list[ParsedClaim]:
        if not claims:
            return []
        
        schemas_text = "\n\n".join([format_schema_for_prompt(c.claim_type) for c in claims])
        
        claims_text = "\n\n".join([
            f"Claim {idx + 1}:\n"
            f"  - Type: {claim.claim_type}\n"
            f"  - Text: \"{claim.text}\"\n"
            f"  - Position: [{claim.start}:{claim.end}]"
            for idx, claim in enumerate(claims)
        ])
        
        examples_text = ""
        if self.training_data:
            for idx, claim in enumerate(claims):
                retrieved = self._retrieve_examples(claim.text, claim.claim_type, self.num_examples)
                if retrieved:
                    examples_text += f"\n\nExamples for {claim.claim_type}:\n"
                    for ex_idx, ex in enumerate(retrieved[:self.num_examples]):
                        examples_text += f"{ex_idx + 1}. \"{ex.claim_text}\" → "
                        examples_text += f"canonical: \"{ex.canonical_form}\", "
                        examples_text += f"slots: {json.dumps(ex.slots)}\n"
        
        prompt = f"""You are a claim parsing expert. Parse each claim into a canonical form and extract structured slot values.

SMS Message: "{message.text}"

Extracted Claims:
{claims_text}

Slot Schemas:
{schemas_text}
{examples_text}

Return ONLY a valid JSON array of objects with this structure:
{{
  "claim_id": <claim number 1-indexed>,
  "canonical_form": "<canonical statement>",
  "slots": {{"<slot_name>": "<value>", ...}}
}}

Rules:
- Only include slots explicitly present or inferable
- Use null for undetermined slots
- Keep slot values concise and normalized
- Ensure canonical form is complete and understandable

Return the JSON array now:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            parsed_data = json.loads(content)
        except json.JSONDecodeError:
            return []
        
        if not isinstance(parsed_data, list):
            return []
        
        results = []
        for item in parsed_data:
            claim_idx = item.get("claim_id", 0) - 1
            
            if claim_idx < 0 or claim_idx >= len(claims):
                continue
            
            original_claim = claims[claim_idx]
            
            results.append(ParsedClaim(
                message_id=message.message_id,
                message_label=message.label,
                claim_id=original_claim.claim_id,
                claim_type=original_claim.claim_type,
                claim_text=original_claim.text,
                canonical_form=item.get("canonical_form", ""),
                slots=item.get("slots", {}),
            ))
        
        return results
    
    def parse_batch(
        self,
        messages: list[MessageRecord],
        claims_by_message: dict[str, list[ClaimSpan]],
        sleep_between_calls: float = 0.5,
    ) -> dict[str, list[ParsedClaim]]:
        from .gpt_labeling import label_batch_with_gpt
        
        return label_batch_with_gpt(
            messages=messages,
            claims_by_message=claims_by_message,
            api_key=self.client.api_key,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            sleep_between_calls=sleep_between_calls,
            parser_instance=self,
        )

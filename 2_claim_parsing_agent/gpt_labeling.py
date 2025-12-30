from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI

from .models import ClaimSpan, MessageRecord, ParsedClaim
from .schemas import format_schema_for_prompt

logger = logging.getLogger(__name__)


def label_claims_with_gpt(
    message: MessageRecord,
    claims: list[ClaimSpan],
    client: OpenAI,
    model: str = "gpt-4o-mini",
    max_tokens: int = 2000,
    temperature: float = 0.0,
) -> list[ParsedClaim]:
    if not claims:
        return []
    
    schemas_text = "\n\n".join([format_schema_for_prompt(c.claim_type) for c in claims])
    
    claims_list = []
    for idx, claim in enumerate(claims):
        claims_list.append(
            f"Claim {idx + 1}:\n"
            f"  - Type: {claim.claim_type}\n"
            f"  - Text: \"{claim.text}\"\n"
            f"  - Position: [{claim.start}:{claim.end}]"
        )
    claims_text = "\n\n".join(claims_list)
    
    prompt = f"""You are a claim parsing expert. Given an SMS message and extracted claim spans, 
parse each claim into a canonical form (a clear, standalone statement) and extract structured slot values.

SMS Message:
"{message.text}"

Extracted Claims:
{claims_text}

Slot Schemas:
{schemas_text}

For EACH claim, provide:
1. canonical_form: A clear, standalone statement of what is being claimed
2. slots: A dictionary of slot values extracted from the claim text and message context

Return ONLY a valid JSON array of objects, each with this structure:
{{
  "claim_id": <claim number 1-indexed>,
  "canonical_form": "<canonical statement>",
  "slots": {{"<slot_name>": "<value>", ...}}
}}

Rules:
- Only include slots that are explicitly present or can be inferred from the message
- Use null for slots that cannot be determined
- Keep slot values concise and normalized
- Ensure the canonical form is a complete, understandable statement

Return the JSON array now:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        parsed_data = json.loads(content)
        
        if not isinstance(parsed_data, list):
            logger.error(f"Expected list, got {type(parsed_data)}")
            return []
        
        results = []
        for item in parsed_data:
            claim_idx = item.get("claim_id", 0) - 1
            
            if claim_idx < 0 or claim_idx >= len(claims):
                logger.warning(f"Invalid claim_id {item.get('claim_id')} for message {message.message_id}")
                continue
            
            original_claim = claims[claim_idx]
            
            results.append(
                ParsedClaim(
                    message_id=message.message_id,
                    message_label=message.label,
                    claim_id=original_claim.claim_id,
                    claim_type=original_claim.claim_type,
                    canonical_form=item.get("canonical_form", ""),
                    slots=item.get("slots", {}),
                )
            )
        
        return results
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON for message {message.message_id}: {e}")
        logger.error(f"Response content: {content[:500]}")
        return []
    
    except Exception as e:
        logger.error(f"Error labeling claims for message {message.message_id}: {e}")
        return []


def label_batch_with_gpt(
    messages: list[MessageRecord],
    claims_by_message: dict[str, list[ClaimSpan]],
    client: OpenAI,
    model: str = "gpt-4o-mini",
    max_tokens: int = 2000,
    temperature: float = 0.0,
    sleep_between_calls: float = 0.5,
) -> dict[str, list[ParsedClaim]]:
    results = {}
    
    total = len(messages)
    logger.info(f"Starting batch labeling for {total} messages")
    
    for idx, message in enumerate(messages, 1):
        msg_claims = claims_by_message.get(message.message_id, [])
        
        if not msg_claims:
            results[message.message_id] = []
            continue
        
        logger.info(f"Processing {idx}/{total}: {message.message_id} ({len(msg_claims)} claims)")
        
        parsed_claims = label_claims_with_gpt(
            message=message,
            claims=msg_claims,
            client=client,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        results[message.message_id] = parsed_claims
        
        if idx < total:
            time.sleep(sleep_between_calls)
    
    logger.info(f"Batch labeling complete: {len(results)} messages processed")
    
    return results

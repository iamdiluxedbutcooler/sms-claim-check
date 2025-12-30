from __future__ import annotations

import logging

from openai import OpenAI

from .gpt_labeling import label_claims_with_gpt
from .models import ClaimSpan, MessageRecord, ParsedClaim

logger = logging.getLogger(__name__)


class GPTClaimParser:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def parse_message(
        self,
        message: MessageRecord,
        claim_spans: list[ClaimSpan],
    ) -> list[ParsedClaim]:
        return label_claims_with_gpt(
            message=message,
            claims=claim_spans,
            client=self.client,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
    
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
            client=self.client,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            sleep_between_calls=sleep_between_calls,
        )

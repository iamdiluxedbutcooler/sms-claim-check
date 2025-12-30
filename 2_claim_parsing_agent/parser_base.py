from __future__ import annotations

from typing import Protocol

from .models import ClaimSpan, MessageRecord, ParsedClaim


class ClaimParser(Protocol):
    def parse_message(
        self,
        message: MessageRecord,
        claim_spans: list[ClaimSpan],
    ) -> list[ParsedClaim]:
        ...

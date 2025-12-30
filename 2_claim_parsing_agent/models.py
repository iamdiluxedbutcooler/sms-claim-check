from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MessageRecord:
    message_id: str
    text: str
    label: str | None = None
    split: str | None = None


@dataclass
class ClaimSpan:
    message_id: str
    message_label: str | None
    claim_id: int
    claim_type: str
    text: str
    start: int
    end: int
    confidence: float | None = None


@dataclass
class ParsedClaim:
    message_id: str
    message_label: str | None
    claim_id: int
    claim_type: str
    canonical_form: str
    slots: dict[str, str | int | float | None] = field(default_factory=dict)
    claim_text: str = ""


@dataclass
class VerificationPlan:
    message_id: str
    claim_id: int
    claim_type: str
    questions: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)

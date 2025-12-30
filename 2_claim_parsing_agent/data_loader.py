from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import CLAIM_ANNOTATIONS_FILE
from .models import ClaimSpan, MessageRecord

logger = logging.getLogger(__name__)


def load_annotations_file() -> dict:
    if not CLAIM_ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Annotations file not found: {CLAIM_ANNOTATIONS_FILE}\n"
            f"Expected path: {CLAIM_ANNOTATIONS_FILE.absolute()}"
        )
    
    with open(CLAIM_ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} records from {CLAIM_ANNOTATIONS_FILE}")
    return data


def load_all_messages() -> list[MessageRecord]:
    data = load_annotations_file()
    messages = []
    
    for entry in data:
        message_id = entry.get("id", "")
        text = entry.get("data", {}).get("text", "")
        
        # Label is in meta, not data.meta
        meta = entry.get("meta", {})
        label = meta.get("label")
        split = meta.get("split")
        
        messages.append(
            MessageRecord(
                message_id=message_id,
                text=text,
                label=label,
                split=split,
            )
        )
    
    logger.info(f"Loaded {len(messages)} messages")
    return messages


def load_claim_spans() -> list[ClaimSpan]:
    data = load_annotations_file()
    claims = []
    
    for entry in data:
        message_id = entry.get("id", "")
        text = entry.get("data", {}).get("text", "")
        meta = entry.get("meta", {})
        label = meta.get("label")
        
        annotations = entry.get("annotations", [])
        if not annotations:
            continue
        
        annotation = annotations[0]
        results = annotation.get("result", [])
        
        for idx, result in enumerate(results):
            value = result.get("value", {})
            labels_list = value.get("labels", [])
            
            if not labels_list:
                continue
            
            claim_type = labels_list[0]
            claim_text = value.get("text", "")
            start = value.get("start", 0)
            end = value.get("end", 0)
            
            claims.append(
                ClaimSpan(
                    message_id=message_id,
                    message_label=label,
                    claim_id=idx,
                    claim_type=claim_type,
                    text=claim_text,
                    start=start,
                    end=end,
                    confidence=None,
                )
            )
    
    logger.info(f"Loaded {len(claims)} claim spans")
    return claims


def get_claims_for_message(message_id: str) -> list[ClaimSpan]:
    all_claims = load_claim_spans()
    return [c for c in all_claims if c.message_id == message_id]


def get_messages_by_split(split: str) -> list[MessageRecord]:
    all_messages = load_all_messages()
    return [m for m in all_messages if m.split == split]


def get_message_text(message_id: str) -> str:
    all_messages = load_all_messages()
    for msg in all_messages:
        if msg.message_id == message_id:
            return msg.text
    raise ValueError(f"Message not found: {message_id}")


def get_messages_by_ids(message_ids: list[str]) -> list[MessageRecord]:
    all_messages = load_all_messages()
    id_set = set(message_ids)
    return [m for m in all_messages if m.message_id in id_set]


def get_claims_for_messages(message_ids: list[str]) -> list[ClaimSpan]:
    all_claims = load_claim_spans()
    id_set = set(message_ids)
    return [c for c in all_claims if c.message_id in id_set]

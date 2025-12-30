from __future__ import annotations

import logging
import random
from collections import defaultdict

from .models import ClaimSpan, MessageRecord

logger = logging.getLogger(__name__)


def select_messages_for_parsing_gold_set(
    messages: list[MessageRecord],
    claims: list[ClaimSpan],
    target_num_messages: int = 300,
    min_per_rare_type: dict[str, int] | None = None,
    random_seed: int | None = 42,
) -> list[str]:
    if min_per_rare_type is None:
        min_per_rare_type = {
            "DELIVERY_CLAIM": 10,
            "VERIFICATION_CLAIM": 10,
            "SOCIAL_CLAIM": 5,
            "IDENTITY_CLAIM": 10,
            "LEGAL_CLAIM": 5,
            "SECURITY_CLAIM": 10,
            "CREDENTIALS_CLAIM": 5,
        }
    
    rng = random.Random(random_seed)
    
    message_id_to_label = {m.message_id: m.label for m in messages}
    
    message_claims = defaultdict(list)
    for claim in claims:
        if claim.message_id in message_id_to_label:
            message_claims[claim.message_id].append(claim)
    
    claim_type_to_messages = defaultdict(set)
    for msg_id, msg_claims in message_claims.items():
        for claim in msg_claims:
            claim_type_to_messages[claim.claim_type].add(msg_id)
    
    selected_messages = set()
    
    for claim_type, min_count in min_per_rare_type.items():
        messages_with_type = list(claim_type_to_messages.get(claim_type, set()))
        
        if not messages_with_type:
            logger.warning(f"No messages found with claim type: {claim_type}")
            continue
        
        needed = min_count
        available_messages = [m for m in messages_with_type if m not in selected_messages]
        
        if len(available_messages) < needed:
            logger.warning(
                f"Only {len(available_messages)} messages available for {claim_type}, "
                f"need {needed}"
            )
            selected_messages.update(available_messages)
        else:
            sampled = rng.sample(available_messages, needed)
            selected_messages.update(sampled)
            logger.info(f"Selected {len(sampled)} messages for {claim_type}")
    
    remaining_slots = target_num_messages - len(selected_messages)
    
    if remaining_slots > 0:
        ham_messages = [m.message_id for m in messages 
                       if m.label == "ham" and m.message_id not in selected_messages]
        smish_messages = [m.message_id for m in messages 
                         if m.label == "smish" and m.message_id not in selected_messages]
        
        ham_needed = remaining_slots // 2
        smish_needed = remaining_slots - ham_needed
        
        if len(ham_messages) >= ham_needed:
            selected_messages.update(rng.sample(ham_messages, ham_needed))
        else:
            selected_messages.update(ham_messages)
            logger.warning(f"Only {len(ham_messages)} ham messages available, needed {ham_needed}")
        
        if len(smish_messages) >= smish_needed:
            selected_messages.update(rng.sample(smish_messages, smish_needed))
        else:
            selected_messages.update(smish_messages)
            logger.warning(f"Only {len(smish_messages)} smish messages available, needed {smish_needed}")
    
    selected_list = list(selected_messages)
    
    logger.info(f"Selected {len(selected_list)} messages for parsing gold set")
    
    ham_count = sum(1 for m in messages if m.message_id in selected_list and m.label == "ham")
    smish_count = sum(1 for m in messages if m.message_id in selected_list and m.label == "smish")
    logger.info(f"Label distribution: {ham_count} ham, {smish_count} smish")
    
    selected_claims = [c for c in claims if c.message_id in selected_list]
    claim_type_counts = defaultdict(int)
    for claim in selected_claims:
        claim_type_counts[claim.claim_type] += 1
    
    logger.info(f"Total claims in selected messages: {len(selected_claims)}")
    for claim_type, count in sorted(claim_type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {claim_type}: {count}")
    
    return selected_list

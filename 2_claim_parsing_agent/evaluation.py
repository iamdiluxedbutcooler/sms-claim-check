from __future__ import annotations

import logging
from collections import defaultdict

from .models import ParsedClaim

logger = logging.getLogger(__name__)


def align_claims(
    gold: list[ParsedClaim],
    pred: list[ParsedClaim],
) -> list[tuple[ParsedClaim | None, ParsedClaim | None]]:
    gold_by_key = {}
    for gc in gold:
        key = (gc.message_id, gc.claim_id, gc.claim_type)
        gold_by_key[key] = gc
    
    pred_by_key = {}
    for pc in pred:
        key = (pc.message_id, pc.claim_id, pc.claim_type)
        pred_by_key[key] = pc
    
    all_keys = set(gold_by_key.keys()) | set(pred_by_key.keys())
    
    aligned = []
    for key in all_keys:
        gold_claim = gold_by_key.get(key)
        pred_claim = pred_by_key.get(key)
        aligned.append((gold_claim, pred_claim))
    
    return aligned


def evaluate_parsing(gold: list[ParsedClaim], pred: list[ParsedClaim]) -> dict[str, float]:
    aligned = align_claims(gold, pred)
    
    total_gold_slots = 0
    total_pred_slots = 0
    total_correct_slots = 0
    
    canonical_exact_match = 0
    total_canonical = 0
    
    by_type = defaultdict(lambda: {"gold": 0, "pred": 0, "correct": 0})
    
    for gold_claim, pred_claim in aligned:
        if gold_claim is None:
            if pred_claim:
                total_pred_slots += len(pred_claim.slots)
                by_type[pred_claim.claim_type]["pred"] += len(pred_claim.slots)
            continue
        
        if pred_claim is None:
            total_gold_slots += len(gold_claim.slots)
            by_type[gold_claim.claim_type]["gold"] += len(gold_claim.slots)
            total_canonical += 1
            continue
        
        gold_slots = gold_claim.slots
        pred_slots = pred_claim.slots
        
        claim_type = gold_claim.claim_type
        
        total_gold_slots += len(gold_slots)
        total_pred_slots += len(pred_slots)
        
        by_type[claim_type]["gold"] += len(gold_slots)
        by_type[claim_type]["pred"] += len(pred_slots)
        
        for slot_name, gold_value in gold_slots.items():
            if slot_name in pred_slots:
                pred_value = pred_slots[slot_name]
                
                if _values_match(gold_value, pred_value):
                    total_correct_slots += 1
                    by_type[claim_type]["correct"] += 1
        
        if gold_claim.canonical_form and pred_claim.canonical_form:
            total_canonical += 1
            if gold_claim.canonical_form.strip().lower() == pred_claim.canonical_form.strip().lower():
                canonical_exact_match += 1
    
    precision = total_correct_slots / total_pred_slots if total_pred_slots > 0 else 0.0
    recall = total_correct_slots / total_gold_slots if total_gold_slots > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    canonical_match_rate = canonical_exact_match / total_canonical if total_canonical > 0 else 0.0
    
    results = {
        "slot_precision_overall": precision,
        "slot_recall_overall": recall,
        "slot_f1_overall": f1,
        "canonical_exact_match": canonical_match_rate,
        "total_gold_slots": total_gold_slots,
        "total_pred_slots": total_pred_slots,
        "total_correct_slots": total_correct_slots,
    }
    
    for claim_type, counts in by_type.items():
        gold_count = counts["gold"]
        pred_count = counts["pred"]
        correct_count = counts["correct"]
        
        type_precision = correct_count / pred_count if pred_count > 0 else 0.0
        type_recall = correct_count / gold_count if gold_count > 0 else 0.0
        type_f1 = (
            2 * type_precision * type_recall / (type_precision + type_recall)
            if (type_precision + type_recall) > 0
            else 0.0
        )
        
        results[f"slot_f1_{claim_type}"] = type_f1
        results[f"slot_precision_{claim_type}"] = type_precision
        results[f"slot_recall_{claim_type}"] = type_recall
    
    return results


def _values_match(gold_value: any, pred_value: any) -> bool:
    if gold_value is None and pred_value is None:
        return True
    
    if gold_value is None or pred_value is None:
        return False
    
    gold_str = str(gold_value).strip().lower()
    pred_str = str(pred_value).strip().lower()
    
    return gold_str == pred_str


def print_evaluation_summary(metrics: dict[str, float]):
    print("=" * 80)
    print("CLAIM PARSING EVALUATION SUMMARY")
    print("=" * 80)
    
    print("\nOverall Slot-Level Metrics:")
    print(f"  Precision: {metrics.get('slot_precision_overall', 0.0):.3f}")
    print(f"  Recall:    {metrics.get('slot_recall_overall', 0.0):.3f}")
    print(f"  F1 Score:  {metrics.get('slot_f1_overall', 0.0):.3f}")
    
    print(f"\n  Total Gold Slots:    {metrics.get('total_gold_slots', 0)}")
    print(f"  Total Pred Slots:    {metrics.get('total_pred_slots', 0)}")
    print(f"  Total Correct Slots: {metrics.get('total_correct_slots', 0)}")
    
    print(f"\nCanonical Form Exact Match: {metrics.get('canonical_exact_match', 0.0):.3f}")
    
    print("\nPer-Claim-Type F1 Scores:")
    
    type_f1s = [(k, v) for k, v in metrics.items() if k.startswith("slot_f1_") and not k.endswith("_overall")]
    type_f1s.sort(key=lambda x: -x[1])
    
    for key, f1 in type_f1s:
        claim_type = key.replace("slot_f1_", "")
        precision = metrics.get(f"slot_precision_{claim_type}", 0.0)
        recall = metrics.get(f"slot_recall_{claim_type}", 0.0)
        print(f"  {claim_type:25s}  P: {precision:.3f}  R: {recall:.3f}  F1: {f1:.3f}")
    
    print("=" * 80)

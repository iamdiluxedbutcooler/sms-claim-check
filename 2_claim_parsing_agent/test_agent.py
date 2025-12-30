#!/usr/bin/env python3
"""
Demo script to test Agent 2 components
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib

data_loader = importlib.import_module("2_claim_parsing_agent.data_loader")
schemas = importlib.import_module("2_claim_parsing_agent.schemas")
gold_sampling = importlib.import_module("2_claim_parsing_agent.gold_sampling")

def test_data_loading():
    print("=" * 80)
    print("TEST 1: Data Loading")
    print("=" * 80)
    
    messages = data_loader.load_all_messages()
    print(f"✓ Loaded {len(messages)} messages")
    
    claims = data_loader.load_claim_spans()
    print(f"✓ Loaded {len(claims)} claim spans")
    
    test_messages = data_loader.get_messages_by_split("test")
    print(f"✓ Found {len(test_messages)} test messages")
    
    train_messages = data_loader.get_messages_by_split("train")
    print(f"✓ Found {len(train_messages)} train messages")
    
    if messages:
        sample_msg = messages[0]
        print(f"\nSample message:")
        print(f"  ID: {sample_msg.message_id}")
        print(f"  Text: {sample_msg.text[:100]}...")
        print(f"  Label: {sample_msg.label}")
        print(f"  Split: {sample_msg.split}")
    
    if claims:
        sample_claim = claims[0]
        print(f"\nSample claim:")
        print(f"  Message ID: {sample_claim.message_id}")
        print(f"  Claim Type: {sample_claim.claim_type}")
        print(f"  Text: {sample_claim.text}")


def test_schemas():
    print("\n" + "=" * 80)
    print("TEST 2: Claim Schemas")
    print("=" * 80)
    
    for claim_type in schemas.CLAIM_TYPE_SCHEMAS.keys():
        slot_schema = schemas.get_slot_schema(claim_type)
        print(f"\n{claim_type}:")
        for slot in slot_schema:
            print(f"  - {slot.name}: {slot.description}")


def test_gold_sampling():
    print("\n" + "=" * 80)
    print("TEST 3: Gold Set Sampling")
    print("=" * 80)
    
    test_messages = data_loader.get_messages_by_split("test")
    all_claims = data_loader.load_claim_spans()
    test_claims = [c for c in all_claims if c.message_id in {m.message_id for m in test_messages}]
    
    print(f"Test set: {len(test_messages)} messages, {len(test_claims)} claims")
    
    selected_ids = gold_sampling.select_messages_for_parsing_gold_set(
        messages=test_messages,
        claims=test_claims,
        target_num_messages=50,
    )
    
    print(f"\n✓ Selected {len(selected_ids)} messages for parsing gold set")
    
    selected_claims = [c for c in test_claims if c.message_id in selected_ids]
    print(f"✓ Contains {len(selected_claims)} claims")
    
    ham_count = sum(1 for m in test_messages if m.message_id in selected_ids and m.label == "ham")
    smish_count = sum(1 for m in test_messages if m.message_id in selected_ids and m.label == "smish")
    print(f"✓ Label distribution: {ham_count} ham, {smish_count} smish")


def main():
    print("\n" + "=" * 80)
    print("CLAIM PARSING AGENT (Agent 2) - Component Tests")
    print("=" * 80 + "\n")
    
    try:
        test_data_loading()
        test_schemas()
        test_gold_sampling()
        
        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80 + "\n")
        
        print("Next steps:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Run parsing experiment:")
        print("   python scripts/run_parsing_experiment.py --parser-type gpt --target-num-messages 300 --save-predictions")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

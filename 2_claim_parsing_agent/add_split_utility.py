#!/usr/bin/env python3
"""
Utility to add train/test split information to messages.

Since the annotations file doesn't include split information,
this script creates a deterministic 80/20 split stratified by label.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
from sklearn.model_selection import train_test_split

data_loader = importlib.import_module("2_claim_parsing_agent.data_loader")


def add_splits_to_messages(random_seed: int = 42):
    """
    Adds train/test split to messages in-memory.
    
    Returns updated messages with split field set to "train" or "test".
    Uses 80/20 split stratified by label (ham/smish).
    """
    messages = data_loader.load_all_messages()
    
    # Separate by label
    ham = [m for m in messages if m.label == "ham"]
    smish = [m for m in messages if m.label == "smish"]
    other = [m for m in messages if m.label not in ["ham", "smish"]]
    
    print(f"Total messages: {len(messages)}")
    print(f"  Ham: {len(ham)}")
    print(f"  Smish: {len(smish)}")
    print(f"  Other/None: {len(other)}")
    
    # If no labels, fall back to random split
    if not ham and not smish:
        print("\nNo label information found - using random 80/20 split")
        train_messages, test_messages = train_test_split(
            messages, test_size=0.2, random_state=random_seed
        )
    else:
        # Stratified split for labeled messages
        ham_train, ham_test = train_test_split(ham, test_size=0.2, random_state=random_seed) if ham else ([], [])
        smish_train, smish_test = train_test_split(smish, test_size=0.2, random_state=random_seed) if smish else ([], [])
        
        # Other messages go to train
        train_messages = ham_train + smish_train + other
        test_messages = ham_test + smish_test
    
    # Assign splits
    for m in train_messages:
        m.split = "train"
    for m in test_messages:
        m.split = "test"
    
    all_messages = train_messages + test_messages
    
    print(f"\nSplit created:")
    print(f"  Train: {len(train_messages)} messages")
    print(f"  Test: {len(test_messages)} messages")
    
    # Show train/test distribution by label
    train_ham = sum(1 for m in train_messages if m.label == "ham")
    train_smish = sum(1 for m in train_messages if m.label == "smish")
    test_ham = sum(1 for m in test_messages if m.label == "ham")
    test_smish = sum(1 for m in test_messages if m.label == "smish")
    
    print(f"\nTrain distribution:")
    print(f"  Ham: {train_ham}")
    print(f"  Smish: {train_smish}")
    
    print(f"\nTest distribution:")
    print(f"  Ham: {test_ham}")
    print(f"  Smish: {test_smish}")
    
    return all_messages


def main():
    print("=" * 80)
    print("ADDING TRAIN/TEST SPLIT TO MESSAGES")
    print("=" * 80 + "\n")
    
    messages = add_splits_to_messages()
    
    print("\n" + "=" * 80)
    print("✓ Split created successfully!")
    print("=" * 80)
    
    print("\nNote: This split is created in-memory each time.")
    print("To use this in your code, call add_splits_to_messages() from data_loader.py")
    print("or modify data_loader.py to include this logic in load_all_messages().")


if __name__ == "__main__":
    main()

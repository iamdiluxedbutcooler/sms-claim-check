#!/usr/bin/env python3
"""Split all messages (ham + smish) into train/test sets"""

import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load all messages
annotations_path = Path("data/annotations/claim_annotations_2000_reviewed.json")
with open(annotations_path, "r") as f:
    all_messages = json.load(f)

print(f"Loaded {len(all_messages)} total messages")

# Create message_id -> label mapping
message_labels = {}
for entry in all_messages:
    msg_id = entry["id"]
    label = entry.get("meta", {}).get("label")
    if label is None:
        label = "smish"
    message_labels[msg_id] = label

# Split by label for balanced sampling
ham_ids = [mid for mid, lbl in message_labels.items() if lbl == "ham"]
smish_ids = [mid for mid, lbl in message_labels.items() if lbl != "ham"]

print(f"Total: {len(ham_ids)} ham, {len(smish_ids)} smish")

# 80/20 split on both ham and smish
ham_train, ham_test = train_test_split(ham_ids, test_size=0.2, random_state=42)
smish_train, smish_test = train_test_split(smish_ids, test_size=0.2, random_state=42)

train_ids = set(ham_train + smish_train)
test_ids = set(ham_test + smish_test)

print(f"Train: {len(train_ids)} messages ({len(ham_train)} ham, {len(smish_train)} smish)")
print(f"Test: {len(test_ids)} messages ({len(ham_test)} ham, {len(smish_test)} smish)")

# Load parsed claims
labels_path = Path("data/annotations/claim_parsing_all_2000.json")
with open(labels_path, "r") as f:
    all_parsed = json.load(f)

print(f"Total parsed claims: {len(all_parsed)}")

# Split parsed claims by message assignment
train_parsed = [p for p in all_parsed if p["message_id"] in train_ids]
test_parsed = [p for p in all_parsed if p["message_id"] in test_ids]

print(f"Train: {len(train_parsed)} parsed claims")
print(f"Test: {len(test_parsed)} parsed claims")

Path("data/annotations/claim_parsing_train_full.json").write_text(json.dumps(train_parsed, indent=2))
Path("data/annotations/claim_parsing_test_full.json").write_text(json.dumps(test_parsed, indent=2))

split_info = {
    "train_message_ids": [str(x) for x in train_ids],
    "test_message_ids": [str(x) for x in test_ids],
    "train_stats": {
        "total": len(train_ids),
        "ham": len(ham_train),
        "smish": len(smish_train),
        "claims": len(train_parsed)
    },
    "test_stats": {
        "total": len(test_ids),
        "ham": len(ham_test),
        "smish": len(smish_test),
        "claims": len(test_parsed)
    }
}
Path("data/annotations/split_info.json").write_text(json.dumps(split_info, indent=2))

print("\nSaved:")
print("  - data/annotations/claim_parsing_train_full.json")
print("  - data/annotations/claim_parsing_test_full.json")
print("  - data/annotations/split_info.json")

claim_counts = defaultdict(int)
for p in test_parsed:
    claim_counts[p["claim_type"]] += 1

print(f"\nTest set claim distribution:")
for claim_type, count in sorted(claim_counts.items(), key=lambda x: -x[1]):
    print(f"  {claim_type:25s}: {count:3d}")

print(f"\nNote: Train includes {len(ham_train)} ham messages with 0 claims")
print(f"      Test includes {len(ham_test)} ham messages with 0 claims")


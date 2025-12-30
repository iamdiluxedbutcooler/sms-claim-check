#!/usr/bin/env python3
"""Split all_gpt_labels.json into train/test sets"""

import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load all GPT labels
labels_path = Path("data/all_gpt_labels.json")
with open(labels_path, "r") as f:
    all_labels = json.load(f)

print(f"Loaded {len(all_labels)} parsed claims")

# Load original annotations to get correct labels
annotations_path = Path("data/annotations/claim_annotations_2000_reviewed.json")
with open(annotations_path, "r") as f:
    annotations = json.load(f)

# Create message_id -> label mapping from annotations
message_labels = {}
for entry in annotations:
    msg_id = entry["id"]
    label = entry.get("meta", {}).get("label")
    message_labels[msg_id] = label

# Get unique message IDs and their labels
message_info = {}
for label in all_labels:
    msg_id = label["message_id"]
    if msg_id not in message_info:
        # Get label from annotations, default to "smish" if not ham
        msg_label = message_labels.get(msg_id)
        if msg_label is None:
            msg_label = "smish"  # Assume unlabeled are smish
        message_info[msg_id] = msg_label

# Split by label
ham_ids = [mid for mid, lbl in message_info.items() if lbl == "ham"]
smish_ids = [mid for mid, lbl in message_info.items() if lbl != "ham"]

print(f"Messages: {len(ham_ids)} ham, {len(smish_ids)} smish")

# Only split smish messages (ham messages have no claims to parse)
if len(smish_ids) > 0:
    smish_train, smish_test = train_test_split(smish_ids, test_size=0.2, random_state=42)
    train_ids = set(smish_train)
    test_ids = set(smish_test)
else:
    train_ids = set()
    test_ids = set()

print(f"Train: {len(train_ids)} messages | Test: {len(test_ids)} messages")

# Split labels
train_labels = [l for l in all_labels if l["message_id"] in train_ids]
test_labels = [l for l in all_labels if l["message_id"] in test_ids]

print(f"Train: {len(train_labels)} claims | Test: {len(test_labels)} claims")

# Save
Path("data/train_silver_labels.json").write_text(json.dumps(train_labels, indent=2))
Path("data/test_gpt_labels.json").write_text(json.dumps(test_labels, indent=2))

print("\n✓ Saved:")
print("  - data/train_silver_labels.json (training data)")
print("  - data/test_gpt_labels.json (test data)")

# Stats
claim_counts = defaultdict(int)
for label in test_labels:
    claim_counts[label["claim_type"]] += 1

print("\nTest set claim distribution:")
for claim_type, count in sorted(claim_counts.items(), key=lambda x: -x[1]):
    print(f"  {claim_type:25s}: {count:3d}")

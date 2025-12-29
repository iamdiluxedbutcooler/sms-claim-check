# Claim Extraction Research Pipeline

## Overview

This project implements and compares **4 approaches** for claim extraction from phishing SMS messages. The goal is to extract **atomic, verifiable claims** that can be fact-checked against authoritative sources for phishing detection.

### Problem Statement

Traditional phishing detection relies on pattern matching and is vulnerable to out-of-distribution (OOD) attacks. Our approach:
1. **Decomposes** SMS messages into atomic, verifiable claims
2. **Structures** claims into verifiable queries
3. **Verifies** claims against authoritative sources (not implemented in this stage)

This repository focuses on **Stage 1: Claim Extraction**.

---

## The 4 Approaches

### Approach 1: Entity-First NER

**Pipeline:**
```
SMS → Entity-NER Model → [BRAND, PHONE, URL, ...] → Rule-based/T5 → Structured Claims
```

**Training Data:** Entity-based annotations (BRAND, PHONE, URL, ORDER_ID, AMOUNT, DATE, DEADLINE, ACTION_REQUIRED)

**Pros:**
- Entities are concrete and well-defined
- Can reuse entity extraction for other tasks
- Entity-level verification possible
- Clear intermediate representation

**Cons:**
- May not be robust against extremely OOD data
- Requires additional parsing step to form claims
- May miss implicit claims without explicit entities

**Config:** `configs/entity_ner.yaml`

---

### Approach 2: Claim-Phrase NER

**Pipeline:**
```
SMS → Claim-NER Model → [IDENTITY_CLAIM, DELIVERY_CLAIM, ...] → Parse → Structured Claims
```

**Training Data:** Claim-based annotations (IDENTITY_CLAIM, DELIVERY_CLAIM, FINANCIAL_CLAIM, ACCOUNT_CLAIM, URGENCY_CLAIM, ACTION_CLAIM)

**Pros:**
- Directly captures semantic claim units
- May be more robust to variations
- Captures implicit claims without explicit entities
- More aligned with verification task

**Cons:**
- Claim boundaries more ambiguous than entities
- Requires more sophisticated annotation
- Less reusable for other tasks

**Config:** `configs/claim_ner.yaml`

---

### Approach 3a: Hybrid Entity-NER + LLM

**Pipeline:**
```
SMS → Entity-NER Model → [entities] → GPT-4o-mini → Structured, Verifiable Claims (JSON)
```

**Training Data:** Entity-based annotations for NER training

**LLM Component:** Uses GPT-4o-mini with structured prompts to convert entities into verifiable claims

**Pros:**
- Can handle more complex reasoning
- More interpretable (can debug NER vs LLM separately)
- Leverages pre-trained LLM knowledge
- Produces structured output with verification steps

**Cons:**
- Extra cost for API calls
- Dependent on LLM quality
- Potential latency issues

**Config:** `configs/hybrid_llm.yaml`

**Prompt:** See `src/models/hybrid_prompts.py::ENTITY_TO_CLAIM_PROMPT`

---

### Approach 3b: Hybrid Claim-NER + LLM

**Pipeline:**
```
SMS → Claim-NER Model → [claim phrases] → GPT-4o-mini → Structured Verification Queries (JSON)
```

**Training Data:** Claim-based annotations for NER training

**LLM Component:** Uses GPT-4o-mini to structure claim phrases into detailed verification queries

**Pros:**
- Best of both worlds: semantic claim extraction + LLM reasoning
- More robust claim boundaries from NER
- Detailed verification plans from LLM
- Can identify implicit assertions

**Cons:**
- Extra cost for API calls
- Most complex pipeline
- Requires both annotation types

**Config:** `configs/hybrid_claim_llm.yaml`

**Prompt:** See `src/models/hybrid_prompts.py::CLAIM_TO_STRUCTURED_PROMPT`

---

### Approach 4: Contrastive Learning

**Pipeline:**
```
SMS → RoBERTa Encoder → Embedding → Contrastive Loss → Learned Representations
```

**Training Method:** Supervised Contrastive Learning (SupCon)
- Similar claims (same type) pulled together in embedding space
- Phishing vs legitimate pushed apart
- Claims from same message are positive pairs

**Pros:**
- Designed for generalization and OOD performance
- Learns semantic representations
- No need for explicit claim/entity boundaries
- Can use both entity and claim annotations as supervision signal

**Cons:**
- Need high-quality data
- Risk of semantic overgeneralization
- May not capture fine-grained distinctions
- Potential bias from training data

**Config:** `configs/contrastive.yaml`

---

## Annotation Schema

### Entity-Based Annotations

Used for **Approach 1** and **Approach 3a**.

**Entity Types:**
- `BRAND`: Company/organization names (Amazon, PayPal, IRS, etc.)
- `PHONE`: Phone numbers in any format
- `URL`: Web addresses, domains, shortened links
- `ORDER_ID`: Tracking numbers, order numbers, confirmation codes
- `AMOUNT`: Monetary values with currency symbols
- `DATE`: Non-urgent temporal references
- `DEADLINE`: Urgent time references (within 24h, immediately, ASAP)
- `ACTION_REQUIRED`: Imperative verbs (click, call, verify, confirm)

**Example:**
```
Message: "URGENT: Your Amazon package #ABC123 is delayed. Click bit.ly/pkg123"

Entities:
- DEADLINE: "URGENT"
- BRAND: "Amazon"
- ORDER_ID: "#ABC123"
- ACTION_REQUIRED: "Click"
- URL: "bit.ly/pkg123"
```

---

### Claim-Based Annotations

Used for **Approach 2** and **Approach 3b**.

**Claim Types:**
- `IDENTITY_CLAIM`: Who sent this / Who is this about
- `DELIVERY_CLAIM`: Package/delivery related assertions
- `FINANCIAL_CLAIM`: Money/payment related assertions
- `ACCOUNT_CLAIM`: Account status assertions
- `URGENCY_CLAIM`: Time-sensitive assertions
- `ACTION_CLAIM`: Required user actions

**Example:**
```
Message: "URGENT: Your Amazon package #ABC123 is delayed. Click bit.ly/pkg123"

Claims:
- IDENTITY_CLAIM: "Amazon" (implicitly claims to be from Amazon)
- DELIVERY_CLAIM: "Your Amazon package #ABC123 is delayed"
- ACTION_CLAIM: "Click bit.ly/pkg123"
- URGENCY_CLAIM: "URGENT" (implies immediate action needed)
```

**Key Difference:** Claims capture **semantic assertions** that need verification, while entities are **concrete spans** that can be extracted independently.

---

## Research Pipeline

### Phase 1: Annotation (Current)

1. **Entity Annotation:**
   ```bash
   python scripts/ai_preannotate_entities.py --submit
   python scripts/ai_preannotate_entities.py --check-status <batch_id>
   python scripts/ai_preannotate_entities.py --download <batch_id>
   ```

2. **Claim Annotation:**
   ```bash
   python scripts/ai_preannotate_claims.py --submit
   python scripts/ai_preannotate_claims.py --check-status <batch_id>
   python scripts/ai_preannotate_claims.py --download <batch_id>
   ```

3. **EDA:**
   ```bash
   python scripts/eda_comprehensive.py --entity data/annotations/entity_annotations.json
   python scripts/eda_comprehensive.py --claim data/annotations/claim_annotations.json
   python scripts/eda_comprehensive.py --entity <entity_file> --claim <claim_file> --compare
   ```

### Phase 2: Training

1. **Train NER Models:**
   ```bash
   python train.py --config configs/entity_ner.yaml
   python train.py --config configs/claim_ner.yaml
   ```

2. **Train Contrastive Model:**
   ```bash
   python train.py --config configs/contrastive.yaml
   ```

### Phase 3: Evaluation

1. **Run All Experiments:**
   ```bash
   ./run_all_experiments.sh
   ```

2. **Compare Models:**
   ```bash
   python scripts/compare_models.py
   ```

### Phase 4: Hybrid Inference

For hybrid approaches, inference happens in two stages:

1. **Entity/Claim Extraction** (NER model)
2. **LLM Parsing** (GPT-4o-mini batch API)

```bash
python inference.py --config configs/hybrid_llm.yaml --input test_messages.csv
```

---

## Evaluation Metrics

For all approaches, we evaluate:

1. **Extraction Quality:**
   - Precision, Recall, F1 (span-level)
   - Exact match accuracy
   - Partial match score

2. **Claim Verification Readiness:**
   - Are extracted claims verifiable?
   - Do they contain sufficient context?
   - Are verification steps actionable?

3. **OOD Robustness:**
   - Performance on unseen phishing patterns
   - Zero-shot capability on new brands
   - Resilience to adversarial modifications

4. **Efficiency:**
   - Inference time
   - API cost (for hybrid approaches)
   - Model size

---

## Expected Outcomes

**Hypothesis:** 
- **Claim-based approaches** (Approach 2, 3b) will perform better for claim verification tasks
- **Entity-based approaches** (Approach 1, 3a) may be more precise but less robust
- **Hybrid approaches** (3a, 3b) will achieve best quality but at higher cost
- **Contrastive learning** (Approach 4) will generalize best to OOD data

**Research Questions:**
1. Do claim-based annotations lead to better claim extraction?
2. Is the LLM overhead worth the quality improvement?
3. Can contrastive learning match supervised approaches?
4. Which approach is best for production deployment?

---

## Directory Structure

```
├── configs/                    # Experiment configurations
│   ├── entity_ner.yaml        # Approach 1
│   ├── claim_ner.yaml         # Approach 2
│   ├── hybrid_llm.yaml        # Approach 3a
│   ├── hybrid_claim_llm.yaml  # Approach 3b
│   └── contrastive.yaml       # Approach 4
├── scripts/
│   ├── ai_preannotate_entities.py  # Entity annotation with GPT-4o
│   ├── ai_preannotate_claims.py    # Claim annotation with GPT-4o
│   ├── eda_comprehensive.py        # Annotation analysis
│   └── compare_models.py           # Model comparison
├── src/
│   ├── models/
│   │   ├── entity_ner.py      # Approach 1
│   │   ├── claim_ner.py       # Approach 2
│   │   ├── hybrid_llm.py      # Approach 3a, 3b
│   │   ├── hybrid_prompts.py  # LLM prompts
│   │   └── contrastive.py     # Approach 4
│   └── data/
│       ├── loader.py
│       └── preprocessor.py
├── data/
│   ├── raw/
│   │   └── mendeley.csv       # Original dataset
│   ├── annotations/
│   │   ├── entity_annotations.json   # Entity-based
│   │   ├── claim_annotations.json    # Claim-based
│   │   └── *_batch_metadata.json     # Batch job metadata
│   └── eda/                   # EDA outputs
└── experiments/               # Trained models and results
    ├── approach1_entity_ner/
    ├── approach2_claim_ner/
    ├── approach3a_hybrid_entity_llm/
    ├── approach3b_hybrid_claim_llm/
    └── approach4_contrastive/
```

---

## Next Steps

1. [OK] Create entity and claim annotation scripts
2. [OK] Submit batch jobs to GPT-4o
3. [WAIT] Download and review annotations
4. [WAIT] Run comprehensive EDA
5. [WAIT] Manual annotation review and correction in Label Studio
6. [WAIT] Train all 4 approaches
7. [WAIT] Compare results
8. [WAIT] Select best approach for deployment

---

## References

- Label Studio: https://labelstud.io/
- OpenAI Batch API: https://platform.openai.com/docs/guides/batch
- RoBERTa: https://arxiv.org/abs/1907.11692
- Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362

# Project Restructure Summary

## Date: November 17, 2025

## Problem Statement

The original codebase was focused only on entity-based annotation and didn't align with the research goal of **claim extraction for phishing verification**. The project needed:

1. [OK] **Both entity-based AND claim-based annotations**
2. [OK] **Clear separation of 4 experimental approaches**
3. [OK] **Robust annotation prompts using GPT-4o**
4. [OK] **Proper experiment configurations**
5. [OK] **Comprehensive documentation**

## What Was Done

### 1. Created Entity-Based Annotation Pipeline [OK]

**File:** `scripts/ai_preannotate_entities.py`

- Uses GPT-4o batch API for cost-effective, high-quality annotations
- Extracts 8 entity types: BRAND, PHONE, URL, ORDER_ID, AMOUNT, DATE, DEADLINE, ACTION_REQUIRED
- Robust prompt with explicit rules and examples
- Output: `data/annotations/entity_annotations.json`

**Batch Job Status:**
- **Batch ID:** `batch_691a2e5b747881909a32552782d6e38f`
- **Messages:** 638 smishing messages
- **Status:** Validating (check after ~30min, complete in ~24h)
- **Check:** `python scripts/ai_preannotate_entities.py --check-status batch_691a2e5b747881909a32552782d6e38f`

### 2. Created Claim-Based Annotation Pipeline [OK]

**File:** `scripts/ai_preannotate_claims.py`

- Uses GPT-4o batch API for semantic claim extraction
- Extracts 6 claim types: IDENTITY_CLAIM, DELIVERY_CLAIM, FINANCIAL_CLAIM, ACCOUNT_CLAIM, URGENCY_CLAIM, ACTION_CLAIM
- Captures verifiable assertions and their verification requirements
- Output: `data/annotations/claim_annotations.json`

**Batch Job Status:**
- **Batch ID:** `batch_691a2f1215a48190806fcbe64a420aff`
- **Messages:** 638 smishing messages
- **Status:** Validating (check after ~30min, complete in ~24h)
- **Check:** `python scripts/ai_preannotate_claims.py --check-status batch_691a2f1215a48190806fcbe64a420aff`

### 3. Created Hybrid LLM Prompts [OK]

**File:** `src/models/hybrid_prompts.py`

Two hybrid approaches for inference (not training):

**Approach 3a:** Entity-NER + LLM
- NER model extracts entities
- LLM structures entities into verifiable claims with verification steps
- Prompt: `ENTITY_TO_CLAIM_PROMPT`

**Approach 3b:** Claim-NER + LLM
- NER model extracts claim phrases
- LLM structures claims into detailed verification queries
- Prompt: `CLAIM_TO_STRUCTURED_PROMPT`

Both use GPT-4o-mini for cost efficiency with batch processing support.

### 4. Created Comprehensive EDA Script [OK]

**File:** `scripts/eda_comprehensive.py`

Analyzes annotation quality with:
- Basic statistics (total annotations, avg per message, etc.)
- Label distribution
- Overlap detection
- Quality checks (whitespace, punctuation, etc.)
- Label co-occurrence analysis
- Visualization plots
- Comparison between entity and claim annotations

**Usage:**
```bash
python scripts/eda_comprehensive.py --entity data/annotations/entity_annotations.json
python scripts/eda_comprehensive.py --claim data/annotations/claim_annotations.json
python scripts/eda_comprehensive.py --entity <entity_file> --claim <claim_file> --compare
```

### 5. Updated All Experiment Configs [OK]

**Approach 1: Entity-First NER**
- File: `configs/entity_ner.yaml`
- Model: RoBERTa-base
- Data: `data/annotations/entity_annotations.json`
- Output: `experiments/approach1_entity_ner/`

**Approach 2: Claim-Phrase NER**
- File: `configs/claim_ner.yaml`
- Model: RoBERTa-base
- Data: `data/annotations/claim_annotations.json`
- Output: `experiments/approach2_claim_ner/`

**Approach 3a: Hybrid Entity-NER + LLM**
- File: `configs/hybrid_llm.yaml`
- NER: Entity-based RoBERTa
- LLM: GPT-4o-mini
- Output: `experiments/approach3a_hybrid_entity_llm/`

**Approach 3b: Hybrid Claim-NER + LLM**
- File: `configs/hybrid_claim_llm.yaml`
- NER: Claim-based RoBERTa
- LLM: GPT-4o-mini
- Output: `experiments/approach3b_hybrid_claim_llm/`

**Approach 4: Contrastive Learning**
- File: `configs/contrastive.yaml`
- Model: RoBERTa-base with supervised contrastive loss
- Data: Both entity and claim annotations
- Output: `experiments/approach4_contrastive/`

### 6. Created Comprehensive Documentation [OK]

**Research Pipeline:** `docs/RESEARCH_PIPELINE.md`
- Overview of all 4 approaches
- Annotation schema definitions
- Pros/cons of each approach
- Expected outcomes and research questions
- Complete workflow from annotation to evaluation

**Quick Start Guide:** `docs/QUICKSTART.md`
- Step-by-step instructions
- Installation and setup
- Running annotations, training, evaluation
- Troubleshooting common issues
- Performance benchmarks

## Directory Structure (Updated)

```
sms-claim-check/
├── configs/                      # Experiment configurations
│   ├── entity_ner.yaml          # Approach 1: Entity-First NER
│   ├── claim_ner.yaml           # Approach 2: Claim-Phrase NER
│   ├── hybrid_llm.yaml          # Approach 3a: Entity+LLM
│   ├── hybrid_claim_llm.yaml    # Approach 3b: Claim+LLM
│   └── contrastive.yaml         # Approach 4: Contrastive Learning
│
├── scripts/
│   ├── ai_preannotate_entities.py   # NEW: Entity annotation (GPT-4o)
│   ├── ai_preannotate_claims.py     # NEW: Claim annotation (GPT-4o)
│   ├── eda_comprehensive.py         # NEW: Comprehensive EDA
│   └── [other scripts...]
│
├── src/
│   └── models/
│       ├── hybrid_prompts.py    # NEW: LLM prompts for hybrid approaches
│       └── [other models...]
│
├── docs/
│   ├── RESEARCH_PIPELINE.md     # NEW: Complete research documentation
│   ├── QUICKSTART.md            # NEW: Quick start guide
│   └── annotation_guidelines.md
│
├── data/
│   ├── annotations/
│   │   ├── entity_batch_input.jsonl              # Batch input (entity)
│   │   ├── entity_batch_input_metadata.json      # Batch metadata
│   │   ├── entity_batch_id.txt                   # Batch ID reference
│   │   ├── claim_batch_input.jsonl               # Batch input (claim)
│   │   ├── claim_batch_input_metadata.json       # Batch metadata
│   │   ├── claim_batch_id.txt                    # Batch ID reference
│   │   ├── entity_annotations.json               # TO BE DOWNLOADED
│   │   └── claim_annotations.json                # TO BE DOWNLOADED
│   └── eda/                                       # EDA outputs (to be generated)
│
└── experiments/                  # Model outputs
    ├── approach1_entity_ner/
    ├── approach2_claim_ner/
    ├── approach3a_hybrid_entity_llm/
    ├── approach3b_hybrid_claim_llm/
    └── approach4_contrastive/
```

## Next Steps

### Immediate (24 hours)

1. **Wait for batch jobs to complete** (~24h)
   - Check status periodically
   - Entity batch: `batch_691a2e5b747881909a32552782d6e38f`
   - Claim batch: `batch_691a2f1215a48190806fcbe64a420aff`

2. **Download annotations**
   ```bash
   python scripts/ai_preannotate_entities.py --download batch_691a2e5b747881909a32552782d6e38f
   python scripts/ai_preannotate_claims.py --download batch_691a2f1215a48190806fcbe64a420aff
   ```

3. **Run comprehensive EDA**
   ```bash
   python scripts/eda_comprehensive.py --entity data/annotations/entity_annotations.json --output data/eda
   python scripts/eda_comprehensive.py --claim data/annotations/claim_annotations.json --output data/eda
   python scripts/eda_comprehensive.py --entity data/annotations/entity_annotations.json --claim data/annotations/claim_annotations.json --compare --output data/eda
   ```

### Short-term (1-2 days)

4. **Review annotation quality**
   - Check EDA reports in `data/eda/`
   - Review sample annotations
   - Fix any systematic errors
   - Optional: Import to Label Studio for manual review

5. **Train NER models**
   ```bash
   python train.py --config configs/entity_ner.yaml
   python train.py --config configs/claim_ner.yaml
   python train.py --config configs/contrastive.yaml
   ```

### Medium-term (1 week)

6. **Run hybrid approaches**
   ```bash
   python inference.py --config configs/hybrid_llm.yaml --input data/processed/test.csv
   python inference.py --config configs/hybrid_claim_llm.yaml --input data/processed/test.csv
   ```

7. **Compare all approaches**
   ```bash
   python scripts/compare_models.py
   ```

8. **Analyze results and iterate**

## Key Improvements

### Before [ERROR]
- Only entity-based annotations
- Unclear experiment structure
- Poor annotation quality (overlapping, whitespace issues)
- No claim-based approach
- Confusing codebase

### After [OK]
- **Both entity AND claim annotations**
- **4 clearly defined approaches** with separate configs
- **Robust GPT-4o annotation prompts**
- **Hybrid approaches** with LLM integration
- **Comprehensive documentation**
- **EDA pipeline** for quality analysis
- **Clear research pipeline** from annotation to evaluation

## Cost Estimate

**Annotation (GPT-4o batch API):**
- 638 messages × 2 annotation types = 1,276 requests
- Estimated cost: ~$30-50 total (batch API 50% discount)
- Expected completion: 24 hours

**Hybrid Inference (GPT-4o-mini):**
- Will be calculated after NER training
- Estimated: ~$5-10 per 1000 messages

## Files Modified/Created

### Created (13 files)
1. `scripts/ai_preannotate_entities.py`
2. `scripts/ai_preannotate_claims.py`
3. `scripts/eda_comprehensive.py`
4. `src/models/hybrid_prompts.py`
5. `configs/hybrid_claim_llm.yaml`
6. `docs/RESEARCH_PIPELINE.md`
7. `docs/QUICKSTART.md`
8. `data/annotations/entity_batch_input.jsonl`
9. `data/annotations/entity_batch_input_metadata.json`
10. `data/annotations/entity_batch_id.txt`
11. `data/annotations/claim_batch_input.jsonl`
12. `data/annotations/claim_batch_input_metadata.json`
13. `data/annotations/claim_batch_id.txt`

### Modified (4 files)
1. `configs/entity_ner.yaml` - Updated for approach 1
2. `configs/claim_ner.yaml` - Updated for approach 2
3. `configs/hybrid_llm.yaml` - Updated for approach 3a
4. `configs/contrastive.yaml` - Updated for approach 4

## Status: [OK] COMPLETE

All tasks completed successfully:
- [OK] Entity-based annotation pipeline
- [OK] Claim-based annotation pipeline
- [OK] Hybrid LLM prompts
- [OK] Batch jobs submitted (in progress)
- [OK] EDA script created
- [OK] All configs updated
- [OK] Comprehensive documentation

**The project now has a clear, well-structured research pipeline for comparing claim extraction approaches.**

---

## Questions?

Check:
- `docs/RESEARCH_PIPELINE.md` for detailed documentation
- `docs/QUICKSTART.md` for step-by-step instructions
- Batch job status commands in this document

## Researcher Notes

The project is now properly structured for comparing 4 distinct approaches to claim extraction:

1. **Entity-First NER** - Concrete, reusable, but may miss implicit claims
2. **Claim-Phrase NER** - Semantic, robust, but ambiguous boundaries
3. **Hybrid Entity+LLM** - Interpretable, high quality, but costly
4. **Hybrid Claim+LLM** - Best of both worlds, most complex
5. **Contrastive Learning** - OOD robust, no explicit boundaries needed

**Hypothesis:** Claim-based approaches (2, 3b) will outperform entity-based (1, 3a) for claim verification tasks, with hybrid approaches achieving best quality at higher cost.

Good luck with your research! [START]

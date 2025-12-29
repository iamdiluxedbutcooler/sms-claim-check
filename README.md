# SMS Phishing Detection via Claim Extraction

**A comparative study of Named Entity Recognition approaches for extracting verifiable claims from SMS phishing messages**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository presents a comparative evaluation of **two NER-based approaches** for extracting structured, verifiable claims from SMS phishing messages. The extracted claims serve as atomic units for downstream fact-checking and verification systems.

**Key Contributions:**
- Novel claim-phrase extraction task for phishing detection
- Comparative analysis of entity-first vs. claim-direct approaches
- Annotated dataset of 2,000 SMS messages (530 original + 438 augmented + 1,032 legitimate)
- Pre-trained RoBERTa-based models with 78% F1 on claim extraction

**Dataset Statistics:**
- **Total Messages:** 2,000 (balanced 51.6% HAM / 48.4% SMISH)
- **Claim Annotations:** 618 ground truth claims across 8 claim types
- **Source:** Mendeley SMS Spam Collection + GPT-4o augmentation

---

## Research Motivation

Traditional SMS phishing detection relies on surface-level features (keywords, URLs, sender patterns), which are easily evaded through obfuscation and adversarial attacks. This work explores **claim-based verification** as a more robust alternative:

1. **Claim Extraction** (this work): Identify atomic, verifiable statements in suspicious SMS
2. **Claim Verification** (future): Cross-reference claims against authoritative sources
3. **Risk Assessment** (future): Aggregate verified claims for final phishing determination

**Example:**
```
SMS: "Your Amazon package is delayed. Click here urgently to reschedule delivery."

Extracted Claims:
 IDENTITY_CLAIM: "Amazon" (Verify: Is sender legitimate Amazon?)
 DELIVERY_CLAIM: "package is delayed" (Verify: Do I have an active Amazon order?)
 ACTION_CLAIM: "Click here" (Verify: Is URL Amazon's official domain?)
 URGENCY_CLAIM: "urgently" (Red flag: Artificial time pressure)
```

---

## Methodology

### Data Collection & Annotation

#### 1. **Raw Data Acquisition**
- **Source:** Mendeley SMS Spam Collection (5,574 messages)
- **Filtering:** Selected 530 phishing-specific messages with clear claim indicators
- **Augmentation:** Generated 438 synthetic variations using GPT-4o to improve coverage
- **HAM Dataset:** Added 1,032 legitimate SMS for balanced training

#### 2. **Annotation Process**
- **Tool:** Custom review GUI (`review_claims_gui.py`) for manual validation
- **Annotators:** 2 researchers with inter-annotator agreement validation
- **Claim Types:** 8 categories (IDENTITY, ACTION, URGENCY, ACCOUNT, FINANCIAL, REWARD, VERIFICATION, DELIVERY)
- **Entity Types:** 9 categories (BRAND, PHONE, URL, EMAIL, AMOUNT, DATE, ACCOUNT, PERSON, LOCATION)
- **Quality Control:** Duplicate detection, boundary consistency checks, stratified splits

#### 3. **Pre-processing Pipeline**
```
Raw SMS → Tokenization → Character-level BIO alignment → 
Offset mapping → Overlapping span resolution → Stratified sampling
```

**Key Techniques:**
- **Character-Level BIO Tag Alignment:** Maps RoBERTa subword tokens to character spans, preventing tokenization-induced label misalignment (e.g., "Hurry!" → ["Hur", "##ry", "!"] maintains correct B-/I- tags)
- **Overlapping Span Resolution:** Greedy selection algorithm that sorts annotations by (start_position, -span_length) and filters conflicts, retaining longer spans (resolved 112 overlapping instances)
- **Stratified Sampling:** 80/20 train/test split stratified on HAM/SMISH labels to maintain class balance

---

### Training Strategy

**5-Fold Cross-Validation with Averaged Test Evaluation**

```
Training Samples (1,600 messages)
    ↓
Split into 5 folds (320 messages each)
    ↓
Train 5 models (each sees 4/5 of data for training, 1/5 for validation)
    ↓
Average validation metrics across 5 folds
    ↓
Evaluate all 5 models on held-out test set (400 messages)
    ↓
Report final averaged test performance
```

**Hyperparameters:**
- **Model:** `roberta-base` (125M parameters)
- **Learning Rate:** 2e-5 with cosine annealing
- **Batch Size:** 8 (with gradient accumulation = 2, effective batch = 16)
- **Epochs:** 15 with early stopping (patience = 3)
- **Optimizer:** AdamW (weight decay = 0.01, warmup ratio = 0.2)
- **Sequence Length:** 128 tokens (covers 95th percentile of SMS length)

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA 11.8+ (for GPU training)

### Setup

```bash
# Clone repository
git clone https://github.com/iamdiluxedbutcooler/sms-claim-check.git
cd sms-claim-check

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Download pre-trained models
# Model weights not included in repo due to size (408MB each)
# Download from: [Google Drive link] or train from scratch
```

---

## Usage

### 1. **Training** (Google Colab Recommended)

Use provided Jupyter notebooks for training:

```bash
# Open in Colab (GPU required)
approach5_pure_ner_improved.ipynb  # Claim-Direct NER (MAIN MODEL)
approach1_entity_first_ner.ipynb   # Entity-First NER (Comparison)
```

**Training Time:** ~25-30 minutes per approach on Colab (T4 GPU)

### 2. **Inference**

```python
# Load pre-trained model
from inference import ClaimExtractor

extractor = ClaimExtractor(model_path="claim_based_model/")

# Extract claims
sms = "Your Amazon account suspended. Call 0800-123-456 NOW to verify."
claims = extractor.extract_claims(sms)

print(claims)
# Output:
# [
#   {"type": "IDENTITY_CLAIM", "text": "Amazon", "confidence": 0.95},
#   {"type": "ACCOUNT_CLAIM", "text": "account suspended", "confidence": 0.89},
#   {"type": "ACTION_CLAIM", "text": "Call 0800-123-456", "confidence": 0.92},
#   {"type": "URGENCY_CLAIM", "text": "NOW", "confidence": 0.87}
# ]
```

### 3. **Evaluation**

```bash
# Evaluate model on test set
python scripts/evaluate_model_performance.py \
    evaluation_output/test_results_detailed.json \
    --output-dir evaluation_output/
```

Generates:
- Confusion matrix
- Per-claim type performance charts
- Error analysis report

### 4. **Review Tool**

Manual annotation/correction interface:

```bash
python review_claims_gui.py
```

---

## Repository Structure

```
sms-claim-check/
 approach1_entity_first_ner.ipynb        # Entity-First training notebook
 approach5_pure_ner_improved.ipynb       # Claim-Direct training notebook
 inference.py                            # Production inference script
 ood_test_smishtank.py                  # Out-of-distribution testing
 review_claims_gui.py                    # Manual review interface

 data/
    annotations/
       claim_annotations_2000_reviewed.json  # FINAL dataset
       entity_annotations_2000.json          # Entity annotations
       balanced_dataset_2000.json            # Balanced HAM/SMISH
    raw/
       mendeley.csv                          # Original source (530 messages)
       smishtank.csv                         # OOD test data
    processed/
        sms_phishing_ham_balanced_2000.csv   # Final CSV export

 evaluation_output/
    confusion_matrix.png
    performance_by_type.png
    evaluation_report.txt

 scripts/
    evaluate_model_performance.py       # Generate metrics & visualizations
    eda_annotations.py                  # Exploratory data analysis
    quality_control_annotations.py      # Annotation validation
    compare_models.py                   # Approach comparison

 notebooks/
    model_evaluation_and_visualization.ipynb

 requirements.txt                        # Unified dependencies
 README.md                              # This file
```

---

## Model Weights

**Note:** Model weights are **NOT included** in this repository due to size constraints (408MB each).

**Options:**
1. **Download pre-trained models:** [Contact authors for Google Drive link]
2. **Train from scratch:** Use provided Colab notebooks (~30 min per model)

**Saved Model Structure:**
```
claim_based_model.zip (408MB)
 pytorch_model.bin          # RoBERTa weights
 config.json                # Model configuration
 tokenizer_config.json      # RoBERTa tokenizer
 vocab.json                 # Vocabulary
 label_mappings.json        # Claim type mappings
```

---

## Future Work

1. **Dataset Expansion:** Annotate 2,000 additional messages to improve low-support claim types (DELIVERY_CLAIM, VERIFICATION_CLAIM)
2. **Claim Verification Module:** Implement fact-checking against authoritative sources (brand databases, account status APIs)
3. **Multi-language Support:** Extend to non-English SMS phishing (Spanish, French, etc.)
4. **Deployment:** Package as REST API for real-time SMS scanning
5. **Adversarial Robustness:** Test against obfuscation attacks (leetspeak, homoglyphs)
- **Data:** \`data/annotations/entity_annotations.json\`

### 2. Claim-Phrase NER
Directly extract semantic claim phrases (IDENTITY_CLAIM, FINANCIAL_CLAIM, etc.).
- **Config:** \`configs/claim_ner.yaml\`
- **Data:** \`data/annotations/claim_annotations.json\`

### 3a. Hybrid Entity-NER + LLM
Extract entities with NER, then use GPT-4o-mini to structure into verifiable claims.
- **Config:** \`configs/hybrid_llm.yaml\`
- **Requires:** Trained entity NER model + OpenAI API

### 3b. Hybrid Claim-NER + LLM
Extract claim phrases with NER, then use GPT-4o-mini to create verification queries.
- **Config:** \`configs/hybrid_claim_llm.yaml\`
- **Requires:** Trained claim NER model + OpenAI API

### 4. Contrastive Learning
Learn claim embeddings via supervised contrastive loss for OOD generalization.
- **Config:** \`configs/contrastive.yaml\`
- **Method:** SupCon loss with RoBERTa encoder

## Documentation

- **[ANNOTATION_RESULTS.md](ANNOTATION_RESULTS.md)** - Annotation quality and statistics
- **[docs/RESEARCH_PIPELINE.md](docs/RESEARCH_PIPELINE.md)** - Detailed research approach
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Step-by-step tutorial

## Key Features

- **Both entity & claim annotations** using GPT-4o batch API (100% success rate)
- **Clean architecture** with separate configs for each approach
- **Comprehensive EDA** with visualization and quality reports
- **Hybrid LLM prompts** for structured claim verification
- **No emojis** in code (removed for professionalism)

## Requirements

\`\`\`
Python 3.8+
PyTorch 2.0+
Transformers 4.30+
OpenAI API (for annotations and hybrid approaches)
\`\`\`

See \`requirements.txt\` for full list.

6. **Hybrid Approaches:** Combine entity extraction with claim detection for improved precision
7. **Explainability:** Generate natural language explanations for detected claims

---

## Limitations

- **Dataset Bias:** Training data primarily from English-language UK/US phishing SMS (2015-2020), may not generalize to other regions or time periods
- **Claim Type Ambiguity:** `OTHER_CLAIM` acts as confusion sink, indicating need for more granular claim taxonomy
- **Manual Annotation Bottleneck:** Current dataset limited to 2,000 messages due to annotation cost (~$0.05/message with GPT-4o validation)
- **No Temporal Evolution:** Static dataset doesn't capture evolving phishing tactics over time


## Acknowledgments

- **Data Source:** [UCI SMS Spam Collection (Mendeley)](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Pre-trained Models:** [HuggingFace Transformers](https://huggingface.co/roberta-base)


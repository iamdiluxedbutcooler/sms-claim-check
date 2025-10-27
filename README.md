# SMS Phishing Detection via Multi-Agent Claim Verification

## Overview

This project implements a phishing SMS detection system designed to achieve high accuracy on known attacks while maintaining robust performance on zero-day and out-of-distribution attacks.

The system employs a multi-agent claim verification architecture that extracts and verifies factual claims against external knowledge sources, rather than relying solely on pattern matching approaches.

## Architecture

```
SMS Input 
  |
  v
Claim Extraction Agent (extracts entities and claims)
  |
  v
Claim Parsing Agent (structures into predicate format)
  |
  v
Claim Verification Agent (verifies against external sources)
  |
  v
Classification Decision (Supported/Unsupported/Unsure + Confidence)
```

## Dataset

Source: Mendeley SMS Dataset

Training Set: 4,776 messages
- Ham: 3,875
- Smishing: 510
- Spam: 391

Test Set: 1,195 messages
- Ham: 969
- Smishing: 128
- Spam: 98

## Phase 1: Data Annotation

### Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install label-studio
```

### Data Preparation

```bash
python scripts/prepare_data.py
python scripts/convert_to_label_studio.py
```

### Annotation Interface

```bash
label-studio start
```

Access the interface at http://localhost:8080 and configure using `config/label_studio_config.xml`.

### Project Structure
```
sms-claim-check/
├── data/
│   ├── raw/
│   ├── processed/
│   └── annotations/
├── config/
│   ├── entity_schema.py
│   └── label_studio_config.xml
├── scripts/
│   ├── prepare_data.py
│   ├── convert_to_label_studio.py
│   ├── quality_check.py
│   └── export_annotations.py
├── docs/
│   └── annotation_guidelines.md
├── notebooks/
└── requirements.txt
```

## Annotation Targets

Primary dataset: 510 smishing messages
Negative examples: 100 ham messages
Additional: 50 spam messages

Data split ratios:
- Training: 67%
- Validation: 17%
- Test: 16%

## Entity Schema

The annotation schema consists of eight entity types:

1. BRAND: Company or organization names
2. PHONE: Phone numbers in any format
3. URL: Web links, including shortened URLs
4. ORDER_ID: Order, tracking, or invoice identifiers
5. AMOUNT: Monetary amounts with or without currency symbols
6. DATE: Temporal references
7. DEADLINE: Time pressure indicators
8. ACTION_REQUIRED: Imperative action verbs

Detailed annotation guidelines are provided in `docs/annotation_guidelines.md`.

## Subsequent Phases

Phase 2: NER model training using multiple approaches
Phase 3: Claim parsing implementation
Phase 4: Verification agent development
Phase 5: Baseline comparison and evaluation

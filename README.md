# SMS Claim Extraction for Phishing Detection

A research project comparing **4 approaches** to extract atomic, verifiable claims from phishing SMS messages. This is the first stage of a multi-agent claim verification system for phishing detection.

## Project Overview

**Goal:** Extract structured, verifiable claims from SMS messages that can be fact-checked against authoritative sources.

**Approach:** Compare 4 different methods for claim extraction:
1. **Entity-First NER** - Extract entities then parse to claims
2. **Claim-Phrase NER** - Directly extract claim phrases
3. **Hybrid NER + LLM** - Combine NER with GPT-4o for structured output
4. **Contrastive Learning** - Learn claim embeddings for OOD robustness

## Quick Start

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env  # Add your OPENAI_API_KEY

# View annotation results (already completed!)
cat ANNOTATION_RESULTS.md

# Train models
python train.py --config configs/entity_ner.yaml
python train.py --config configs/claim_ner.yaml

# Compare results
python scripts/compare_models.py
\`\`\`

## Current Status

[OK] **Annotations Complete** - Both entity and claim-based annotations done via GPT-4o
- Entity annotations: 638 messages, 3,141 entities, 100% success
- Claim annotations: 638 messages, 1,833 claims, 100% success
- See \`ANNOTATION_RESULTS.md\` for details

[OK] **Code Restructured** - Clear separation of 4 approaches with proper configs

[WAIT] **Next:** Train models and compare performance

## The 4 Approaches

### 1. Entity-First NER
Extract concrete entities (BRAND, PHONE, URL, etc.) then parse to structured claims.
- **Config:** \`configs/entity_ner.yaml\`
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

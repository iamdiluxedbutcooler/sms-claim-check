# Quick Start Guide

## Prerequisites

- Python 3.8+
- OpenAI API key (for annotation and hybrid approaches)
- Label Studio (optional, for manual annotation review)

## Installation

```bash
# Clone repository
git clone <repo_url>
cd sms-claim-check

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
.\venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Setup Environment

Create `.env` file in project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
LABEL_STUDIO_API_KEY=your_label_studio_key_here  # Optional
```

## Step-by-Step Workflow

### Step 1: Generate Annotations

#### Option A: Entity-Based Annotations (for Approach 1 & 3a)

```bash
# Submit batch job to GPT-4o
python scripts/ai_preannotate_entities.py --submit

# Check status (replace <batch_id> with actual ID from previous command)
python scripts/ai_preannotate_entities.py --check-status <batch_id>

# Download results when complete (usually 24h)
python scripts/ai_preannotate_entities.py --download <batch_id>

# Output: data/annotations/entity_annotations.json
```

#### Option B: Claim-Based Annotations (for Approach 2 & 3b)

```bash
# Submit batch job to GPT-4o
python scripts/ai_preannotate_claims.py --submit

# Check status
python scripts/ai_preannotate_claims.py --check-status <batch_id>

# Download results
python scripts/ai_preannotate_claims.py --download <batch_id>

# Output: data/annotations/claim_annotations.json
```

**Cost Estimate:** ~$20-40 for 500 messages with GPT-4o

### Step 2: Run EDA

```bash
# Analyze entity annotations
python scripts/eda_comprehensive.py --entity data/annotations/entity_annotations.json --output data/eda

# Analyze claim annotations
python scripts/eda_comprehensive.py --claim data/annotations/claim_annotations.json --output data/eda

# Compare both
python scripts/eda_comprehensive.py \
  --entity data/annotations/entity_annotations.json \
  --claim data/annotations/claim_annotations.json \
  --compare \
  --output data/eda
```

**Outputs:**
- `data/eda/entity_eda_report.json` - Entity annotation statistics
- `data/eda/claim_eda_report.json` - Claim annotation statistics
- `data/eda/*.png` - Visualization plots
- `data/eda/entity_vs_claim_comparison.json` - Comparison report

### Step 3: Review Annotations (Optional)

If you have Label Studio installed:

```bash
# Import annotations
label-studio start

# Navigate to http://localhost:8080
# Create new project
# Import data/annotations/entity_annotations.json or claim_annotations.json
# Import config from config/label_studio_config.xml
# Review and correct annotations
# Export corrected annotations
```

### Step 4: Train Models

#### Approach 1: Entity-First NER

```bash
python train.py --config configs/entity_ner.yaml

# Output: experiments/approach1_entity_ner/
```

#### Approach 2: Claim-Phrase NER

```bash
python train.py --config configs/claim_ner.yaml

# Output: experiments/approach2_claim_ner/
```

#### Approach 4: Contrastive Learning

```bash
python train.py --config configs/contrastive.yaml

# Output: experiments/approach4_contrastive/
```

### Step 5: Run Hybrid Approaches (Inference)

Hybrid approaches don't require training - they use trained NER models + LLM.

#### Approach 3a: Entity-NER + LLM

```bash
# First, train entity NER (Step 4, Approach 1)
# Then run hybrid inference
python inference.py --config configs/hybrid_llm.yaml --input data/processed/test.csv

# Output: experiments/approach3a_hybrid_entity_llm/results.json
```

#### Approach 3b: Claim-NER + LLM

```bash
# First, train claim NER (Step 4, Approach 2)
# Then run hybrid inference
python inference.py --config configs/hybrid_claim_llm.yaml --input data/processed/test.csv

# Output: experiments/approach3b_hybrid_claim_llm/results.json
```

### Step 6: Compare All Approaches

```bash
python scripts/compare_models.py

# Output: 
# - experiments/model_comparison.json
# - experiments/comparison_plots/*.png
```

## Quick Test (Small Dataset)

To test the pipeline with a small dataset:

```bash
# Test with 10 messages
python scripts/ai_preannotate_entities.py --submit --limit 10
python scripts/ai_preannotate_claims.py --submit --limit 10

# Wait for batch completion (~30 minutes for small batch)
# Download and run EDA
python scripts/ai_preannotate_entities.py --check-status <batch_id>
python scripts/ai_preannotate_entities.py --download <batch_id>

python scripts/eda_comprehensive.py --entity data/annotations/entity_annotations.json
```

## Common Issues

### Issue: "OPENAI_API_KEY not found"

**Solution:** Make sure `.env` file exists with your API key:
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Issue: Batch job takes too long

**Solution:** OpenAI batch API has 24h window. Check status:
```bash
python scripts/ai_preannotate_entities.py --check-status <batch_id>
```

### Issue: Out of memory during training

**Solution:** Reduce batch size in config file:
```yaml
training_config:
  batch_size: 8  # Reduce from 16
```

### Issue: Poor annotation quality

**Solution:** 
1. Run EDA to identify issues
2. Import to Label Studio for manual correction
3. Adjust prompts in annotation scripts if needed

## Performance Benchmarks

**Expected Results (on test set):**

| Approach | F1 Score | Inference Time | API Cost |
|----------|----------|----------------|----------|
| Entity-NER | 0.82-0.88 | Fast | $0 |
| Claim-NER | 0.78-0.85 | Fast | $0 |
| Hybrid Entity+LLM | 0.85-0.92 | Medium | $$ |
| Hybrid Claim+LLM | 0.88-0.94 | Medium | $$ |
| Contrastive | 0.75-0.82 | Fast | $0 |

*Actual results depend on annotation quality and dataset characteristics*

## Next Steps

After completing experiments:

1. **Analyze results:** Review comparison report
2. **Error analysis:** Identify failure cases for each approach
3. **Iterate:** Improve prompts, adjust configs, add training data
4. **Deploy:** Select best approach for production

## Getting Help

- Check `docs/RESEARCH_PIPELINE.md` for detailed documentation
- Review example annotations in `data/annotations/`
- See config examples in `configs/`

## Citation

If you use this code in your research, please cite:
```
[Your paper citation here]
```

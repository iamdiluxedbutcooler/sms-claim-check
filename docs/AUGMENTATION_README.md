# SMS Phishing Dataset Augmentation

## Overview

This pipeline generates **1000 high-quality, diverse phishing messages** from the Mendeley SMS dataset by:

1. **Cleaning & deduplicating** the original dataset
2. **Keeping originals intact** (no modifications to source data)
3. **Generating augmented variations** using GPT-4o-mini
4. **Improving quality** of poorly-written messages
5. **Ensuring coverage** of all 12 claim types

## 12 Claim Types Covered

The augmentation ensures representation of:

1. **IDENTITY_CLAIM** - "We are Amazon/PayPal/IRS"
2. **DELIVERY_CLAIM** - "Your package is delayed"
3. **FINANCIAL_CLAIM** - "You won $5000"
4. **ACCOUNT_CLAIM** - "Your account is suspended"
5. **URGENCY_CLAIM** - "Act now within 24h"
6. **ACTION_CLAIM** - "Click here / Call now"
7. **VERIFICATION_CLAIM** - "Verify your identity"
8. **SECURITY_CLAIM** - "Suspicious activity detected"
9. **REWARD_CLAIM** - "Loyalty bonus available"
10. **LEGAL_CLAIM** - "Legal action pending"
11. **SOCIAL_CLAIM** - "Friend needs help"
12. **CREDENTIALS_CLAIM** - "Update password"

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements_augmentation.txt

# Set OpenAI API key
export OPENAI_API_KEY='sk-your-key-here'
```

### Run Augmentation

```bash
# Option 1: Use the shell script
chmod +x run_augmentation.sh
./run_augmentation.sh

# Option 2: Run Python directly
python scripts/augment_dataset.py
```

## What You Get

### Output Files

1. **`data/annotations/augmented_phishing_1000.json`**
 - 1000 phishing messages in Label Studio format
 - Ready for annotation
 - Original + augmented messages

2. **`data/annotations/augmented_phishing_1000_metadata.json`**
 - Statistics and coverage metrics
 - Claim type distribution
 - Generation details

### Dataset Structure

```json
{
 "id": "mendeley_0", // or "augmented_123"
 "data": {
 "text": "URGENT: Your Amazon package is delayed..."
 },
 "annotations": [{
 "result": [], // To be annotated
 "was_cancelled": false,
 "ground_truth": false
 }],
 "meta": {
 "claim_types": ["DELIVERY_CLAIM", "URGENCY_CLAIM"],
 "is_augmented": false,
 "source": "mendeley_original" // or "gpt_augmented"
 }
}
```

## Augmentation Strategies

The pipeline uses 3 GPT-powered strategies:

### 1. **Rephrase** (30%)
- Keeps same meaning and tactics
- Changes wording completely
- Maintains SMS length and urgency

**Example:**
- Original: `"Your package delivery failed. Click bit.ly/abc to reschedule"`
- Rephrased: `"We couldn't deliver your parcel. Visit amzn.to/xyz to update address"`

### 2. **Enhance** (30%)
- Fixes grammar/spelling
- Makes more convincing
- Improves professional tone
- Keeps core tactics

**Example:**
- Original: `"u won 500$ call now!!!!"`
- Enhanced: `"Congratulations! You've won $500. Call 1-800-555-0123 to claim within 24h"`

### 3. **Rewrite** (40%)
- Completely new scenario
- Different brand/service
- Same claim types
- Fresh social engineering

**Example:**
- Original: `"Amazon account suspended, verify now"`
- Rewritten: `"Your PayPal payment was declined. Update billing info immediately to avoid fees"`

## Pipeline Flow

```

 Mendeley CSV 
 (5973 messages) 

 
 

 Filter Phishing 
 (638 smishing) 

 
 

 Deduplicate 
 (Remove ~150 dupes) 

 
 

 ~490 Clean Original 
 (Kept as-is) 

 
 
 
 
 GPT Augmentation
 (Generate ~510) 
 
 
 

 Final Dataset: 1000 Messages 
 - 490 Original 
 - 510 Augmented 
 - All 12 claim types covered 

```

## Configuration

Edit `scripts/augment_dataset.py` to customize:

```python
# Target dataset size
TARGET_SIZE = 1000

# Probability of augmenting each message
AUGMENTATION_PROB = 0.5 # 50%

# Paths
MENDELEY_PATH = 'data/raw/mendeley.csv'
OUTPUT_PATH = 'data/annotations/augmented_phishing_1000.json'
```

## Expected Statistics

After running, you should see:

```
 Dataset complete!
 Total messages: 1000
 Original: 488
 Augmented: 512

 Claim Type Coverage:
 IDENTITY_CLAIM : 245 messages (24.5%)
 DELIVERY_CLAIM : 189 messages (18.9%)
 FINANCIAL_CLAIM : 312 messages (31.2%)
 ACCOUNT_CLAIM : 201 messages (20.1%)
 URGENCY_CLAIM : 567 messages (56.7%)
 ACTION_CLAIM : 723 messages (72.3%)
 VERIFICATION_CLAIM : 156 messages (15.6%)
 SECURITY_CLAIM : 134 messages (13.4%)
 REWARD_CLAIM : 98 messages (9.8%)
 LEGAL_CLAIM : 45 messages (4.5%)
 SOCIAL_CLAIM : 67 messages (6.7%)
 CREDENTIALS_CLAIM : 89 messages (8.9%)
```

## Quality Assurance

The pipeline ensures:

1. **No duplicates** - MD5 hash checking
2. **Realistic length** - 50-160 characters (SMS standard)
3. **Natural language** - GPT-4o-mini quality
4. **Diverse tactics** - Multiple augmentation strategies
5. **Claim coverage** - All 12 types represented
6. **Original preserved** - Source data unchanged

## Next Steps

After augmentation:

1. **Review Quality**
 ```bash
 # Check a sample
 python -c "import json; data=json.load(open('data/annotations/augmented_phishing_1000.json')); [print(f\"{d['id']}: {d['data']['text']}\") for d in data[:10]]"
 ```

2. **Annotate in Label Studio**
 - Import `augmented_phishing_1000.json`
 - Annotate claim spans
 - Export annotations

3. **Train Models**
 ```bash
 python train_kfold.py --config configs/claim_ner.yaml --n_folds 5
 ```

## Troubleshooting

### Issue: OpenAI API Rate Limit

**Solution:** Add delays between API calls
```python
# In MessageAugmenter._call_gpt_for_variation()
time.sleep(1) # Increase from 0.5 to 1 second
```

### Issue: Not enough original messages

**Solution:** Lower augmentation probability to generate more
```python
AUGMENTATION_PROB = 0.7 # Augment 70% of messages
```

### Issue: Poor quality augmentations

**Solution:** Adjust GPT temperature
```python
temperature=0.7 # Lower = more conservative (default: 0.8)
```

## Tips

1. **Cost Estimate**: ~$0.50-1.00 for 1000 messages with GPT-4o-mini
2. **Time**: ~15-30 minutes depending on API speed
3. **Review**: Always manually review ~50 samples before full annotation
4. **Iterate**: Run multiple times with different seeds for more diversity

## References

- Mendeley SMS Spam Collection: https://data.mendeley.com/datasets/5fa33zng9v/1
- GPT-4o-mini: https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/
- Phishing Taxonomy: "A Comprehensive Taxonomy of Phishing" (APWG)

## Contributing

To improve augmentation quality:

1. Add more claim pattern regexes in `ClaimAnalyzer`
2. Refine GPT prompts for better variations
3. Add domain-specific augmentation strategies
4. Implement claim type balancing

---

**Author:** SMS Claim Check Research Team 
**Last Updated:** December 2025 
**License:** MIT

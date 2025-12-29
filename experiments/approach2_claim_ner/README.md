# Approach 2: Claim-Phrase NER

## Overview
Train a RoBERTa-based NER model to directly extract claim phrases from SMS messages.

## Model Architecture
- **Base Model**: `roberta-base` or `distilroberta-base`
- **Task**: Token Classification (NER)
- **Labels**: 12 claim types + O (non-claim)
  - IDENTITY_CLAIM
  - DELIVERY_CLAIM
  - FINANCIAL_CLAIM
  - ACCOUNT_CLAIM
  - URGENCY_CLAIM
  - ACTION_CLAIM
  - VERIFICATION_CLAIM
  - SECURITY_CLAIM
  - REWARD_CLAIM
  - LEGAL_CLAIM
  - SOCIAL_CLAIM
  - CREDENTIALS_CLAIM

## Label Format
Using BIO (Beginning-Inside-Outside) tagging:
- `B-{CLAIM_TYPE}`: Beginning of a claim
- `I-{CLAIM_TYPE}`: Inside/continuation of a claim
- `O`: Outside any claim (regular text)

Total labels: 1 (O) + 12*2 (B- and I- for each claim type) = 25 labels

## Training Data
- **Source**: `data/annotations/claim_annotations_2000.json`
- **Total samples**: 2,000 SMS messages
- **Split**: 70% train, 15% val, 15% test (stratified by ham/spam)

## Advantages
1. **Direct claim extraction** - no intermediate entity parsing needed
2. **Semantic understanding** - captures claim meaning, not just entities
3. **Handles implicit claims** - can detect claims without explicit brand names
4. **Robust to variations** - learns patterns beyond specific entities
5. **Perfect for verification** - extracted claims map directly to verification questions

## Training Strategy
1. Convert annotations to BIO format
2. Tokenize with RoBERTa tokenizer (handle subword alignment)
3. Train with standard NER losses (CrossEntropy)
4. Use class weights to handle label imbalance
5. Evaluate with seqeval metrics (precision, recall, F1 per claim type)

## Expected Performance
- **In-distribution**: 85-90% F1 on test set
- **Out-of-distribution**: Should generalize better than entity-based NER
- **Key metrics**: Per-claim-type F1, overall F1, support for each class

## Output Format
```python
{
    "text": "Your Amazon package is delayed. Click here urgently.",
    "claims": [
        {
            "type": "IDENTITY_CLAIM",
            "text": "Amazon",
            "start": 5,
            "end": 11,
            "confidence": 0.95
        },
        {
            "type": "DELIVERY_CLAIM",
            "text": "package is delayed",
            "start": 12,
            "end": 30,
            "confidence": 0.92
        },
        {
            "type": "ACTION_CLAIM",
            "text": "Click here",
            "start": 32,
            "end": 42,
            "confidence": 0.88
        },
        {
            "type": "URGENCY_CLAIM",
            "text": "urgently",
            "start": 43,
            "end": 51,
            "confidence": 0.91
        }
    ]
}
```

## Files Structure
```
experiments/approach2_claim_ner/
├── README.md (this file)
├── data_loader.py       # Convert annotations to NER format
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── inference.py         # Inference script
├── config.yaml          # Training configuration
├── models/              # Saved model checkpoints
│   └── best_model/
├── logs/                # Training logs
└── results/             # Evaluation results
```

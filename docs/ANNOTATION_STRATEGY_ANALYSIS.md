# Annotation Strategy Analysis & Recommendations

## TL;DR Recommendations

**[OK] YES - Separate annotation schemas is GOOD**
**[OK] NO - Don't annotate all 2000 messages initially**
**[TARGET] Recommended: 500 messages with dual annotation (1.4 work days)**

---

## Q1: Is Separation Good or Bad?

### [OK] **SEPARATION IS GOOD** - Here's Why:

#### 1. **Different Cognitive Tasks**
- **Entity NER**: Objective, concrete, clear boundaries
  - "Is this a brand name?" → YES/NO
  - "Is this a phone number?" → YES/NO
  - Low inter-annotator disagreement (~90%+ agreement expected)
  
- **Claim-Phrase NER**: Subjective, semantic, fuzzy boundaries
  - "Where does URGENCY_CLAIM start/end?" → Debatable
  - "Is 'call now' an ACTION_CLAIM or part of URGENCY_CLAIM?" → Ambiguous
  - Higher inter-annotator disagreement (~70-80% agreement expected)

#### 2. **Complementary Strengths**
```
Entity-First:                    Claim-Phrase:
[OK] High precision                [OK] High recall
[OK] Verifiable outputs            [OK] Captures implicit claims
[OK] Reusable components           [OK] End-to-end extraction
[OK] Easy quality control          [OK] Robust to paraphrasing
[X] Misses implicit claims        [X] Ambiguous boundaries
[X] Brittle to variations         [X] Harder to verify
```

#### 3. **Pipeline vs End-to-End Trade-off**
- **Entity-First**: Better for production (explainable, debuggable, modular)
- **Claim-Phrase**: Better for research (flexible, robust, fewer steps)

#### 4. **Ensemble Potential**
- Train both models independently
- Combine predictions using:
  - Voting (majority wins)
  - Stacking (meta-classifier on top)
  - Rule-based fusion (entities → claims → structured)

### [X] **Why NOT to Merge Them:**

1. **Annotator Confusion**: Mixing concrete entities with abstract claims = cognitive overload
2. **Quality Degradation**: Annotators will default to "easy" entities, rush through claims
3. **Inflexible Training**: Can't experiment with different architectures per task
4. **Harder Analysis**: Can't isolate which approach works better

---

## Q2: How Much Data to Annotate?

### [X] **DON'T Annotate All 2000 Messages** - Here's Why:

#### Annotation Burden
- **Entity-only (all 2000)**: 25 hours (~3 work days)
- **Claim-only (1000 phishing)**: 19 hours (~2.3 work days)
- **Dual annotation (all)**: 44 hours (~5.5 work days)

#### Diminishing Returns
Research shows NER performance plateaus:
- 200 samples: ~60-70% F1
- 500 samples: ~75-85% F1
- 1000 samples: ~80-87% F1
- 2000 samples: ~82-88% F1 (only +2-3% improvement!)

### [OK] **RECOMMENDED Strategy: Stratified Smart Sampling**

#### Phase 1: Pilot (100 messages - 1.5 hours dual)
**Purpose**: Test annotation quality, refine guidelines

```
50 phishing (mix of original + augmented)
50 ham
```

**Action**: 
- Annotate with both schemas
- Measure inter-annotator agreement
- Refine label definitions
- Update annotation guidelines

#### Phase 2: Core Dataset (500 messages - 11 hours dual)
**Purpose**: Train production models

```
250 phishing (125 original + 125 augmented)
250 ham
```

**Stratification**:
- Balance claim types (IDENTITY, DELIVERY, FINANCIAL, etc.)
- Include diverse brands (PayPal, Amazon, banks, etc.)
- Mix message lengths (short, medium, long)
- Include edge cases (minimal entities, implicit claims)

**Why 500?**
- Sufficient for RoBERTa fine-tuning (typically 300-1000 samples)
- Manageable annotation time (~1.4 work days dual)
- Leaves 1500 messages for future expansion

#### Phase 3: Active Learning (Optional - 200-300 more)
**Purpose**: Target model weaknesses

After initial training:
1. Run models on remaining 1500 messages
2. Identify low-confidence predictions
3. Annotate only the hardest 200-300 cases
4. Retrain → expect +3-5% F1 improvement

---

## Detailed Sampling Strategy

### Recommended: 500-Message Dual Annotation

```python
Phishing Breakdown (250 total):
  - Original Mendeley: 125
    * High-confidence phishing: 80
    * Edge cases (short, typos): 45
  
  - GPT Augmented: 125
    * Rephrase: 42
    * Enhance: 42
    * Rewrite: 41

HAM Breakdown (250 total):
  - Diverse conversation styles
  - Include SMS-like short messages
  - Include longer benign messages
  
Claim Type Distribution (target for phishing):
  IDENTITY_CLAIM:      15-20%
  DELIVERY_CLAIM:      15-20%
  FINANCIAL_CLAIM:     25-30%
  ACCOUNT_CLAIM:       10-15%
  URGENCY_CLAIM:       40-50%
  ACTION_CLAIM:        50-60%
  VERIFICATION_CLAIM:  10-15%
  SECURITY_CLAIM:      10-15%
  REWARD_CLAIM:        15-20%
  LEGAL_CLAIM:         5-10%
  SOCIAL_CLAIM:        5-10%
  CREDENTIALS_CLAIM:   10-15%
```

---

## Workflow Recommendation

### Step 1: Create Stratified Sample (10 min)
```bash
python scripts/create_annotation_sample.py \
  --input data/annotations/balanced_dataset_2000.json \
  --output data/annotations/annotation_sample_500.json \
  --size 500 \
  --stratify claim_types
```

### Step 2: Dual Export for Label Studio (5 min)
```bash
# Entity schema
python scripts/export_for_label_studio.py \
  --input data/annotations/annotation_sample_500.json \
  --schema entity \
  --output data/annotations/entity_annotation_sample_500.json

# Claim schema  
python scripts/export_for_label_studio.py \
  --input data/annotations/annotation_sample_500.json \
  --schema claim \
  --output data/annotations/claim_annotation_sample_500.json
```

### Step 3: Annotate in Parallel (11 hours total)
- **Entity annotation**: 6.2 hours (500 msgs × 45 sec)
- **Claim annotation**: 4.7 hours (250 phishing × 67.5 sec)
- HAM messages in claim schema automatically get 'O' labels

### Step 4: Quality Check (30 min)
- Calculate inter-annotator agreement (if multiple annotators)
- Check for label inconsistencies
- Validate entity/claim alignment

### Step 5: Train Models (2-4 hours)
- Entity-First NER: RoBERTa-base fine-tuned
- Claim-Phrase NER: RoBERTa-base fine-tuned
- Hybrid: Entity NER → Rule-based claim structuring

### Step 6: Evaluate & Iterate
- Test on held-out 100 messages
- Identify failure modes
- Use active learning on remaining 1400 messages if needed

---

## Cost-Benefit Analysis

### Option A: All 2000 messages dual annotation
- **Time**: 44 hours (~5.5 days)
- **Cost**: $1,100 (assuming $25/hour annotator)
- **Expected F1**: 82-88%

### Option B: 500 messages dual annotation (RECOMMENDED)
- **Time**: 11 hours (~1.4 days)
- **Cost**: $275
- **Expected F1**: 75-85%
- **ROI**: Save 33 hours, lose only 5-7% F1
- **Bonus**: 1,500 messages for active learning

### Option C: 500 entity + 500 claim (separate sets)
- **Time**: 11 hours
- **Cost**: $275
- **Expected F1**: Same as Option B
- **Con**: Can't compare entity vs claim on same messages
- **Pro**: More diverse training data

---

## Final Recommendation

### [TARGET] **Start with 500 messages, dual annotation**

**Why?**
1. [OK] Manageable time investment (~1.4 work days)
2. [OK] Sufficient data for RoBERTa fine-tuning
3. [OK] Enables direct entity ↔ claim comparison
4. [OK] Leaves budget for active learning iterations
5. [OK] Reduces annotation fatigue = higher quality
6. [OK] Faster time-to-results for paper/thesis

**Next Steps:**
1. Create stratified 500-message sample
2. Annotate entities on all 500 messages
3. Annotate claims on 250 phishing messages
4. Train both models
5. Evaluate on held-out 100 messages
6. Use active learning on remaining 1,400 if needed

**Expected Timeline:**
- Day 1: Setup + pilot (100 messages)
- Day 2: Core annotation (400 messages)
- Day 3: Training + evaluation
- Day 4-5: Active learning (optional)

---

## Annotation Guidelines to Prepare

### Entity Schema Guidelines
- Minimal span principle
- Nested entity handling
- Ambiguous brand resolution
- Phone/URL format variations

### Claim Schema Guidelines  
- Claim boundary definitions
- Overlapping claim handling
- Implicit vs explicit claims
- Edge case examples (50+ examples)

### Quality Metrics
- Inter-annotator agreement target: >80% for entities, >70% for claims
- Review frequency: Every 50 messages
- Difficult case discussion: Weekly

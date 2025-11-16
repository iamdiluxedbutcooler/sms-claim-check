# EDA Visualization Summary

**Generated:** November 17, 2025  
**Total Visualizations:** 46 (25 entity + 21 claim)

## Visualization Categories

### 1. Top Terms Analysis (by label)
Shows the most frequent terms/phrases for each annotation type.

**Entity annotations (8 files):**
- `entity_top_terms_ACTION_REQUIRED.png` - Most common action verbs
- `entity_top_terms_PHONE.png` - Common phone number patterns
- `entity_top_terms_BRAND.png` - Most targeted brands
- `entity_top_terms_URL.png` - Common URL patterns
- `entity_top_terms_AMOUNT.png` - Frequent monetary values
- `entity_top_terms_DEADLINE.png` - Urgency phrases
- `entity_top_terms_ORDER_ID.png` - Tracking number patterns
- `entity_top_terms_DATE.png` - Temporal references

**Claim annotations (6 files):**
- `claim_top_terms_ACTION_CLAIM.png` - Required actions
- `claim_top_terms_FINANCIAL_CLAIM.png` - Money-related claims
- `claim_top_terms_IDENTITY_CLAIM.png` - Sender identity claims
- `claim_top_terms_URGENCY_CLAIM.png` - Time pressure claims
- `claim_top_terms_ACCOUNT_CLAIM.png` - Account status claims
- `claim_top_terms_DELIVERY_CLAIM.png` - Package/delivery claims

### 2. Character Pattern Analysis (2 files)
Analyzes character-level features in annotations.

- `entity_character_patterns.png` - Entity patterns
  - Presence of numbers
  - Special characters
  - Case patterns (all caps, mixed case)
  - Length distributions by label

- `claim_character_patterns.png` - Claim patterns
  - Similar analysis for claim text

### 3. Co-occurrence Analysis (2 files)
Heatmaps showing which labels frequently appear together in messages.

- `entity_cooccurrence_heatmap.png` - Entity label co-occurrence
  - Shows which entities appear together (e.g., ACTION + PHONE)
  
- `claim_cooccurrence_heatmap.png` - Claim label co-occurrence
  - Shows which claim types co-occur (e.g., ACTION + FINANCIAL)

### 4. Message Pattern Analysis (2 files)
Statistical analysis of message characteristics.

- `entity_message_patterns.png` - Entity message analysis
  - Message length distribution
  - Annotations per message
  - Annotation density (per 100 chars)
  - Correlation: length vs annotations

- `claim_message_patterns.png` - Claim message analysis
  - Similar metrics for claims

### 5. N-gram Analysis

**Bigrams (2-word phrases):**
- Entity: 8 files (one per label)
- Claim: 6 files (one per label)

Examples:
- `entity_2grams_ACTION_REQUIRED.png` - "click here", "call now", etc.
- `claim_2grams_FINANCIAL_CLAIM.png` - "claim refund", "prize money", etc.

**Trigrams (3-word phrases):**
- Entity: 6 files (labels with enough data)
- Claim: 6 files (one per label)

Examples:
- `entity_3grams_BRAND.png` - Company name patterns
- `claim_3grams_ACTION_CLAIM.png` - Common action phrases

---

## Key Insights from Visualizations

### Entity Patterns

**Most Common:**
- ACTION_REQUIRED dominates (33.7% of all entities)
- Strong co-occurrence: ACTION + PHONE (call this number)
- Phone numbers appear in ~20% of annotations
- URLs often shortened (bit.ly patterns)

**Character Patterns:**
- 60%+ of entities contain numbers
- ACTION_REQUIRED rarely has numbers
- Brands often have mixed case
- Average entity length: 7.3 characters

### Claim Patterns

**Most Common:**
- ACTION_CLAIM and FINANCIAL_CLAIM make up 72.8% of claims
- Strong co-occurrence: ACTION + FINANCIAL (685 times)
- Claims are much longer than entities (avg 32 chars)

**Character Patterns:**
- Claims more likely to have special characters
- Lower percentage of all-caps compared to entities
- More diverse length distribution

### Message Patterns

**Entity messages:**
- Average: 140 characters, 4.95 entities
- Moderate correlation (0.38) between length and entity count
- Annotation density varies widely

**Claim messages:**
- Average: 140 characters, 2.89 claims
- Higher correlation (0.40) between length and claim count
- More consistent annotation density

---

## File Locations

```
data/eda/viz/
├── entity/                           # 25 files
│   ├── entity_character_patterns.png
│   ├── entity_cooccurrence_heatmap.png
│   ├── entity_message_patterns.png
│   ├── entity_top_terms_*.png        # 8 files
│   ├── entity_2grams_*.png           # 8 files
│   └── entity_3grams_*.png           # 6 files
└── claim/                            # 21 files
    ├── claim_character_patterns.png
    ├── claim_cooccurrence_heatmap.png
    ├── claim_message_patterns.png
    ├── claim_top_terms_*.png         # 6 files
    ├── claim_2grams_*.png            # 6 files
    └── claim_3grams_*.png            # 6 files
```

---

## How to Use These Visualizations

### For Research Paper
- Use co-occurrence heatmaps to show annotation patterns
- Include top terms to demonstrate dataset characteristics
- Show message patterns to justify modeling decisions

### For Model Development
- Top terms reveal what the model needs to learn
- N-grams show phrase patterns for sequence modeling
- Character patterns inform tokenization strategy

### For Error Analysis
- Compare model predictions against frequency distributions
- Identify if model misses rare but important patterns
- Use co-occurrence to understand context dependencies

---

## Next Steps

1. **Optional: Install wordcloud**
   ```bash
   pip install wordcloud
   python scripts/eda_visualizations.py --entity data/annotations/entity_annotations.json --claim data/annotations/claim_annotations.json
   ```
   This will add word cloud visualizations (prettier but not essential).

2. **Review Key Visualizations**
   - `*_cooccurrence_heatmap.png` - Understand label relationships
   - `*_top_terms_*.png` - See what model needs to learn
   - `*_message_patterns.png` - Statistical overview

3. **Use in Training**
   - Inform data augmentation strategies
   - Guide hyperparameter selection
   - Identify class imbalance issues

---

## Visualization Quality

All visualizations:
- High resolution (300 DPI)
- Publication-ready
- Clear labels and titles
- Professional color schemes
- Grid lines for readability

**Note:** Word clouds were skipped (wordcloud not installed). These are nice-to-have but not essential for analysis.

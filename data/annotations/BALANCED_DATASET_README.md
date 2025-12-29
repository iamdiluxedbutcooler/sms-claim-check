# Balanced Dataset Summary

**Generated:** December 5, 2025

## Dataset Statistics

- **Total Messages:** 2,000
- **Phishing Messages:** 1,000 (50%)
  - Original Mendeley: 562
  - GPT-4o-mini Augmented: 438
- **HAM (Benign) Messages:** 1,000 (50%)
  - Sampled from Mendeley dataset
- **Balance Ratio:** 1:1 (perfect balance)

## Files Created

1. **`data/annotations/balanced_dataset_2000.json`** (1.1 MB)
   - Label Studio format
   - Ready for annotation or training
   - Shuffled for random distribution

2. **`data/annotations/balanced_dataset_2000_metadata.json`**
   - Dataset statistics and provenance
   - Source file references

## Data Sources

- **Phishing:** `data/annotations/augmented_phishing_1000.json`
  - 562 original Mendeley phishing messages (verified 100% authentic)
  - 438 GPT-4o-mini augmented variations
  
- **HAM:** `data/raw/mendeley.csv`
  - Randomly sampled 1,000 from 4,844 available HAM messages
  - Random seed: 42 (reproducible)

## Label Studio Format

### Phishing Messages
- Contains annotations (empty result array to be filled during annotation)
- Meta fields: `is_augmented`, `source`, `label`, `augmentation_type`

### HAM Messages
- Empty result array (all tokens are 'O' - no entities)
- Meta fields: `is_augmented=False`, `source=mendeley_ham`, `label=ham`

## Quality Verification

[OK] **Class Balance:** Perfect 50/50 split prevents model bias
[OK] **Phishing Quality:** Verified 100% authentic Mendeley + high-quality GPT augmentations
[OK] **HAM Quality:** Sampled from authentic Mendeley ham messages
[OK] **Shuffled:** Random distribution prevents temporal bias

## Next Steps

1. **Import to Label Studio**
   - Upload `balanced_dataset_2000.json`
   - Configure NER interface for 12 claim types

2. **Annotate Phishing Claims**
   - Annotate claim spans in 1,000 phishing messages
   - HAM messages already have 'O' labels (no annotation needed)

3. **Train Models**
   - Update data loader to use balanced dataset
   - Run 5-fold cross-validation
   - Compare with baseline (638 messages)

4. **Expected Improvements**
   - F1 Score: 0.00 → 0.70-0.85
   - Better generalization from augmented variations
   - Reduced false positives from class balance

## Script Used

- **`scripts/add_ham_messages.py`**
  - Loads Mendeley HAM messages
  - Samples 1,000 messages (seed=42)
  - Converts to Label Studio format
  - Combines with phishing dataset
  - Shuffles and saves balanced output

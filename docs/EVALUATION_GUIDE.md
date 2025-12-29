# Model Evaluation Script

Comprehensive evaluation and analytics for NER model performance.

## Installation

```bash
pip install -r requirements_evaluation.txt
```

## Usage

```bash
python scripts/evaluate_model_performance.py <path_to_test_results_detailed.json> [--output-dir OUTPUT_DIR]
```

### Example

```bash
python scripts/evaluate_model_performance.py data/results/approach5/test_results_detailed.json --output-dir evaluation_results
```

## Outputs

The script generates:

### 1. Visualizations
- **confusion_matrix.png** - TP/FP/FN breakdown per claim type
- **performance_by_type.png** - Precision/Recall/F1 bar chart per claim type
- **support_distribution.png** - Number of instances per claim type
- **confidence_distribution.png** - Distribution of model confidence scores

### 2. Reports
- **evaluation_report.txt** - Comprehensive text report with:
  - Overall statistics
  - Per-type metrics (Precision, Recall, F1, Support)
  - Error analysis
  - Sample false positives and false negatives
  - Type confusion matrix

### 3. Metrics
- **metrics_summary.json** - Machine-readable metrics summary

## Metrics Explained

- **Precision**: Of all predicted claims, how many were correct?
- **Recall**: Of all actual claims, how many did the model find?
- **F1 Score**: Harmonic mean of Precision and Recall
- **Support**: Number of actual instances in test set
- **True Positives (TP)**: Correctly identified claims
- **False Positives (FP)**: Incorrectly identified claims
- **False Negatives (FN)**: Missed claims

## Example Output

```
MODEL EVALUATION REPORT
================================================================================

OVERALL STATISTICS
--------------------------------------------------------------------------------
Total test messages: 400
Total ground truth claims: 1847
Total predicted claims: 1823
Perfect matches (exact count): 306 (76.5%)

Messages with claims (GT): 200
Messages with claims (Pred): 198
Avg claims per message (GT): 4.62
Avg claims per message (Pred): 4.56

OVERALL METRICS
--------------------------------------------------------------------------------
Precision: 0.876
Recall:    0.865
F1 Score:  0.870

PER-TYPE METRICS
--------------------------------------------------------------------------------
Type                      Precision    Recall       F1           Support   
--------------------------------------------------------------------------------
ACTION_CLAIM              0.912        0.898        0.905        523
URGENCY_CLAIM             0.834        0.821        0.827        412
REWARD_CLAIM              0.889        0.876        0.882        387
...
```

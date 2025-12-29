#!/bin/bash
# Quick evaluation setup and run

echo "Setting up evaluation environment..."

# Install dependencies
pip install -q -r requirements_evaluation.txt

echo ""
echo "Evaluation script ready!"
echo ""
echo "Usage:"
echo "  python scripts/evaluate_model_performance.py <path_to_test_results_detailed.json>"
echo ""
echo "Example:"
echo "  python scripts/evaluate_model_performance.py data/results/test_results_detailed.json --output-dir evaluation_output"
echo ""
echo "The script will generate:"
echo "  - 4 visualization PNG files"
echo "  - 1 comprehensive text report"
echo "  - 1 JSON metrics summary"
echo ""

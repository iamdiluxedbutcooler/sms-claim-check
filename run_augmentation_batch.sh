#!/bin/bash
# Runner script for batch augmentation

echo "SMS Phishing Dataset Augmentation Pipeline (BATCH API - 50% cheaper!)"
echo "======================================================================"
echo ""

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='sk-...'"
    echo ""
    exit 1
fi

# Install requirements if needed
echo "Checking dependencies..."
pip install -q -r requirements_augmentation.txt

echo ""

# Check if we're retrieving results
if [ "$1" == "--retrieve" ]; then
    echo "Retrieving batch results..."
    echo ""
    python scripts/augment_dataset_batch.py --retrieve $2
else
    echo "Submitting batch job..."
    echo ""
    python scripts/augment_dataset_batch.py
fi

echo ""
echo "Done!"

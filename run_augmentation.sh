#!/bin/bash
# Runner script for dataset augmentation

echo "SMS Phishing Dataset Augmentation Pipeline"
echo "=============================================="
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
echo "Running augmentation pipeline..."
echo ""

# Run the augmentation script
python scripts/augment_dataset.py

echo ""
echo "Done! Check data/annotations/augmented_phishing_1000.json"

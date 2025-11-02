#!/bin/bash

set -e

echo "================================================"
echo "SMS Phishing Detection - Multi-Model Training"
echo "================================================"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements_new.txt

echo ""
echo "================================================"
echo "Starting Training Pipeline"
echo "================================================"

echo ""
echo ">>> Training Approach 1: Entity-First NER Pipeline <<<"
python train.py --config configs/entity_ner.yaml

echo ""
echo ">>> Training Approach 2: Claim-Phrase NER Pipeline <<<"
python train.py --config configs/claim_ner.yaml

echo ""
echo ">>> Training Approach 3: Hybrid NER + LLM Pipeline <<<"
python train.py --config configs/hybrid_llm.yaml

echo ""
echo ">>> Training Approach 4: Contrastive Learning <<<"
python train.py --config configs/contrastive.yaml

echo ""
echo "================================================"
echo "All Training Complete!"
echo "================================================"
echo "Results saved in experiments/ directory"
echo ""
echo "To compare models, run:"
echo "  python scripts/compare_models.py"

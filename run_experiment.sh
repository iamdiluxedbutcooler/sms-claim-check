#!/bin/bash

set -e

if [ $# -eq 0 ]; then
    echo "Usage: ./run_experiment.sh <config_name>"
    echo "Available configs: entity_ner, claim_ner, hybrid_llm, contrastive"
    exit 1
fi

CONFIG_NAME=$1
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -1 configs/*.yaml | xargs -n 1 basename | sed 's/.yaml//'
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements_new.txt
    touch venv/.installed
fi

echo "================================================"
echo "Training: $CONFIG_NAME"
echo "Config: $CONFIG_FILE"
echo "================================================"

python train.py --config "$CONFIG_FILE"

echo ""
echo "================================================"
echo "Training Complete!"
echo "================================================"

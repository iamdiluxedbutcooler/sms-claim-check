#!/bin/bash

set -e

echo "================================"
echo "NER Model Training Setup"
echo "================================"

echo -e "\nChecking Python version..."
python3 --version

echo -e "\nCreating virtual environment..."
python3 -m venv venv_ner
source venv_ner/bin/activate

echo -e "\nUpgrading pip..."
pip install --upgrade pip

echo -e "\nInstalling training dependencies..."
pip install -r requirements_training.txt

echo -e "\nChecking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo -e "\n================================"
echo "Setup complete!"
echo "================================"
echo -e "\nTo activate environment: source venv_ner/bin/activate"
echo "To start training: python scripts/train_ner.py"
echo "================================"

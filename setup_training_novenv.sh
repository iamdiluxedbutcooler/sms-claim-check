#!/bin/bash

set -e

echo "================================"
echo "NER Model Training Setup"
echo "================================"

echo -e "\nChecking Python version..."
python3 --version

echo -e "\nInstalling training dependencies in user directory..."
python3 -m pip install --user --upgrade pip

echo -e "\nInstalling PyTorch and dependencies..."
python3 -m pip install --user -r requirements_training.txt

echo -e "\nChecking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null || echo "PyTorch installed, GPU check will run during training"

echo -e "\n================================"
echo "Setup complete!"
echo "================================"
echo -e "\nTo start training: python3 scripts/train_ner.py"
echo "================================"

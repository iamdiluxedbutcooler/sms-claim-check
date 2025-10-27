#!/bin/bash

echo "========================================="
echo "Installing packages in background..."
echo "========================================="

cd ~/sms-claim-check

nohup python3 -m pip install --user torch transformers datasets evaluate scikit-learn seqeval accelerate > install_packages.log 2>&1 &

PID=$!
echo "Installation PID: $PID"
echo "Monitor progress: tail -f install_packages.log"
echo ""
echo "After installation completes (check with: ps -p $PID), run:"
echo "  nohup python3 scripts/train_ner.py > training.log 2>&1 &"
echo "========================================="

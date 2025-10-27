#!/bin/bash

SERVER="pleng@echo-dongjs.ddns.comp.nus.edu.sg"
REMOTE_DIR="~/sms-claim-check"

echo "================================"
echo "Deploying to Remote Server"
echo "================================"

echo -e "\nServer: $SERVER"
echo "Remote directory: $REMOTE_DIR"

echo -e "\nUploading project files..."
rsync -avz --exclude 'venv*' --exclude '__pycache__' --exclude '.git' \
  --exclude '*.pyc' --exclude '.DS_Store' \
  /Users/pleng/Desktop/scammers/sms-claim-check/ \
  $SERVER:$REMOTE_DIR/

echo -e "\n================================"
echo "Upload complete!"
echo "================================"

echo -e "\nNext steps:"
echo "1. SSH to server: ssh $SERVER"
echo "2. cd $REMOTE_DIR"
echo "3. ./setup_training.sh"
echo "4. source venv_ner/bin/activate"
echo "5. python scripts/train_ner.py"
echo "================================"

## Quick Start Commands

### 1. Upload to Server
```bash
cd /Users/pleng/Desktop/scammers
scp -r sms-claim-check pleng@echo-dongjs.ddns.comp.nus.edu.sg:~/
```

### 2. SSH to Server
```bash
ssh pleng@echo-dongjs.ddns.comp.nus.edu.sg
```

### 3. Setup Environment
```bash
cd ~/sms-claim-check
./setup_training.sh
source venv_ner/bin/activate
```

### 4. Start Training
```bash
# Default (DistilBERT, 10 epochs)
python scripts/train_ner.py

# Or with custom settings
python scripts/train_ner.py --model bert-base-uncased --epochs 15 --batch_size 32
```

### 5. Monitor Progress
```bash
# Training will show:
# - Epoch progress
# - Loss values
# - Validation metrics (precision, recall, F1)
# - Best model checkpoint
```

### 6. Test the Model
```bash
# Interactive mode
python scripts/inference_ner.py

# Single message
python scripts/inference_ner.py --text "Call 08001234567 to claim your £500 prize now!"

# Batch file
python scripts/inference_ner.py --file test_messages.txt
```

### 7. Download Model
```bash
# From local machine
scp -r pleng@echo-dongjs.ddns.comp.nus.edu.sg:~/sms-claim-check/models/ner/final_model ./models/
```

## Expected Output

Training will produce:
- Final test F1 score: ~75-85%
- Trained model saved to: `models/ner/final_model/`
- Test results: `models/ner/test_results.json`

Dataset split:
- Train: 342 messages
- Val: 87 messages  
- Test: 81 messages

Total entities: 1,900 across 9 types

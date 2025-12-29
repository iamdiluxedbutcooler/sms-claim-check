# Updated Training Notebooks - December 2025

## Changes Made

All 4 training notebooks have been updated with the following improvements:

### 1. Google Drive Integration
- Models automatically save to Google Drive: `/content/drive/MyDrive/sms_claim_models/`
- Each approach has its own folder:
  - `approach1_entity_ner/`
  - `approach2_claim_ner/`
  - `approach3_hybrid_llm/`
  - `approach4_contrastive/`
- Models persist even after Colab session ends
- Easy access from any device with your Google Drive

### 2. Training Checkpoints
- **Checkpoints saved every 50 steps** (instead of per epoch)
- Keeps last 5 checkpoints to save space
- Can resume training from any checkpoint if interrupted
- Better granularity for monitoring progress

### 3. Progress Bars & Logging
- Visual progress bars during training (tqdm)
- Logging every 10 steps (instead of 50)
- First step always logged
- Clear visibility into training progress

### 4. Clean Output
- All emojis removed for professional output
- Cleaner console logs
- Better readability in notebooks

---

## File Structure After Training

```
Google Drive/
└── MyDrive/
    └── sms_claim_models/
        ├── approach1_entity_ner/
        │   ├── checkpoint-50/
        │   ├── checkpoint-100/
        │   ├── checkpoint-150/
        │   ├── ...
        │   └── final_model/
        │       ├── config.json
        │       ├── model.safetensors
        │       ├── tokenizer_config.json
        │       └── training_info.json
        ├── approach2_claim_ner/
        │   └── final_model/
        ├── approach3_hybrid_llm/
        │   └── final_model/
        └── approach4_contrastive/
            └── final_model/
```

---

## Training Progress Indicators

During training, you'll see:

```
Epoch 1/5:  20% [=====>           ] 50/250 steps | loss: 0.543
Epoch 1/5:  40% [==========>      ] 100/250 steps | loss: 0.421
Epoch 1/5:  60% [===============> ] 150/250 steps | loss: 0.385
...
```

Plus evaluation metrics every 50 steps:
```
Step 50:  eval_loss: 0.512 | eval_f1: 0.723 | eval_precision: 0.745
Step 100: eval_loss: 0.423 | eval_f1: 0.812 | eval_precision: 0.831
...
```

---

## How to Use

### Starting Training
1. Open notebook in Google Colab
2. Connect to GPU runtime (Runtime → Change runtime type → GPU)
3. Run first cell to mount Google Drive (authorize when prompted)
4. Upload data file when prompted
5. Run all cells

### Monitoring Progress
- Watch the progress bar in real-time
- Check eval metrics every 50 steps
- Training logs show loss decreasing
- Progress saved automatically to Drive

### After Training
Models are saved in **two places**:
1. **Google Drive** - Permanent storage, accessible anytime
2. **Local Colab** - Temporary, available for immediate download as ZIP

### Resuming Interrupted Training
If training stops, you can resume from last checkpoint:
```python
from transformers import Trainer

# Load from checkpoint
trainer = Trainer(...)
trainer.train(resume_from_checkpoint="./claim-ner-model/checkpoint-150")
```

---

## Checkpoint Strategy

**Why checkpoints every 50 steps?**
- Typical training: ~400-500 steps per epoch
- 50 steps = ~10-12% of an epoch
- Checkpoints at: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
- Fine-grained progress tracking
- Minimal risk if training interrupted

**Why keep only 5 checkpoints?**
- Each checkpoint ~450MB
- 5 checkpoints = ~2.25GB
- Saves Drive storage space
- Still have recent checkpoints if needed

---

## Training Time Estimates

With Colab T4 GPU:

| Approach | Data Size | Training Time | Checkpoints Created |
|----------|-----------|---------------|---------------------|
| Approach 1 | 2000 msgs | ~20-25 min | ~20 checkpoints |
| Approach 2 | 2000 msgs | ~20-25 min | ~20 checkpoints |
| Approach 3 | 2000 msgs | ~25-30 min | ~20 + LLM calls |
| Approach 4 | 2000 msgs | ~15-20 min | ~15 checkpoints |

---

## Model Download Options

### Option 1: From Google Drive (Recommended)
- Navigate to Drive folder
- Download entire `final_model/` folder
- Use anytime, even months later

### Option 2: Immediate Download
- ZIP file created at end of training
- Download directly from Colab
- ~450MB compressed file

### Option 3: Copy to Local Drive
```python
# From Colab
from google.colab import files
import shutil

# Zip the model
shutil.make_archive('my_model', 'zip', './final_model')
files.download('my_model.zip')
```

---

## Troubleshooting

### "Out of Memory" Error
```python
# Reduce batch size in Training Arguments
per_device_train_batch_size=8  # instead of 16
per_device_eval_batch_size=8
```

### Training Too Slow
- Make sure GPU is enabled (Runtime → Change runtime type)
- Check GPU usage: `!nvidia-smi`
- Consider using `distilroberta-base` instead of `roberta-base`

### Checkpoints Taking Too Much Space
```python
# Reduce checkpoint frequency
save_steps=100  # instead of 50
save_total_limit=3  # instead of 5
```

### Can't Mount Google Drive
- Clear browser cache
- Re-authorize Google account
- Try in incognito mode

---

## Example: Complete Training Session

```
1. Start notebook
2. Mount Google Drive (10 seconds)
3. Upload data file (5 seconds)
4. Load and preprocess data (30 seconds)
5. Training starts:
   
   Epoch 1/5:
   [==>   ] 50/250  | 2 min elapsed | eval_f1: 0.723
   [====> ] 100/250 | 4 min elapsed | eval_f1: 0.812
   [======>] 150/250 | 6 min elapsed | eval_f1: 0.845
   ...
   
   Epoch 5/5 complete: 20 minutes total
   
6. Final evaluation (30 seconds)
7. Save to Drive (1 minute)
8. Create download ZIP (30 seconds)

Total time: ~22 minutes
```

---

## Backup Information

Original notebooks backed up to: `notebooks_backup/`

If you need to revert changes:
```bash
cp notebooks_backup/approach2_claim_phrase_ner.ipynb .
```

---

## Next Steps

1. **Start with Approach 2** (Claim-Phrase NER) - Most useful
2. **Then train Approach 4** (HAM/SMISH Classifier) - Good for filtering
3. **Optional**: Train Approach 1 or 3 for comparison

All models will be safely stored in your Google Drive for future use!

---

## Questions?

- Need to change checkpoint frequency? Edit `save_steps` parameter
- Want more/less logging? Edit `logging_steps` parameter
- Different Drive location? Change `save_dir` variable
- Training too slow? Reduce `per_device_train_batch_size`

Happy training!

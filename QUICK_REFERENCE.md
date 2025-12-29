# Training Notebooks Quick Reference

## What Changed

ALL DONE:
- [x] Emojis removed from all outputs
- [x] Google Drive auto-save configured  
- [x] Progress bars enabled
- [x] Checkpoints every 50 steps
- [x] Keep last 5 checkpoints
- [x] Backup created

---

## Quick Start

```
1. Open notebook in Colab
2. Runtime → Change runtime type → GPU (T4)
3. Run all cells (Ctrl+F9)
4. Authorize Google Drive when prompted
5. Upload data file
6. Wait ~20 minutes
7. Model saved to Drive automatically
```

---

## Where Your Models Are Saved

```
Google Drive → MyDrive → sms_claim_models/
  ├── approach1_entity_ner/final_model/
  ├── approach2_claim_ner/final_model/
  ├── approach3_hybrid_llm/final_model/
  └── approach4_contrastive/final_model/
```

Access anytime, from any device!

---

## Training Progress

You'll see:
```
Training: [=========>    ] 45%
Step 100/250 | Loss: 0.421 | Time: 4:23 / 9:40
Eval: precision=0.831 recall=0.794 f1=0.812
```

Checkpoints saved at steps: 50, 100, 150, 200, ...

---

## Recommended Training Order

1. **approach2_claim_phrase_ner.ipynb** (BEST - start here)
2. **approach4_contrastive_classification.ipynb** (for filtering)
3. **approach1_entity_first_ner.ipynb** (if you need entities)
4. **approach3_hybrid_claim_llm.ipynb** (needs API key)

---

## Files to Upload

- Approach 1: `entity_annotations_2000.json`
- Approach 2, 3, 4: `claim_annotations_2000.json`

Located in: `data/annotations/`

---

## Troubleshooting

**Out of memory?**
→ Reduce batch size to 8 in TrainingArguments

**Too slow?**
→ Check GPU is enabled: Runtime → Change runtime type

**Drive not mounting?**
→ Clear browser cache, re-authorize

---

## After Training

Download options:
1. From Google Drive (permanent)
2. ZIP file created in Colab (immediate)
3. Access via `!cp` commands

Model size: ~450MB per approach

---

## Compare All 4 Approaches

| Approach | Task | Time | Use When |
|----------|------|------|----------|
| 1 | Extract entities | 20m | Need entity data |
| 2 | Extract claims | 20m | **Best overall** |
| 3 | Claims + LLM | 30m | Need SPOT format |
| 4 | Classify ham/smish | 15m | Pre-screening |

---

All notebooks ready to run on Google Colab!
Original versions backed up in `notebooks_backup/`

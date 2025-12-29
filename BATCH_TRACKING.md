# Batch Job Tracking

## Entity-Based Annotation Job

**Submitted:** November 17, 2025
**Batch ID:** `batch_691a2e5b747881909a32552782d6e38f`
**Model:** GPT-4o
**Messages:** 638 smishing messages
**Status:** Validating → In Progress → Complete (expected ~24h)

**Check Status:**
```bash
python scripts/ai_preannotate_entities.py --check-status batch_691a2e5b747881909a32552782d6e38f
```

**Download Results:**
```bash
python scripts/ai_preannotate_entities.py --download batch_691a2e5b747881909a32552782d6e38f
```

**Output File:** `data/annotations/entity_annotations.json`

---

## Claim-Based Annotation Job

**Submitted:** November 17, 2025
**Batch ID:** `batch_691a2f1215a48190806fcbe64a420aff`
**Model:** GPT-4o
**Messages:** 638 smishing messages
**Status:** Validating → In Progress → Complete (expected ~24h)

**Check Status:**
```bash
python scripts/ai_preannotate_claims.py --check-status batch_691a2f1215a48190806fcbe64a420aff
```

**Download Results:**
```bash
python scripts/ai_preannotate_claims.py --download batch_691a2f1215a48190806fcbe64a420aff
```

**Output File:** `data/annotations/claim_annotations.json`

---

## Timeline

- **T+0 (Now):** Batch jobs submitted, status = "validating"
- **T+30min:** Check status, should be "in_progress"
- **T+12h:** Check progress
- **T+24h:** Jobs should be complete, download results
- **T+24h+1h:** Run EDA on annotations
- **T+48h:** Begin model training

---

## Quick Commands

### Check both jobs at once:
```bash
python scripts/ai_preannotate_entities.py --check-status batch_691a2e5b747881909a32552782d6e38f
python scripts/ai_preannotate_claims.py --check-status batch_691a2f1215a48190806fcbe64a420aff
```

### Download both when complete:
```bash
python scripts/ai_preannotate_entities.py --download batch_691a2e5b747881909a32552782d6e38f
python scripts/ai_preannotate_claims.py --download batch_691a2f1215a48190806fcbe64a420aff
```

### Run EDA after download:
```bash
python scripts/eda_comprehensive.py \
  --entity data/annotations/entity_annotations.json \
  --claim data/annotations/claim_annotations.json \
  --compare \
  --output data/eda
```

---

## Batch API Info

- **Completion Window:** 24 hours
- **Cost:** ~50% discount compared to standard API
- **Estimated Cost:** $30-50 for both jobs
- **Progress Tracking:** Check status every few hours

---

## Notes

- Batch IDs are also saved in:
  - `data/annotations/entity_batch_id.txt`
  - `data/annotations/claim_batch_id.txt`

- Metadata files contain message details:
  - `data/annotations/entity_batch_input_metadata.json`
  - `data/annotations/claim_batch_input_metadata.json`

- If batch fails, re-run with:
  ```bash
  python scripts/ai_preannotate_entities.py --submit
  python scripts/ai_preannotate_claims.py --submit
  ```

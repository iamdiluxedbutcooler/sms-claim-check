# Annotation Quality Control Strategy

## Problem Identified

You're right - **IDENTITY_CLAIM is being over-annotated**. Things like "Dear Valued Customer" are NOT identity claims. An identity claim should be:
- "We are Amazon" (explicit identity)
- "From PayPal Support" (explicit sender)
- "IRS Tax Department" (authority claim)

NOT greetings or salutations!

## Quality Control Options

### Option 1: GPT-4o Cross-Validation (More Accurate, More Expensive)

**Script:** `scripts/quality_control_annotations.py`

**How it works:**
1. Sample 150 messages from the 2000 annotations
2. Re-annotate with GPT-4o (smarter, more careful model)
3. Compare GPT-4o-mini vs GPT-4o annotations
4. Generate quality report with disagreements

**Cost:** ~$0.50-1.00 per 100 messages

**Pros:**
- High quality validation
- GPT-4o is more accurate and conservative
- Detailed reasoning for each claim

**Cons:**
- More expensive
- Slower

**To run:**
```bash
export $(cat .env | grep OPENAI_API_KEY | xargs)
python3 scripts/quality_control_annotations.py
```

---

### Option 2: Self-Consistency Check (Cheaper, Still Effective)

**Script:** `scripts/self_consistency_check.py`

**How it works:**
1. Sample 100-200 messages
2. Run GPT-4o-mini 3 times on each message (with temperature=0.3 for variation)
3. Check if the model is consistent with itself
4. Flag messages where annotations differ across runs

**Cost:** ~$0.06-0.12 per 100 messages (83% cheaper!)

**Logic:** 
- If the model gives different answers on the same message, it's uncertain
- High uncertainty = likely annotation error
- Inconsistent IDENTITY_CLAIMs are flagged as suspicious

**Pros:**
- Much cheaper (uses GPT-4o-mini 3x instead of GPT-4o 1x)
- Fast
- Good at finding uncertain/problematic annotations

**Cons:**
- Less authoritative than GPT-4o
- May miss systematic errors (if model is consistently wrong)

**To run:**
```bash
export $(cat .env | grep OPENAI_API_KEY | xargs)
python3 scripts/self_consistency_check.py
```

---

### Option 3: Hybrid Approach (Recommended)

1. **First:** Run self-consistency check on 200 messages (~$0.25)
   - Identify low-consistency messages
   - Flag suspicious IDENTITY_CLAIMs
   
2. **Then:** Run GPT-4o validation on just the problematic cases (~50 messages, ~$0.25)
   - Get authoritative answers for uncertain cases
   
**Total cost:** ~$0.50 for 200-250 message validation

---

## Expected Issues to Find

Based on your observation, the quality control will likely find:

1. **IDENTITY_CLAIM over-annotation:**
   - "Dear Customer" → NOT identity claim
   - "Valued member" → NOT identity claim
   - "Important notice" → NOT identity claim
   - Only "We are Amazon" or implicit "Amazon:" → YES identity claim

2. **ACTION_CLAIM confusion:**
   - Descriptive text vs actual call-to-action
   - "You can click here" vs "Click here now"

3. **URGENCY_CLAIM vs DATE:**
   - "Tomorrow" → DATE or URGENCY?
   - Depends on context

---

## What Happens After Quality Control?

Two options:

### A. Filter & Retrain (Recommended)
1. Remove low-quality annotations
2. Keep only high-confidence annotations (consistency > 0.8)
3. Train on cleaner, smaller dataset (maybe 800-1000 good annotations)
4. Better quality > quantity

### B. Manual Correction
1. Use quality report to identify problematic patterns
2. Write correction rules (e.g., remove "Dear *" IDENTITY_CLAIMs)
3. Apply rules to all 2000 annotations
4. Faster than full re-annotation

---

## Recommendation

Since the main annotation is still running, I suggest:

1. **Let it finish** (you'll have baseline annotations)

2. **Run self-consistency check FIRST** (cheap, fast):
   ```bash
   python3 scripts/self_consistency_check.py
   ```
   - Will identify ~20-30% problematic annotations
   - Cost: ~$0.25 for 100 messages

3. **Review the consistency report:**
   - See what issues are found
   - Decide if you need GPT-4o validation or can fix with rules

4. **If needed, run GPT-4o on problem cases:**
   ```bash
   python3 scripts/quality_control_annotations.py
   ```

5. **Filter or correct based on findings**

---

## Cost Comparison

For 200 messages validation:

| Method | Cost | Time | Quality |
|--------|------|------|---------|
| **Self-consistency (3x mini)** | $0.12 | 30 min | Good |
| **GPT-4o validation** | $1.00 | 30 min | Excellent |
| **Hybrid (50 self + 50 GPT-4o)** | $0.31 | 20 min | Excellent |
| **Manual review** | $0 | 4+ hours | Perfect but tedious |

---

## Next Steps

Once the main annotation finishes (~2 hours), you can:

1. Run quality control (30 min)
2. Review report (15 min)
3. Decide: filter, correct, or accept
4. Proceed to training

The annotation is running in the background, so these scripts are ready when you need them!

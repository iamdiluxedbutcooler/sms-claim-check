# SMS Claim Extraction - Training Notebooks

This directory contains **4 Google Colab notebooks** for training different claim extraction models. Each approach has different strengths and use cases.

## 📓 Notebooks Overview

### ✅ **RECOMMENDED: `approach2_claim_phrase_ner.ipynb`**
**Claim-Phrase NER (RoBERTa-based)**

**What it does:** Directly extracts claim phrases from SMS messages using NER

**Output Example:**
```
Text: "Your Amazon package is delayed. Click here urgently."
Claims:
  - IDENTITY_CLAIM: "Amazon" 
  - DELIVERY_CLAIM: "package is delayed"
  - ACTION_CLAIM: "Click here"
  - URGENCY_CLAIM: "urgently"
```

**Pros:**
- ✅ Direct semantic capture of claims
- ✅ More robust to variations
- ✅ Handles implicit claims
- ✅ Perfect for your verification pipeline
- ✅ Works with 12 claim types

**When to use:** **START HERE!** This is the best approach for your use case.

---

### 📊 **`approach4_contrastive_classification.ipynb`**
**Binary Classification (HAM vs SMISH)**

**What it does:** Classifies if a message is legitimate (HAM) or scam (SMISH)

**Output Example:**
```
Text: "Your Amazon package is delayed..."
Prediction: SMISH (95% confidence)
```

**Pros:**
- ✅ Simple binary answer
- ✅ Fast inference
- ✅ Good for first-line screening
- ✅ High accuracy

**When to use:** Use as a **pre-filter** before claim extraction, or for simple ham/smish detection.

---

### 🔧 **`approach1_entity_first_ner.ipynb`**
**Entity-First NER + Claim Parsing**

**What it does:** First extracts entities (BRAND, PHONE, URL), then parses them into claims

**Output Example:**
```
Step 1 - Extract Entities:
  - BRAND: "Amazon"
  - PHONE: "0800-123-456"
  
Step 2 - Parse to Claims:
  - IDENTITY_CLAIM: "Amazon"
  - ACTION_CLAIM: "Call 0800-123-456"
```

**Pros:**
- ✅ Entities are concrete and well-defined
- ✅ Can reuse entity extraction
- ✅ Clear intermediate representation

**Cons:**
- ❌ Two-step process (more complex)
- ❌ May miss implicit claims without explicit entities
- ❌ Less robust to OOD data

**When to use:** If you specifically need entity extraction for other purposes.

---

### 🤖 **`approach3_hybrid_claim_llm.ipynb`**
**Claim-Phrase NER + LLM Restructuring**

**What it does:** Extracts claims with NER, then uses LLM to structure them in SPOT format (Subject-Predicate-Object-Time)

**Output Example:**
```
Step 1 - Extract Claims (NER):
  - IDENTITY_CLAIM: "Amazon"
  - DELIVERY_CLAIM: "package is delayed"
  
Step 2 - LLM Restructure to SPOT:
  - Subject: "Amazon"
  - Predicate: "claims delivery is"
  - Object: "delayed"
  - Time: "now"
```

**Pros:**
- ✅ Most structured output
- ✅ Good for complex verification logic
- ✅ Combines neural + LLM strengths

**Cons:**
- ❌ Requires LLM API (costs)
- ❌ Slower inference
- ❌ More complex pipeline

**When to use:** If you need highly structured claims for complex verification agent.

---

## 🎯 Quick Decision Guide

**Start with:** `approach2_claim_phrase_ner.ipynb` ← **Best for your use case**

**Then train:** `approach4_contrastive_classification.ipynb` ← For pre-filtering

**Optional:** Try Approach 1 or 3 if you have specific needs

---

## 🚀 How to Use

1. **Open in Google Colab**
   - Upload notebook to Colab
   - Or use: `Open in Colab` button (if viewing on GitHub)

2. **Upload Data**
   - Upload `claim_annotations_2000.json` when prompted
   - For Approach 1, also upload `entity_annotations_2000.json`

3. **Run All Cells**
   - Click `Runtime` → `Run all`
   - Training takes 15-30 minutes on free T4 GPU

4. **Download Model**
   - Model saved in notebook
   - Optional: download .zip file

---

## 📊 Expected Performance

| Approach | Task | Expected F1 | Training Time | Inference Speed |
|----------|------|-------------|---------------|-----------------|
| Approach 2 | Claim NER | 85-90% | ~20 min | Fast |
| Approach 4 | Classification | 95-98% | ~15 min | Very Fast |
| Approach 1 | Entity NER | 80-85% | ~20 min | Fast |
| Approach 3 | Hybrid | 85-90% | ~30 min | Slow (LLM) |

---

## 🔧 Requirements

- **Google Colab** (Free tier is fine, GPU recommended)
- **Data Files:**
  - `claim_annotations_2000.json` (for Approaches 2, 3, 4)
  - `entity_annotations_2000.json` (for Approach 1)

---

## 💡 Recommendations

### For Production Pipeline:
```
1. Run Approach 4 (HAM/SMISH classifier) first
   ↓
2. If SMISH, run Approach 2 (Claim extraction)
   ↓
3. Send claims to verification agent
```

### For Research/Comparison:
Train all 4 approaches and compare performance!

---

## 📝 Notes

- All notebooks are **self-contained** and **ready to run**
- No local setup needed - everything runs on Colab
- Models use **RoBERTa-base** (can switch to `distilroberta-base` for faster training)
- Data is automatically split: 70% train, 15% val, 15% test

---

## ❓ Questions?

- **Which approach to start?** → Approach 2 (Claim-Phrase NER)
- **Need binary classification?** → Approach 4
- **Want entity extraction?** → Approach 1
- **Need structured SPOT format?** → Approach 3

**Happy Training! 🚀**

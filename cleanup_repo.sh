#!/bin/bash
# Repository Cleanup Script
# Generated: December 29, 2025
# Removes outdated/duplicate scripts identified in CLEANUP_REPORT.md

echo "🧹 Starting repository cleanup..."
echo ""

# Create backup first
echo "📦 Creating backup..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Count files before
BEFORE_COUNT=$(find . -type f -name "*.py" | wc -l)

echo "📊 Before cleanup: $BEFORE_COUNT Python files"
echo ""

# 1. Remove outdated training scripts
echo "🗑️  Removing outdated training scripts..."
rm -f train.py
rm -f train_kfold.py
rm -f scripts/train_ner.py

# 2. Remove one-time data processing scripts
echo "🗑️  Removing one-time data processing scripts..."
rm -f augment_with_gpt.py
rm -f fix_missing_annotations.py
rm -f fix_compressed_cell.py
rm -f remove_comments.py

# 3. Remove duplicate cleanup scripts (cleanup is done)
echo "🗑️  Removing duplicate cleanup scripts..."
rm -f check_duplicates_strict.py
rm -f find_specific_duplicate.py
rm -f remove_near_duplicates.py
rm -f smart_cleanup.py

# 4. Remove one-time update scripts
echo "🗑️  Removing one-time update scripts..."
rm -f update_notebooks.py
rm -f update_split.py
rm -f update_stratified_split.py

# 5. Remove outdated scripts folder files
echo "🗑️  Removing outdated scripts/ files..."
rm -f scripts/automated_annotation.py
rm -f scripts/augment_dataset.py
rm -f scripts/convert_to_label_studio.py
rm -f scripts/export_annotations.py
rm -f scripts/self_consistency_check.py
rm -f scripts/test_annotation_prompts.py
rm -f scripts/inference_ner.py

# 6. Remove old experiment folder
echo "🗑️  Removing old experiment folder..."
rm -rf experiments/approach2_claim_ner/

# 7. Remove temporary/test scripts
echo "🗑️  Removing temp scripts..."
rm -f check_gpt_integrity.py
rm -f clean_invalid_claims.py
rm -f remove_false_positives.py
rm -f apply_manual_review.py

# Count files after
AFTER_COUNT=$(find . -type f -name "*.py" | wc -l)
REMOVED=$((BEFORE_COUNT - AFTER_COUNT))

echo ""
echo "✅ Cleanup complete!"
echo "📊 After cleanup: $AFTER_COUNT Python files"
echo "🗑️  Removed: $REMOVED files"
echo ""
echo "📝 Kept active files:"
echo "  ✅ approach5_pure_ner_improved.ipynb (MAIN MODEL)"
echo "  ✅ inference.py"
echo "  ✅ ood_test_smishtank.py"
echo "  ✅ scripts/evaluate_model_performance.py"
echo "  ✅ validate_dataset_integrity.py"
echo "  ✅ review_claims_gui.py"
echo "  ✅ All data/ and docs/"
echo ""
echo "📖 See CLEANUP_REPORT.md for details"

#!/bin/bash
# Final Deep Cleanup - December 29, 2025
# Remove all unnecessary files, keep only essentials

echo "🧹 FINAL DEEP CLEANUP"
echo "===================="
echo ""

# 1. Remove ALL .sh scripts (except this one)
echo "🗑️  Removing shell scripts..."
rm -f setup_training.sh
rm -f setup_training_novenv.sh
rm -f run_experiment.sh
rm -f setup_annotation.sh
rm -f run_augmentation.sh
rm -f run_augmentation_batch.sh
rm -f view_visualizations.sh
rm -f deploy_to_server.sh
rm -f cleanup_repo.sh
rm -f run_all_experiments.sh
rm -f install_and_train.sh
rm -f setup_evaluation.sh

# 2. Remove ALL documentation .md files (keep only README.md)
echo "🗑️  Removing documentation files..."
rm -f PROJECT_SUMMARY.md
rm -f APPROACH5_IMPROVEMENTS.md
rm -f SETUP_COMPLETE.md
rm -f ANNOTATION_RESULTS.md
rm -f UPDATES_COMPLETE.md
rm -f NOTEBOOKS_UPDATED.md
rm -f CLEANUP_REPORT.md
rm -f BATCH_TRACKING.md
rm -f README_OLD.md
rm -f TRAINING_NOTEBOOKS_README.md
rm -f DATASET_FIXED_SUMMARY.md
rm -f QUICK_REFERENCE.md

# Remove docs folder (all documentation)
rm -rf docs/

# Keep only scripts/README.md for reference
# rm -f scripts/README.md  # Keep this one

# 3. Clean data/annotations - Keep ONLY final datasets
echo "🗑️  Cleaning data/annotations folder..."
cd data/annotations

# Remove ALL archived/backup versions
rm -f archived_*.json
rm -f augmented_phishing_1000.json
rm -f augmented_phishing_1000_metadata.json

# KEEP ONLY:
# - claim_annotations_2000_reviewed.json (FINAL claim dataset)
# - entity_annotations_2000.json (FINAL entity dataset)
# - balanced_dataset_2000.json (FINAL balanced dataset)
# - balanced_dataset_2000_metadata.json
# - BALANCED_DATASET_README.md

cd ../..

# 4. Clean data/processed - Remove old CSVs
echo "🗑️  Cleaning data/processed folder..."
rm -f data/processed/annotation_split_mapping.json
rm -f data/processed/annotation_split_mapping_clean.json
rm -f data/processed/splits_manifest.json
rm -f data/processed/train.csv
rm -f data/processed/test.csv

# KEEP: data/processed/sms_phishing_ham_balanced_2000.csv (final CSV)

# 5. Clean data/eda - Remove report JSONs (keep visualizations)
echo "🗑️  Cleaning data/eda folder..."
rm -f data/eda/*.json
rm -f data/eda/*.png
rm -f data/eda/DATA_INTEGRITY_REPORT.md
rm -f data/eda/VISUALIZATION_SUMMARY.md

# KEEP: data/eda/viz/ folder with visualizations

# 6. Remove experiments/approach3 and approach4 (unused)
echo "🗑️  Removing unused experiment folders..."
rm -rf experiments/approach3_hybrid_llm/
rm -rf experiments/approach4_contrastive/

# KEEP: experiments/archive/ (for reference)

# 7. Remove duplicate model code in src/models (training is in notebooks)
echo "🗑️  Removing duplicate model code..."
rm -rf src/models/
rm -rf src/data/
rm -f src/__init__.py
rm -rf src/utils/
rm -rf src/evaluation/

# Check if src/ is empty, remove it
if [ -z "$(ls -A src/)" ]; then
    rm -rf src/
fi

# 8. Remove config folder (not used)
echo "🗑️  Removing config folder..."
rm -rf config/

# 9. Remove configs folder (YAML files not used for notebooks)
echo "🗑️  Removing configs folder..."
rm -rf configs/

# 10. Remove notebooks_backup (duplicates)
echo "🗑️  Removing notebook backups..."
rm -rf notebooks_backup/

# 11. Remove temporary/test files
echo "🗑️  Removing temporary files..."
rm -f annotation_issues.json
rm -f false_positives.json
rm -f test_setup.py

echo ""
echo "✅ FINAL CLEANUP COMPLETE!"
echo ""
echo "📁 KEPT (Essential files only):"
echo "  ✅ Notebooks:"
echo "     - approach1_entity_first_ner.ipynb"
echo "     - approach2_claim_phrase_ner.ipynb" 
echo "     - approach3_hybrid_claim_llm.ipynb"
echo "     - approach4_contrastive_classification.ipynb"
echo "     - approach5_pure_ner_improved.ipynb (MAIN)"
echo "     - colab_training.ipynb"
echo ""
echo "  ✅ Scripts:"
echo "     - inference.py"
echo "     - ood_test_smishtank.py"
echo "     - ood_test_smishtank_colab.py"
echo "     - scripts/evaluate_model_performance.py"
echo "     - scripts/add_ham_messages.py"
echo "     - scripts/eda_annotations.py"
echo "     - scripts/quality_control_annotations.py"
echo "     - scripts/prepare_data.py"
echo "     - scripts/compare_models.py"
echo "     - scripts/convert_balanced_to_csv.py"
echo ""
echo "  ✅ Review Tools:"
echo "     - review_claims_gui.py"
echo "     - review_identity_claims.py"
echo "     - analyze_claim_dataset_quality.py"
echo "     - validate_dataset_integrity.py"
echo ""
echo "  ✅ Data (Final versions only):"
echo "     - data/annotations/claim_annotations_2000_reviewed.json"
echo "     - data/annotations/entity_annotations_2000.json"
echo "     - data/annotations/balanced_dataset_2000.json"
echo "     - data/processed/sms_phishing_ham_balanced_2000.csv"
echo "     - data/raw/mendeley.csv"
echo "     - data/raw/smishtank.csv"
echo "     - data/eda/viz/ (visualizations)"
echo ""
echo "  ✅ Results:"
echo "     - evaluation_output/"
echo ""
echo "  ✅ Documentation:"
echo "     - README.md"
echo "     - scripts/README.md"
echo "     - data/annotations/BALANCED_DATASET_README.md"
echo ""
echo "🗑️  REMOVED:"
echo "  - All .sh scripts (12 files)"
echo "  - All .md docs (11 files + docs/ folder)"
echo "  - Archived/backup data files (10+ files)"
echo "  - Old processed data (5 files)"
echo "  - Unused experiments (approach3, approach4)"
echo "  - Duplicate model code (src/models/)"
echo "  - Backup notebooks (notebooks_backup/)"
echo "  - Config folders (config/, configs/)"
echo "  - Temporary files"
echo ""
echo "📊 Summary: Reduced repo to essential working files only!"

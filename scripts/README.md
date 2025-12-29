# Scripts Directory

This directory contains all the scripts used for data processing, annotation, and model training.

## Active Scripts

### Data Preparation
- **prepare_data.py** - Loads and preprocesses raw SMS data
- **add_ham_messages.py** - Adds HAM (benign) messages to create balanced dataset
- **convert_balanced_to_csv.py** - Converts balanced JSON dataset to CSV format

### Data Augmentation
- **augment_dataset.py** - Augments phishing messages using GPT-4o-mini API
- **validate_augmented_dataset.py** - Validates quality of augmented messages

### Automated Annotation
- **automated_annotation.py** - Main script for automated annotation using OpenAI API
  - Annotates all 2000 messages with both entity-based and claim-based schemas
  - Uses robust prompts with validation
  - Requires OPENAI_API_KEY environment variable
- **test_annotation_prompts.py** - Tests annotation prompts on sample messages

### Training & Evaluation
- **train_ner.py** - Trains NER models (entity-based or claim-based)
- **inference_ner.py** - Runs inference on new messages
- **compare_models.py** - Compares performance of different model approaches

### Analysis
- **eda_annotations.py** - Exploratory data analysis on annotated datasets
- **export_annotations.py** - Exports annotations in various formats
- **convert_to_label_studio.py** - Converts data to Label Studio format

## Usage

### Setup Environment
```bash
# Copy and configure API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run setup script
./setup_annotation.sh
```

### Run Automated Annotation
```bash
# Annotate all 2000 messages (both schemas)
python3 scripts/automated_annotation.py

# Test prompts first
python3 scripts/test_annotation_prompts.py
```

### Data Processing Pipeline
```bash
# 1. Augment phishing messages
python3 scripts/augment_dataset.py

# 2. Add HAM messages for balance
python3 scripts/add_ham_messages.py

# 3. Convert to CSV (optional)
python3 scripts/convert_balanced_to_csv.py

# 4. Run automated annotation
python3 scripts/automated_annotation.py
```

### Training Models
```bash
# Train entity-based NER
python3 scripts/train_ner.py --config configs/entity_ner.yaml

# Train claim-based NER
python3 scripts/train_ner.py --config configs/claim_ner.yaml

# Compare models
python3 scripts/compare_models.py
```

## Notes

- All scripts use environment variables for API keys (never hardcode keys!)
- Annotation scripts include validation to ensure span correctness
- Progress is saved periodically during long-running operations
- See individual script headers for detailed documentation

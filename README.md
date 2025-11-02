# SMS Phishing Detection via Multi-Agent Claim Verification# SMS Phishing Detection via Multi-Agent Claim Verification



## Project Goal## Overview



Develop a phishing SMS detection system that achieves approximately 99% accuracy on known attacks AND robust performance on zero-day/OOD (out-of-distribution) attacks.This project implements a phishing SMS detection system designed to achieve high accuracy on known attacks while maintaining robust performance on zero-day and out-of-distribution attacks.



Key Innovation: Multi-agent claim verification architecture that extracts and verifies factual claims against external sources, similar to how humans detect phishing.The system employs a multi-agent claim verification architecture that extracts and verifies factual claims against external knowledge sources, rather than relying solely on pattern matching approaches.



## System Architecture## Architecture



``````

SMS Input SMS Input 

  |  |

  v  v

[Claim Extraction Agent]Claim Extraction Agent (extracts entities and claims)

  - Extracts entities (BRAND, URL, PHONE, ORDER_ID, etc.)  |

  - Extracts claim phrases  v

  - Extracts psychological cues (urgency, authority)Claim Parsing Agent (structures into predicate format)

  |  |

  v  v

[Claim Parsing Agent]  Claim Verification Agent (verifies against external sources)

  - Structures claims into predicate format  |

  - Format: [Subject] [Predicate] [Object] [Time]  v

  - Example: [PayPal account] [will be suspended] [null] [tonight]Classification Decision (Supported/Unsupported/Unsure + Confidence)

  |```

  v

[Claim Verification Agent]## Dataset

  - Searches external sources (Google, Bing, company websites)

  - Checks Reddit/Twitter for scam reportsSource: Mendeley SMS Dataset

  - Verifies against user context (Gmail, calendar, purchase history)

  - Validates URLs, phone numbers, brand infoTraining Set: 4,776 messages

  |- Ham: 3,875

  v- Smishing: 510

Decision: Supported / Unsupported / Unsure + Confidence Score- Spam: 391

```

Test Set: 1,195 messages

## Dataset- Ham: 969

- Smishing: 128

Source: Mendeley SMS Dataset- Spam: 98



- Training Set: 4,776 messages (Ham: 3,875, Smishing: 510, Spam: 391)## Phase 1: Data Annotation

- Test Set: 1,195 messages (Ham: 969, Smishing: 128, Spam: 98)

- Annotated: 510 smishing messages with 8 entity types### Environment Setup

- Data split: 67% train / 17% val / 16% test

```bash

## Model Approachespython -m venv venv

source venv/bin/activate

This project implements 4 different approaches to claim extraction:pip install -r requirements.txt

pip install label-studio

### Approach 1: Entity-First NER Pipeline```

Fine-tune BERT-based NER to extract entities, then parse into structured claims.

- Entities: BRAND, PHONE, URL, ORDER_ID, AMOUNT, DATE, DEADLINE, ACTION_REQUIRED### Data Preparation

- Pros: Well-defined entities, can reuse pretrained models, clear intermediate representation

```bash

### Approach 2: Claim-Phrase NER Pipelinepython scripts/prepare_data.py

Train NER to identify claim phrase boundaries directly.python scripts/convert_to_label_studio.py

- Pros: Captures semantic claim units, may be more robust to variations```



### Approach 3: Hybrid NER + LLM### Annotation Interface

Combine fine-tuned NER for entity extraction with LLM for claim structuring.

- NER: Fast, cheap, accurate entity extraction```bash

- LLM: Handles complex reasoning (Flan-T5/GPT-3.5/Claude)label-studio start

- Pros: Best of both worlds, cost-effective```



### Approach 4: Contrastive LearningAccess the interface at http://localhost:8080 and configure using `config/label_studio_config.xml`.

Train encoder to learn claim embeddings using supervised contrastive loss.

- Designed for OOD robustness and zero-day attacks### Project Structure

- Pros: Learns semantic representations, not templates```

sms-claim-check/

## Installation├── data/

│   ├── raw/

```bash│   ├── processed/

python3 -m venv venv│   └── annotations/

source venv/bin/activate├── config/

pip install -r requirements_new.txt│   ├── entity_schema.py

```│   └── label_studio_config.xml

├── scripts/

For LLM-based approaches, set API keys:│   ├── prepare_data.py

```bash│   ├── convert_to_label_studio.py

export OPENAI_API_KEY="your-key-here"│   ├── quality_check.py

export ANTHROPIC_API_KEY="your-key-here"│   └── export_annotations.py

```├── docs/

│   └── annotation_guidelines.md

## Quick Start├── notebooks/

└── requirements.txt

### Test Installation```

```bash

python test_setup.py## Annotation Targets

```

Primary dataset: 510 smishing messages

### Train a ModelNegative examples: 100 ham messages

```bashAdditional: 50 spam messages

./run_experiment.sh entity_ner

./run_experiment.sh claim_nerData split ratios:

./run_experiment.sh hybrid_llm- Training: 67%

./run_experiment.sh contrastive- Validation: 17%

```- Test: 16%



### Train All Models## Entity Schema

```bash

./run_all_experiments.shThe annotation schema consists of eight entity types:

```

1. BRAND: Company or organization names

### Custom Configuration2. PHONE: Phone numbers in any format

```bash3. URL: Web links, including shortened URLs

python train.py --config configs/my_custom_config.yaml4. ORDER_ID: Order, tracking, or invoice identifiers

```5. AMOUNT: Monetary amounts with or without currency symbols

6. DATE: Temporal references

### Run Inference7. DEADLINE: Time pressure indicators

```bash8. ACTION_REQUIRED: Imperative action verbs

python inference.py --model experiments/entity_ner_roberta/final_model --interactive

Detailed annotation guidelines are provided in `docs/annotation_guidelines.md`.

python inference.py --model experiments/entity_ner_roberta/final_model \

  --text "Your Amazon order requires verification"## Subsequent Phases



python inference.py --model experiments/entity_ner_roberta/final_model --file messages.txtPhase 2: NER model training using multiple approaches

```Phase 3: Claim parsing implementation

Phase 4: Verification agent development

### Compare ModelsPhase 5: Baseline comparison and evaluation

```bash
python scripts/compare_models.py
```

## Project Structure

```
sms-claim-check/
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── base.py
│   │   ├── entity_ner.py
│   │   ├── claim_ner.py
│   │   ├── hybrid_llm.py
│   │   └── contrastive.py
│   └── utils/
├── configs/
│   ├── entity_ner.yaml
│   ├── claim_ner.yaml
│   ├── hybrid_llm.yaml
│   └── contrastive.yaml
├── data/
│   ├── annotations/
│   ├── processed/
│   └── raw/
├── experiments/
├── scripts/
├── train.py
├── inference.py
├── test_setup.py
├── run_experiment.sh
└── run_all_experiments.sh
```

## Configuration

All models are configured via YAML files in `configs/`:

```yaml
approach: "entity_ner"
name: "Entity-First NER (RoBERTa)"

model_config:
  model_name: "roberta-base"
  max_length: 128

training_config:
  num_epochs: 10
  batch_size: 16
  learning_rate: 2e-5

data_config:
  annotations_file: "data/annotations/annotated_complete.json"
  train_ratio: 0.67
  val_ratio: 0.17
  test_ratio: 0.16

output_config:
  output_dir: "experiments/entity_ner_roberta"
```

## Entity Schema

Eight entity types for annotation:

1. BRAND: Company or organization names
2. PHONE: Phone numbers in any format
3. URL: Web links, including shortened URLs
4. ORDER_ID: Order, tracking, or invoice identifiers
5. AMOUNT: Monetary amounts with or without currency symbols
6. DATE: Temporal references
7. DEADLINE: Time pressure indicators
8. ACTION_REQUIRED: Imperative action verbs

## Label Studio Setup

### Create Project
1. Start Label Studio: `label-studio start`
2. Access at http://localhost:8080
3. Create project: "SMS Phishing Annotation"

### Import Data
1. Settings > Import
2. Upload `data/annotations/preannotated.json`
3. Check "Treat CSV/TSV as List of tasks"

### Configure Interface
Settings > Labeling Interface > Code:

```xml
<View>
  <Header value="Annotate SMS Entities"/>
  <Text name="text" value="$text"/>
  
  <Labels name="label" toName="text">
    <Label value="BRAND" background="#FF6B6B" hotkey="b"/>
    <Label value="PHONE" background="#4ECDC4" hotkey="p"/>
    <Label value="URL" background="#45B7D1" hotkey="u"/>
    <Label value="ORDER_ID" background="#96CEB4" hotkey="o"/>
    <Label value="AMOUNT" background="#FFEAA7" hotkey="m"/>
    <Label value="DATE" background="#DFE6E9" hotkey="d"/>
    <Label value="DEADLINE" background="#FF7675" hotkey="l"/>
    <Label value="ACTION_REQUIRED" background="#FD79A8" hotkey="a"/>
  </Labels>
</View>
```

### Keyboard Shortcuts
- b: BRAND, p: PHONE, u: URL, o: ORDER_ID
- m: AMOUNT, d: DATE, l: DEADLINE, a: ACTION_REQUIRED
- Ctrl+Enter: Submit, Ctrl+Space: Show predictions

### Export
Settings > Export > JSON format

## Adding New Models

1. Create model class in `src/models/new_model.py` inheriting from `BaseModel`
2. Implement required methods: `train()`, `evaluate()`, `predict()`, `save()`, `load()`
3. Create config file in `configs/new_model.yaml`
4. Import in `train.py`

## Cluster Training

For SLURM clusters:

```bash
#!/bin/bash
#SBATCH --job-name=sms-phishing
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G

source venv/bin/activate
python train.py --config configs/entity_ner.yaml
```

## Model Outputs

Each experiment creates:
```
experiments/<experiment_name>/
├── final_model/
├── results.json
├── data_splits.json
└── training_*.log
```

## Migration from Old Code

### Old Training
```bash
python scripts/train_ner.py --model distilbert-base-uncased --epochs 10
```

### New Training
```bash
./run_experiment.sh entity_ner
python train.py --config configs/entity_ner.yaml
```

### Old Inference
```bash
python scripts/inference_ner.py --model models/ner/final_model --text "message"
```

### New Inference
```bash
python inference.py --model experiments/entity_ner_roberta/final_model --text "message"
```

## Performance Expectations

| Model | Training Time | F1 Score (Est.) | Use Case |
|-------|--------------|-----------------|----------|
| Entity NER (DistilBERT) | 10-15 min | ~0.88 | Fast baseline |
| Entity NER (RoBERTa) | 20-30 min | ~0.92 | Production |
| Claim NER | 20-30 min | ~0.88 | Alternative |
| Hybrid NER + LLM | 25-35 min | ~0.94 | Best performance |
| Contrastive | 40-60 min | N/A | OOD robustness |

Times on GPU (V100/A100). CPU is 5-10x slower.

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Use smaller model: `distilbert-base-uncased`

### Slow Training
- Reduce `num_epochs`
- Use smaller model

### Import Errors
```bash
pip install -r requirements_new.txt
```

## Next Steps

1. Train baseline models
2. Implement claim parsing agent
3. Build verification agent with external APIs
4. Connect all agents for end-to-end pipeline
5. Evaluate on zero-day attacks
6. Compare against traditional phishing detectors

## License

MIT License

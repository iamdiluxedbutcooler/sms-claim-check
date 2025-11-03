# SMS Phishing Detection via Multi-Agent Claim Verification: Claim Detection Agent



## Project Goal



Develop a phishing SMS detection system that achieves approximately 99% accuracy on known attacks AND robust performance on zero-day/OOD (out-of-distribution) attacks.This project implements a phishing SMS detection system designed to achieve high accuracy on known attacks while maintaining robust performance on zero-day and out-of-distribution attacks.



Key Innovation: Multi-agent claim verification architecture that extracts and verifies factual claims against external sources, similar to how humans detect phishing.The system employs a multi-agent claim verification architecture that extracts and verifies factual claims against external knowledge sources, rather than relying solely on pattern matching approaches.

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

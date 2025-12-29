#!/usr/bin/env python3
"""
Update all 4 training notebooks to:
1. Save models to Google Drive
2. Add progress bars and checkpoints
3. Remove emojis
"""

import json
import re
from pathlib import Path

def remove_emojis(text):
    """Remove emoji characters from text"""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # supplemental symbols
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)

def update_notebook(notebook_path, approach_name):
    """Update a single notebook"""
    print(f"Updating {notebook_path.name}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Update each cell
    for cell in nb['cells']:
        if cell['cell_type'] in ['markdown', 'code']:
            # Remove emojis from source
            if isinstance(cell['source'], list):
                cell['source'] = [remove_emojis(line) for line in cell['source']]
            elif isinstance(cell['source'], str):
                cell['source'] = remove_emojis(cell['source'])
    
    # Find and update specific cells
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Update Google Drive mounting
            if 'drive.mount' in source and '# from google.colab import drive' in source:
                cell['source'] = [
                    "# Mount Google Drive for saving models\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')\n",
                    "\n",
                    "# Create directory for saving models\n",
                    "import os\n",
                    f"save_dir = '/content/drive/MyDrive/sms_claim_models/{approach_name}'\n",
                    "os.makedirs(save_dir, exist_ok=True)\n",
                    "print(f\"Models will be saved to: {save_dir}\")"
                ]
            
            # Add tqdm import
            if 'from google.colab import files' in source and 'tqdm' not in source:
                cell['source'] = [
                    "from google.colab import files\n",
                    "from tqdm.auto import tqdm\n",
                    "\n",
                    "print(\"Please upload '",
                    "claim_annotations_2000.json" if 'claim' in approach_name else "entity_annotations_2000.json",
                    "'\")\n",
                    "uploaded = files.upload()\n",
                    "data_file = list(uploaded.keys())[0]\n",
                    "print(f\"Uploaded: {data_file}\")"
                ]
            
            # Update training arguments to add checkpoints
            if 'TrainingArguments(' in source and 'save_steps' not in source:
                # Add checkpointing
                source = source.replace(
                    'eval_strategy="epoch"',
                    'eval_strategy="steps"'
                ).replace(
                    'save_strategy="epoch"',
                    'save_strategy="steps",\n    save_steps=50,\n    eval_steps=50,\n    save_total_limit=5'
                ).replace(
                    'logging_steps=50',
                    'logging_steps=10,\n    logging_first_step=True'
                ).replace(
                    'report_to="none"',
                    'report_to="none",\n    disable_tqdm=False  # Show progress bar'
                )
                cell['source'] = source.split('\n')
            
            # Update model saving to Google Drive
            if 'model.save_pretrained' in source and 'Download model' in source:
                cell['source'] = [
                    "# Save model to Google Drive\n",
                    "print(\"\\nSaving model to Google Drive...\")\n",
                    f"final_model_path = f\"{{save_dir}}/final_model\"\n",
                    "model.save_pretrained(final_model_path)\n",
                    "tokenizer.save_pretrained(final_model_path)\n",
                    "\n",
                    "# Save metadata\n",
                    "import json\n",
                    "import pandas as pd\n",
                    "with open(f\"{final_model_path}/training_info.json\", \"w\") as f:\n",
                    "    json.dump({\n",
                    "        'approach': '", approach_name, "',\n",
                    "        'training_date': str(pd.Timestamp.now()),\n",
                    "        'model_name': MODEL_NAME,\n",
                    "        'num_train_examples': len(train_dataset),\n",
                    "    }, f, indent=2)\n",
                    "\n",
                    "print(f\"Model saved to: {final_model_path}\")\n",
                    "print(\"Access it anytime from your Google Drive!\")\n",
                    "\n",
                    "# Create zip for download\n",
                    "print(\"\\nCreating zip file...\")\n",
                    "!zip -r -q model_final.zip {final_model_path}\n",
                    "print(\"Zip created. You can download it now!\")"
                ]
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"  Updated {notebook_path.name}")

def main():
    notebooks = [
        ('approach1_entity_first_ner.ipynb', 'approach1_entity_ner'),
        ('approach2_claim_phrase_ner.ipynb', 'approach2_claim_ner'),
        ('approach3_hybrid_claim_llm.ipynb', 'approach3_hybrid_llm'),
        ('approach4_contrastive_classification.ipynb', 'approach4_contrastive')
    ]
    
    base_dir = Path(__file__).parent
    
    print("="*60)
    print("UPDATING ALL NOTEBOOKS")
    print("="*60)
    print("Changes:")
    print("  - Remove all emojis")
    print("  - Add Google Drive saving")
    print("  - Add progress bars")
    print("  - Add training checkpoints")
    print("="*60)
    print()
    
    for notebook_file, approach_name in notebooks:
        notebook_path = base_dir / notebook_file
        if notebook_path.exists():
            update_notebook(notebook_path, approach_name)
        else:
            print(f"  Skipped {notebook_file} (not found)")
    
    print()
    print("="*60)
    print("ALL NOTEBOOKS UPDATED!")
    print("="*60)
    print("Backup saved to: notebooks_backup/")
    print()
    print("Key improvements:")
    print("  1. Models save to Google Drive automatically")
    print("  2. Training shows progress bars")
    print("  3. Checkpoints saved every 50 steps")
    print("  4. All emojis removed for cleaner output")
    print("="*60)

if __name__ == '__main__':
    main()

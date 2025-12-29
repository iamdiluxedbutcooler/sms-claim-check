import json

with open('approach2_claim_phrase_ner.ipynb', 'r') as f:
    nb = json.load(f)

# Find and fix the compressed cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if '# Training argumentstraining_args' in source:
            # Replace with properly formatted code
            cell['source'] = [
                "# Training arguments\n",
                "training_args = TrainingArguments(\n",
                "    output_dir=\"./claim-ner-model\",\n",
                "    eval_strategy=\"steps\",\n",
                "    save_strategy=\"steps\",\n",
                "    save_steps=50,\n",
                "    eval_steps=50,\n",
                "    save_total_limit=5,\n",
                "    learning_rate=2e-5,\n",
                "    per_device_train_batch_size=16,\n",
                "    per_device_eval_batch_size=16,\n",
                "    num_train_epochs=5,\n",
                "    weight_decay=0.01,\n",
                "    warmup_ratio=0.1,\n",
                "    logging_steps=10,\n",
                "    logging_first_step=True,\n",
                "    load_best_model_at_end=True,\n",
                "    metric_for_best_model=\"f1\",\n",
                "    push_to_hub=False,\n",
                "    report_to=\"none\",\n",
                "    disable_tqdm=False  # Show progress bar\n",
                ")\n",
                "\n",
                "# Data collator\n",
                "data_collator = DataCollatorForTokenClassification(tokenizer)\n",
                "\n",
                "# Create trainer\n",
                "trainer = Trainer(\n",
                "    model=model,\n",
                "    args=training_args,\n",
                "    train_dataset=train_dataset,\n",
                "    eval_dataset=test_dataset,  # Using test set for evaluation\n",
                "    tokenizer=tokenizer,\n",
                "    data_collator=data_collator,\n",
                "    compute_metrics=compute_metrics\n",
                ")\n",
                "\n",
                "print(\"Trainer initialized\")\n",
                "print(f\"Total training steps: {len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs}\")"
            ]
            print("Fixed training arguments cell")
            break

with open('approach2_claim_phrase_ner.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook fixed!")

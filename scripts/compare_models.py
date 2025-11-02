import json
from pathlib import Path
from typing import Dict, List
from tabulate import tabulate


def load_experiment_results(experiment_dir: Path) -> Dict:
    results_file = experiment_dir / "results.json"
    
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


def extract_metrics(results: Dict) -> Dict:
    if not results:
        return {}
    
    test_metrics = results.get('test_metrics', {})
    
    metrics = {
        'name': results.get('config', {}).get('name', 'Unknown'),
        'approach': results.get('config', {}).get('approach', 'Unknown'),
    }
    
    if 'eval_f1' in test_metrics:
        metrics.update({
            'f1': test_metrics.get('eval_f1', 0),
            'precision': test_metrics.get('eval_precision', 0),
            'recall': test_metrics.get('eval_recall', 0),
            'accuracy': test_metrics.get('eval_accuracy', 0),
        })
    
    if 'test_loss' in test_metrics:
        metrics.update({
            'loss': test_metrics.get('test_loss', 0),
        })
    
    return metrics


def main():
    project_root = Path(__file__).parent.parent
    experiments_dir = project_root / "experiments"
    
    if not experiments_dir.exists():
        print("No experiments directory found!")
        return
    
    experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
    
    if not experiment_dirs:
        print("No experiments found!")
        return
    
    all_results = []
    for exp_dir in experiment_dirs:
        results = load_experiment_results(exp_dir)
        if results:
            metrics = extract_metrics(results)
            metrics['experiment_dir'] = exp_dir.name
            all_results.append(metrics)
    
    if not all_results:
        print("No completed experiments with results found!")
        return
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80 + "\n")
    
    ner_results = [r for r in all_results if 'f1' in r]
    contrastive_results = [r for r in all_results if 'loss' in r]
    
    if ner_results:
        print("NER-Based Approaches (Entity & Claim Extraction)")
        print("-" * 80)
        
        table_data = []
        for r in sorted(ner_results, key=lambda x: x.get('f1', 0), reverse=True):
            table_data.append([
                r.get('name', 'Unknown'),
                r.get('approach', 'Unknown'),
                f"{r.get('f1', 0):.4f}",
                f"{r.get('precision', 0):.4f}",
                f"{r.get('recall', 0):.4f}",
                f"{r.get('accuracy', 0):.4f}",
            ])
        
        headers = ['Model', 'Approach', 'F1', 'Precision', 'Recall', 'Accuracy']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()
    
    if contrastive_results:
        print("Contrastive Learning Approaches")
        print("-" * 80)
        
        table_data = []
        for r in sorted(contrastive_results, key=lambda x: x.get('loss', float('inf'))):
            table_data.append([
                r.get('name', 'Unknown'),
                r.get('approach', 'Unknown'),
                f"{r.get('loss', 0):.4f}",
            ])
        
        headers = ['Model', 'Approach', 'Test Loss']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()
    
    print("Summary")
    print("-" * 80)
    print(f"Total experiments: {len(all_results)}")
    print(f"NER-based models: {len(ner_results)}")
    print(f"Contrastive models: {len(contrastive_results)}")
    
    if ner_results:
        best_model = max(ner_results, key=lambda x: x.get('f1', 0))
        print(f"\nBest NER model: {best_model['name']}")
        print(f"  F1: {best_model.get('f1', 0):.4f}")
        print(f"  Precision: {best_model.get('precision', 0):.4f}")
        print(f"  Recall: {best_model.get('recall', 0):.4f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

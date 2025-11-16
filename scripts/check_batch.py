import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import HybridNERLLMModel


def main():
    parser = argparse.ArgumentParser(description="Check OpenAI Batch API status and retrieve results")
    parser.add_argument('--batch-id', type=str, required=True, help='Batch job ID')
    parser.add_argument('--config', type=str, default='configs/hybrid_llm.yaml', help='Config file')
    parser.add_argument('--retrieve', action='store_true', help='Retrieve results if completed')
    parser.add_argument('--output', type=str, help='Output file for results (optional)')
    
    args = parser.parse_args()
    
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model = HybridNERLLMModel(config['model_config'])
    
    print(f"\nChecking batch: {args.batch_id}")
    print("="*70)
    
    status = model.check_batch_status(args.batch_id)
    
    print(f"\nStatus: {status['status']}")
    print(f"Total requests: {status['request_counts']['total']}")
    print(f"Completed: {status['request_counts']['completed']}")
    print(f"Failed: {status['request_counts']['failed']}")
    
    if status['status'] == 'completed' and args.retrieve:
        print("\nRetrieving results...")
        
        output_file = args.output or f"batch_results_{args.batch_id}.jsonl"
        results = model.retrieve_batch_results(args.batch_id, output_file)
        
        print(f"\nRetrieved {len(results)} results")
        
        structured = model.parse_batch_results(results)
        
        structured_file = Path(output_file).with_suffix('.json')
        with open(structured_file, 'w') as f:
            json.dump(structured, f, indent=2)
        
        print(f"Structured results saved to: {structured_file}")
        
        total_claims = sum(len(claims) for claims in structured.values())
        print(f"\nTotal claims extracted: {total_claims}")
    
    elif status['status'] in ['validating', 'in_progress', 'finalizing']:
        print("\nBatch is still processing. Check again later.")
    elif status['status'] == 'failed':
        print("\nBatch failed!")
    elif status['status'] == 'expired':
        print("\nBatch expired!")


if __name__ == "__main__":
    main()

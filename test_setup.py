import sys
from pathlib import Path

def check_dependencies():
    print("Checking dependencies...")
    missing = []
    
    required = [
        'torch',
        'transformers',
        'datasets',
        'evaluate',
        'sklearn',
        'numpy',
        'pandas',
        'yaml',
    ]
    
    for package in required:
        try:
            __import__(package)
            print(f"  OK {package}")
        except ImportError:
            print(f"  MISSING {package}")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nTo install:")
        print("  pip install -r requirements_new.txt")
        return False
    
    print("  All dependencies installed")
    return True


sys.path.insert(0, str(Path(__file__).parent))

def import_modules():
    global AnnotationLoader, EntityNERPreprocessor, ClaimNERPreprocessor, ContrastivePreprocessor
    global EntityNERModel, ClaimNERModel, HybridNERLLMModel, ContrastiveModel
    
    from src.data import AnnotationLoader, EntityNERPreprocessor, ClaimNERPreprocessor, ContrastivePreprocessor
    from src.models import EntityNERModel, ClaimNERModel, HybridNERLLMModel, ContrastiveModel

import sys
from pathlib import Path

def check_dependencies():
    global AnnotationLoader, EntityNERPreprocessor, ClaimNERPreprocessor, ContrastivePreprocessor
    global EntityNERModel, ClaimNERModel, HybridNERLLMModel, ContrastiveModel
    
    from src.data import AnnotationLoader, EntityNERPreprocessor, ClaimNERPreprocessor, ContrastivePreprocessor
    from src.models import EntityNERModel, ClaimNERModel, HybridNERLLMModel, ContrastiveModel


def test_data_loading():
    print("Testing data loading...")
    
    project_root = Path(__file__).parent
    annotations_file = project_root / "data" / "annotations" / "annotated_complete.json"
    
    if not annotations_file.exists():
        print(f"  WARNING: Annotations file not found: {annotations_file}")
        print("  This is expected if you haven't annotated data yet")
        return False
    
    loader = AnnotationLoader(annotations_file)
    messages = loader.load()
    
    print(f"  PASS: Loaded {len(messages)} messages")
    
    train, val, test = loader.split_data(messages)
    print(f"  PASS: Split into train={len(train)}, val={len(val)}, test={len(test)}")
    
    return True


def test_preprocessors():
    print("\nTesting preprocessors...")
    
    from src.data.loader import Message, Entity
    
    msg = Message(
        id="test1",
        text="Amazon order #12345 requires verification. Call 1-800-FAKE",
        entities=[
            Entity(0, 6, "BRAND", "Amazon"),
            Entity(7, 19, "ORDER_ID", "order #12345"),
            Entity(50, 60, "PHONE", "1-800-FAKE"),
        ]
    )
    
    entity_labels = ["O", "B-BRAND", "I-BRAND", "B-PHONE", "I-PHONE", "B-ORDER_ID", "I-ORDER_ID"]
    preprocessor = EntityNERPreprocessor(entity_labels)
    examples = preprocessor.prepare_examples([msg])
    print(f"  PASS: EntityNERPreprocessor: {len(examples)} examples")
    
    preprocessor = ClaimNERPreprocessor()
    examples = preprocessor.prepare_examples([msg])
    print(f"  PASS: ClaimNERPreprocessor: {len(examples)} examples")
    
    preprocessor = ContrastivePreprocessor()
    examples = preprocessor.prepare_examples([msg])
    print(f"  PASS: ContrastivePreprocessor: {len(examples)} examples")
    
    return True


def test_model_initialization():
    print("\nTesting model initialization...")
    
    config = {
        'model_name': 'distilbert-base-uncased',
        'max_length': 128,
        'num_epochs': 1,
        'batch_size': 8,
    }
    
    model = EntityNERModel(config)
    print(f"  PASS: EntityNERModel initialized")
    
    model = ClaimNERModel(config)
    print(f"  PASS: ClaimNERModel initialized")
    
    config_hybrid = config.copy()
    config_hybrid['ner_config'] = config.copy()
    config_hybrid['use_local_llm'] = True
    config_hybrid['llm_provider'] = 'flan-t5'
    model = HybridNERLLMModel(config_hybrid)
    print(f"  PASS: HybridNERLLMModel initialized")
    
    config_contrastive = config.copy()
    config_contrastive['embedding_dim'] = 256
    model = ContrastiveModel(config_contrastive)
    print(f"  PASS: ContrastiveModel initialized")
    
    return True


def test_preprocessors():
    print("\nTesting model initialization...")
    
    config = {
        'model_name': 'distilbert-base-uncased',
        'max_length': 128,
        'num_epochs': 1,
        'batch_size': 8,
    }
    
    model = EntityNERModel(config)
    print(f"   EntityNERModel initialized")
    
    model = ClaimNERModel(config)
    print(f"   ClaimNERModel initialized")
    
    config_hybrid = config.copy()
    config_hybrid['ner_config'] = config.copy()
    config_hybrid['use_local_llm'] = True
    config_hybrid['llm_provider'] = 'flan-t5'
    model = HybridNERLLMModel(config_hybrid)
    print(f"   HybridNERLLMModel initialized")
    
    config_contrastive = config.copy()
    config_contrastive['embedding_dim'] = 256
    model = ContrastiveModel(config_contrastive)
    print(f"   ContrastiveModel initialized")
    
    return True


def main():
    print("="*70)
    print("Testing New Architecture")
    print("="*70)
    print()
    
    if not check_dependencies():
        print("\n" + "="*70)
        print("Cannot proceed without dependencies")
        print("="*70)
        return
    
    print()
    
    try:
        import_modules()
        
        data_ok = test_data_loading()
        prep_ok = test_preprocessors()
        model_ok = test_model_initialization()
        
        print("\n" + "="*70)
        print("Test Summary")
        print("="*70)
        print(f"Dependencies:     PASS")
        print(f"Data Loading:     {'PASS' if data_ok else 'SKIP (no annotations)'}")
        print(f"Preprocessors:    {'PASS' if prep_ok else 'FAIL'}")
        print(f"Models:           {'PASS' if model_ok else 'FAIL'}")
        
        if prep_ok and model_ok:
            print("\nAll critical tests passed")
            print("\nNext steps:")
            print("  1. Run: ./run_experiment.sh entity_ner")
            print("  2. Or: python train.py --config configs/entity_ner.yaml")
        else:
            print("\nSome tests failed. Check dependencies.")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

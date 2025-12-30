from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
CLAIM_ANNOTATIONS_FILE = ANNOTATIONS_DIR / "claim_annotations_2000_reviewed.json"

# Parsed claim data paths
CLAIM_PARSING_ALL = ANNOTATIONS_DIR / "claim_parsing_all_2000.json"
CLAIM_PARSING_TRAIN = ANNOTATIONS_DIR / "claim_parsing_train_full.json"
CLAIM_PARSING_TEST = ANNOTATIONS_DIR / "claim_parsing_test_full.json"
SPLIT_INFO = ANNOTATIONS_DIR / "split_info.json"


@dataclass
class ParsingConfig:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4o"
    openai_max_tokens: int = 2000
    openai_temperature: float = 0.0
    
    target_num_messages: int = 300
    min_per_rare_type: dict[str, int] = None
    
    t5_model_name: str = "t5-base"
    t5_learning_rate: float = 5e-5
    t5_batch_size: int = 8
    t5_epochs: int = 3
    t5_max_input_length: int = 256
    t5_max_output_length: int = 256
    t5_output_dir: Path = PROJECT_ROOT / "models" / "claim_parsing_t5"
    
    def __post_init__(self):
        if self.min_per_rare_type is None:
            self.min_per_rare_type = {
                "DELIVERY_CLAIM": 10,
                "VERIFICATION_CLAIM": 10,
                "SOCIAL_CLAIM": 5,
                "IDENTITY_CLAIM": 10,
                "LEGAL_CLAIM": 5,
                "SECURITY_CLAIM": 10,
                "CREDENTIALS_CLAIM": 5,
            }
        
        self.t5_output_dir.mkdir(parents=True, exist_ok=True)


from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path


class BaseModel(ABC):
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def train(
        self, 
        train_examples: List[Dict],
        val_examples: List[Dict],
        output_dir: Path
    ) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def evaluate(
        self, 
        test_examples: List[Dict]
    ) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def save(self, output_dir: Path):
        pass
    
    @abstractmethod
    def load(self, model_dir: Path):
        pass
    
    def get_name(self) -> str:
        return self.__class__.__name__

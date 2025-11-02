import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    start: int
    end: int
    label: str
    text: str


@dataclass
class Message:
    id: str
    text: str
    entities: List[Entity]
    metadata: Optional[Dict[str, Any]] = None

    @property
    def is_phishing(self) -> bool:
        return len(self.entities) > 0


class AnnotationLoader:
    
    def __init__(self, annotations_file: Path):
        self.annotations_file = Path(annotations_file)
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    def load(self) -> List[Message]:
        logger.info(f"Loading annotations from {self.annotations_file}")
        
        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = []
        for item in data:
            msg = self._parse_annotation(item)
            if msg:
                messages.append(msg)
        
        logger.info(f"Loaded {len(messages)} annotated messages")
        self._log_statistics(messages)
        
        return messages
    
    def _parse_annotation(self, item: Dict) -> Optional[Message]:
        try:
            msg_id = str(item['data'].get('message_id', item.get('id', 'unknown')))
            text = item['data']['text']
            
            entities = []
            for annotation in item.get('annotations', []):
                for result in annotation.get('result', []):
                    if result.get('type') == 'labels':
                        value = result['value']
                        entity = Entity(
                            start=value['start'],
                            end=value['end'],
                            label=value['labels'][0],
                            text=value.get('text', text[value['start']:value['end']])
                        )
                        entities.append(entity)
            
            entities = sorted(entities, key=lambda x: x.start)
            
            metadata = {
                'original_label': item['data'].get('label'),
                'source': item['data'].get('source'),
            }
            
            return Message(
                id=msg_id,
                text=text,
                entities=entities,
                metadata=metadata
            )
        
        except Exception as e:
            logger.warning(f"Failed to parse annotation item: {e}")
            return None
    
    def _log_statistics(self, messages: List[Message]):
        entity_counts = Counter()
        phishing_count = 0
        
        for msg in messages:
            if msg.is_phishing:
                phishing_count += 1
            for ent in msg.entities:
                entity_counts[ent.label] += 1
        
        logger.info(f"Dataset statistics:")
        logger.info(f"  Total messages: {len(messages)}")
        logger.info(f"  Phishing messages: {phishing_count}")
        logger.info(f"  Ham messages: {len(messages) - phishing_count}")
        logger.info(f"  Total entities: {sum(entity_counts.values())}")
        logger.info(f"\nEntity distribution:")
        for label, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {label:20s}: {count:4d}")
    
    def split_data(
        self, 
        messages: List[Message],
        train_ratio: float = 0.67,
        val_ratio: float = 0.17,
        test_ratio: float = 0.16,
        seed: int = 42
    ) -> Tuple[List[Message], List[Message], List[Message]]:
        from sklearn.model_selection import train_test_split
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        train_msgs, temp_msgs = train_test_split(
            messages,
            test_size=(1 - train_ratio),
            random_state=seed,
            shuffle=True
        )
        
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_msgs, test_msgs = train_test_split(
            temp_msgs,
            test_size=relative_test_ratio,
            random_state=seed,
            shuffle=True
        )
        
        logger.info(f"Data split:")
        logger.info(f"  Train: {len(train_msgs)} messages ({train_ratio*100:.1f}%)")
        logger.info(f"  Val:   {len(val_msgs)} messages ({val_ratio*100:.1f}%)")
        logger.info(f"  Test:  {len(test_msgs)} messages ({test_ratio*100:.1f}%)")
        
        return train_msgs, val_msgs, test_msgs

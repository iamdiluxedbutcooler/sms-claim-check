from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from .loader import Message, Entity

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    
    @abstractmethod
    def prepare_examples(self, messages: List[Message]) -> List[Dict[str, Any]]:
        pass


class EntityNERPreprocessor(BasePreprocessor):
    
    def __init__(self, entity_labels: List[str]):
        self.entity_labels = entity_labels
        self.label2id = {label: idx for idx, label in enumerate(entity_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
    
    def prepare_examples(self, messages: List[Message]) -> List[Dict[str, Any]]:
        examples = []
        
        for msg in messages:
            tokens, labels = self._tokenize_and_label(msg.text, msg.entities)
            
            examples.append({
                'id': msg.id,
                'text': msg.text,
                'tokens': tokens,
                'ner_tags': [self.label2id.get(label, 0) for label in labels],
                'ner_tags_str': labels
            })
        
        return examples
    
    def _tokenize_and_label(
        self, 
        text: str, 
        entities: List[Entity]
    ) -> Tuple[List[str], List[str]]:
        tokens = text.split()
        labels = []
        
        char_pos = 0
        for token in tokens:
            token_start = text.find(token, char_pos)
            token_end = token_start + len(token)
            
            label = self._get_token_label(token_start, token_end, entities)
            labels.append(label)
            
            char_pos = token_end
        
        return tokens, labels
    
    def _get_token_label(
        self, 
        token_start: int, 
        token_end: int, 
        entities: List[Entity]
    ) -> str:
        
        for entity in entities:
            if token_start < entity.end and token_end > entity.start:
                if token_start <= entity.start < token_end:
                    return f"B-{entity.label}"
                elif token_start >= entity.start:
                    return f"I-{entity.label}"
        
        return "O"


class ClaimNERPreprocessor(BasePreprocessor):
    
    def __init__(self):
        self.claim_labels = ["O", "B-CLAIM", "I-CLAIM"]
        self.label2id = {label: idx for idx, label in enumerate(self.claim_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
    
    def prepare_examples(self, messages: List[Message]) -> List[Dict[str, Any]]:
        examples = []
        
        for msg in messages:
            tokens, labels = self._extract_claim_phrases(msg.text, msg.entities)
            
            examples.append({
                'id': msg.id,
                'text': msg.text,
                'tokens': tokens,
                'claim_tags': [self.label2id.get(label, 0) for label in labels],
                'claim_tags_str': labels
            })
        
        return examples
    
    def _extract_claim_phrases(
        self, 
        text: str, 
        entities: List[Entity]
    ) -> Tuple[List[str], List[str]]:
        tokens = text.split()
        labels = ["O"] * len(tokens)
        
        char_to_token = self._build_char_to_token_map(text, tokens)
        
        claim_spans = self._identify_claim_spans(entities, char_to_token, len(tokens))
        
        for start_token, end_token in claim_spans:
            labels[start_token] = "B-CLAIM"
            for i in range(start_token + 1, end_token):
                labels[i] = "I-CLAIM"
        
        return tokens, labels
    
    def _build_char_to_token_map(self, text: str, tokens: List[str]) -> Dict[int, int]:
        char_to_token = {}
        char_idx = 0
        
        for token_idx, token in enumerate(tokens):
            clean_token = token.replace('##', '')
            token_start = text.find(clean_token, char_idx)
            
            if token_start != -1:
                for i in range(token_start, token_start + len(clean_token)):
                    char_to_token[i] = token_idx
                char_idx = token_start + len(clean_token)
        
        return char_to_token
    
    def _identify_claim_spans(self, entities: List[Entity], char_to_token: Dict[int, int], num_tokens: int) -> List[Tuple[int, int]]:
        spans = []
        
        for entity in entities:
            if entity.label in ['ACTION_REQUIRED', 'DEADLINE']:
                start_token = char_to_token.get(entity.start_char, 0)
                end_token = char_to_token.get(entity.end_char - 1, num_tokens - 1) + 1
                
                spans.append((max(0, start_token - 3), min(num_tokens, end_token + 3)))
        
        return self._merge_overlapping_spans(spans)
    
    def _merge_overlapping_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not spans:
            return []
        
        sorted_spans = sorted(spans)
        merged = [sorted_spans[0]]
        
        for current_start, current_end in sorted_spans[1:]:
            last_start, last_end = merged[-1]
            
            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        return merged


class ContrastivePreprocessor(BasePreprocessor):
    
    def __init__(self):
        pass
    
    def prepare_examples(self, messages: List[Message]) -> List[Dict[str, Any]]:
        examples = []
        
        for msg in messages:
            entity_types = [e.label for e in msg.entities]
            entity_texts = [e.text for e in msg.entities]
            
            examples.append({
                'id': msg.id,
                'text': msg.text,
                'label': 1 if msg.is_phishing else 0,  # Binary: phishing vs ham
                'entity_types': entity_types,
                'entity_texts': entity_texts,
                'num_entities': len(msg.entities)
            })
        
        return examples
    
    def create_pairs(
        self, 
        examples: List[Dict[str, Any]]
    ) -> List[Tuple[Dict, Dict, int]]:
        pairs = []
        
        phishing_examples = [ex for ex in examples if ex['label'] == 1]
        ham_examples = [ex for ex in examples if ex['label'] == 0]
        
        for i in range(len(phishing_examples)):
            for j in range(i + 1, len(phishing_examples)):
                pairs.append((phishing_examples[i], phishing_examples[j], 1))
        
        for i in range(len(ham_examples)):
            for j in range(i + 1, min(len(ham_examples), i + 5)):  # Limit ham pairs
                pairs.append((ham_examples[i], ham_examples[j], 1))
        
        import random
        for phishing_ex in phishing_examples:
            for ham_ex in random.sample(ham_examples, min(3, len(ham_examples))):
                pairs.append((phishing_ex, ham_ex, 0))
        
        logger.info(f"Created {len(pairs)} contrastive pairs")
        return pairs

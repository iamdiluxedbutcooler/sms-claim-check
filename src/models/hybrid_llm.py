
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import os

from .entity_ner import EntityNERModel

logger = logging.getLogger(__name__)


class HybridNERLLMModel:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        ner_config = config.get('ner_config', {})
        self.ner_model = EntityNERModel(ner_config)
        
        self.llm_provider = config.get('llm_provider', 'openai')  # or 'anthropic', 'flan-t5'
        self.llm_model = config.get('llm_model', 'gpt-3.5-turbo')
        self.use_local_llm = config.get('use_local_llm', False)
        
        self._init_llm_client()
    
    def _init_llm_client(self):
        if self.use_local_llm or self.llm_provider == 'flan-t5':
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            logger.info(f"Loading local LLM: {self.llm_model}")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
            self.llm = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model)
        elif self.llm_provider == 'openai':
            try:
                from openai import OpenAI
                self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            except ImportError:
                logger.warning("OpenAI package not installed. Install with: pip install openai")
                self.llm = None
        elif self.llm_provider == 'anthropic':
            try:
                from anthropic import Anthropic
                self.llm = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            except ImportError:
                logger.warning("Anthropic package not installed. Install with: pip install anthropic")
                self.llm = None
    
    def train(
        self,
        train_examples: List[Dict],
        val_examples: List[Dict],
        output_dir: Path
    ) -> Dict[str, float]:
        logger.info(f"Training NER component of {self.get_name()}")
        
        ner_metrics = self.ner_model.train(train_examples, val_examples, output_dir / "ner")
        
        logger.info("NER training complete. LLM component uses zero-shot prompting.")
        
        return ner_metrics
    
    def evaluate(self, test_examples: List[Dict]) -> Dict[str, float]:
        logger.info("Evaluating Hybrid model")
        
        ner_metrics = self.ner_model.evaluate(test_examples)
        
        return ner_metrics
    
    def predict(self, text: str) -> Dict[str, Any]:
        ner_result = self.ner_model.predict(text)
        entities = ner_result['entities']
        
        structured_claims = self._structure_claims_with_llm(text, entities)
        
        return {
            'text': text,
            'entities': entities,
            'structured_claims': structured_claims
        }
    
    def _structure_claims_with_llm(
        self, 
        text: str, 
        entities: List[Dict]
    ) -> List[Dict]:
        if not entities:
            return []
        
        prompt = self._build_structuring_prompt(text, entities)
        
        if self.use_local_llm or self.llm_provider == 'flan-t5':
            response = self._call_local_llm(prompt)
        elif self.llm_provider == 'openai':
            response = self._call_openai(prompt)
        elif self.llm_provider == 'anthropic':
            response = self._call_anthropic(prompt)
        else:
            logger.error(f"Unknown LLM provider: {self.llm_provider}")
            return []
        
        structured_claims = self._parse_llm_response(response)
        
        return structured_claims
    
    def _build_structuring_prompt(self, text: str, entities: List[Dict]) -> str:
        entities_str = '\n'.join([
            f"- {e['text']} ({e['type']})"
            for e in entities
        ])
        
        prompt = f"""Given an SMS message and extracted entities, identify and structure the phishing claims.

SMS Text: "{text}"

Extracted Entities:
{entities_str}

Task: Structure the claims in this format: [Subject] [Predicate] [Object] [Time]

Example:
- Subject: The entity making the claim (e.g., "PayPal", "Your account")
- Predicate: The action or state (e.g., "will be suspended", "requires verification")
- Object: What is affected (e.g., "payment", "access")
- Time: When (e.g., "tonight", "within 24 hours", or null if not specified)

Output the claims as JSON array:
[
  {{
    "subject": "...",
    "predicate": "...",
    "object": "...",
    "time": "..."
  }}
]
"""
        return prompt
    
    def _call_local_llm(self, prompt: str) -> str:
        import torch
        
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True
            )
        
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def _call_openai(self, prompt: str) -> str:
        if not self.llm:
            return "[]"
        
        try:
            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that structures phishing claims from SMS messages."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=512
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "[]"
    
    def _call_anthropic(self, prompt: str) -> str:
        if not self.llm:
            return "[]"
        
        try:
            response = self.llm.messages.create(
                model=self.llm_model,
                max_tokens=512,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return "[]"
    
    def _parse_llm_response(self, response: str) -> List[Dict]:
        try:
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                claims = json.loads(json_match.group(0))
                return claims
            return []
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return []
    
    def save(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ner_dir = output_dir / "ner"
        self.ner_model.save(ner_dir)
        
        with open(output_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Saved Hybrid model to {output_dir}")
    
    def load(self, model_dir: Path):
        model_dir = Path(model_dir)
        
        logger.info(f"Loading Hybrid model from {model_dir}")
        
        ner_dir = model_dir / "ner"
        self.ner_model.load(ner_dir)
        
        logger.info("Hybrid model loaded successfully")
    
    def get_name(self) -> str:
        return "HybridNERLLMModel"


import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import time
from datetime import datetime

from .entity_ner import EntityNERModel

logger = logging.getLogger(__name__)


class HybridNERLLMModel:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        ner_config = config.get('ner_config', {})
        self.ner_model = EntityNERModel(ner_config)
        
        self.llm_provider = config.get('llm_provider', 'openai')
        self.llm_model = config.get('llm_model', 'gpt-4o-mini')
        self.use_local_llm = config.get('use_local_llm', False)
        self.use_batch_api = config.get('use_batch_api', True)
        self.openai_api_key = None
        
    def _get_api_key(self):
        if self.openai_api_key:
            return self.openai_api_key
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            import getpass
            logger.info("\n" + "="*70)
            logger.info("OpenAI API Key Required")
            logger.info("="*70)
            api_key = getpass.getpass("Enter your OpenAI API key: ").strip()
            logger.info("API key received\n")
        
        self.openai_api_key = api_key
        return api_key
        
    def _init_llm_client(self):
        if self.use_local_llm or self.llm_provider == 'flan-t5':
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            logger.info(f"Loading local LLM: {self.llm_model}")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
            self.llm = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model)
        elif self.llm_provider == 'openai':
            try:
                from openai import OpenAI
                api_key = self._get_api_key()
                self.llm = OpenAI(api_key=api_key)
                logger.info(f"Initialized OpenAI client with model: {self.llm_model}")
                if self.use_batch_api:
                    logger.info("Batch API mode enabled")
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
        
        logger.info(f"NER training complete: {ner_metrics}")
        return ner_metrics
    
    def evaluate(self, test_examples: List[Dict]) -> Dict[str, float]:
        logger.info("Evaluating Hybrid model")
        
        ner_metrics = self.ner_model.evaluate(test_examples)
        
        if self.llm_provider == 'openai' and self.use_batch_api:
            logger.info("Using OpenAI Batch API for evaluation")
            texts = [ex['text'] for ex in test_examples]
            self._process_batch(texts, purpose="evaluation")
        
        return ner_metrics
    
    def _process_batch(self, texts: List[str], purpose: str = "inference") -> Optional[str]:
        if not self.llm:
            logger.error("OpenAI client not initialized")
            return None
        
        batch_file = self._create_batch_input_file(texts)
        
        logger.info(f"Uploading batch file for {len(texts)} texts...")
        with open(batch_file, 'rb') as f:
            file_response = self.llm.files.create(file=f, purpose='batch')
        
        batch_input_file_id = file_response.id
        logger.info(f"Batch file uploaded: {batch_input_file_id}")
        
        logger.info("Creating batch job...")
        batch_response = self.llm.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "purpose": purpose,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        batch_id = batch_response.id
        logger.info(f"Batch job created: {batch_id}")
        logger.info(f"Status: {batch_response.status}")
        logger.info("Batch will complete within 24 hours. Check status with batch_id.")
        
        return batch_id
    
    def _create_batch_input_file(self, texts: List[str]) -> Path:
        batch_requests = []
        
        for idx, text in enumerate(texts):
            ner_result = self.ner_model.predict(text)
            entities = ner_result['entities']
            
            if not entities:
                continue
            
            prompt = self._build_structuring_prompt(text, entities)
            
            request = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.llm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that structures phishing claims from SMS messages."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 512
                }
            }
            batch_requests.append(request)
        
        batch_file = Path(f"batch_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        
        with open(batch_file, 'w') as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
        
        logger.info(f"Created batch input file: {batch_file} ({len(batch_requests)} requests)")
        return batch_file
    
    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        if not self.llm:
            logger.error("OpenAI client not initialized")
            return {}
        
        batch = self.llm.batches.retrieve(batch_id)
        
        status_info = {
            'id': batch.id,
            'status': batch.status,
            'created_at': batch.created_at,
            'completed_at': batch.completed_at,
            'failed_at': batch.failed_at,
            'request_counts': {
                'total': batch.request_counts.total,
                'completed': batch.request_counts.completed,
                'failed': batch.request_counts.failed
            }
        }
        
        logger.info(f"Batch {batch_id} status: {batch.status}")
        logger.info(f"Progress: {batch.request_counts.completed}/{batch.request_counts.total}")
        
        return status_info
    
    def retrieve_batch_results(self, batch_id: str, output_file: Optional[Path] = None) -> List[Dict]:
        if not self.llm:
            logger.error("OpenAI client not initialized")
            return []
        
        batch = self.llm.batches.retrieve(batch_id)
        
        if batch.status != 'completed':
            logger.warning(f"Batch not completed yet. Status: {batch.status}")
            return []
        
        if not batch.output_file_id:
            logger.error("No output file available")
            return []
        
        logger.info(f"Downloading batch results from file: {batch.output_file_id}")
        file_response = self.llm.files.content(batch.output_file_id)
        
        if output_file:
            output_file = Path(output_file)
            output_file.write_text(file_response.text)
            logger.info(f"Batch results saved to: {output_file}")
        
        results = []
        for line in file_response.text.strip().split('\n'):
            result = json.loads(line)
            results.append(result)
        
        logger.info(f"Retrieved {len(results)} batch results")
        return results
    
    def parse_batch_results(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        structured_outputs = {}
        
        for result in results:
            custom_id = result.get('custom_id', '')
            
            if result.get('response', {}).get('status_code') == 200:
                body = result['response']['body']
                content = body['choices'][0]['message']['content']
                
                claims = self._parse_llm_response(content)
                structured_outputs[custom_id] = claims
            else:
                logger.error(f"Request {custom_id} failed: {result.get('error')}")
                structured_outputs[custom_id] = []
        
        return structured_outputs
    
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

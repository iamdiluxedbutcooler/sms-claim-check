#!/usr/bin/env python3
"""
Automated annotation using OpenAI GPT-4 for both entity-based and claim-based NER.
Processes all 2000 messages from balanced dataset with robust prompts and validation.
"""

import json
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from datetime import datetime
import re

# Initialize OpenAI client (reads from environment variable)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ============================================================================
# ENTITY-BASED ANNOTATION PROMPT
# ============================================================================

ENTITY_SYSTEM_PROMPT = """You are an expert NLP annotator specializing in Named Entity Recognition (NER) for SMS phishing detection.

Your task is to extract concrete, verifiable entities from SMS messages with precise character-level spans.

You MUST follow these rules strictly:
1. Annotate minimal spans only (no extra words)
2. Character positions must be exact (start is inclusive, end is exclusive)
3. Exclude trailing/leading punctuation unless part of entity
4. When in doubt, prefer precision over recall
5. Output ONLY valid JSON, nothing else"""

ENTITY_USER_PROMPT = """Extract entities from this SMS message using these EXACT entity types:

**BRAND**: Company, organization, or service names
- Examples: "Amazon", "PayPal", "IRS", "USPS", "DHL", "Netflix", "HMRC"

**PHONE**: Phone numbers in any format
- Examples: "1-800-123-4567", "(555) 123-4567", "08001234567"

**URL**: Web links, domains, shortened links
- Examples: "http://amazon.com/verify", "bit.ly/abc123", "amzn.to/xyz"

**ORDER_ID**: Order numbers, tracking IDs, reference codes
- Examples: "#12345", "TRK-9876543210", "Order: ABC123", "Ref:12345"

**AMOUNT**: Monetary amounts
- Examples: "$50", "£100", "€99.99", "500 GBP"

**DATE**: Specific dates or temporal references (NOT urgent)
- Examples: "12/25/2024", "December 25", "Monday", "next week"

**DEADLINE**: Time pressure phrases indicating urgency
- Examples: "within 24 hours", "immediately", "by tonight", "expires today", "URGENT"

**ACTION_REQUIRED**: Imperative action verbs or phrases
- Examples: "Click here", "Call now", "Verify account", "Confirm", "Reply YES"

---

**CRITICAL INSTRUCTIONS FOR CHARACTER POSITIONS**:
1. Count characters VERY CAREFULLY from the start of the message (position 0)
2. `start` = position of first character of the entity
3. `end` = position AFTER the last character of the entity
4. To verify: message[start:end] should equal the exact text
5. Count spaces, punctuation, everything!

**OUTPUT FORMAT** (JSON only):
{{
  "entities": [
    {{"text": "exact text", "start": 0, "end": 10, "label": "ENTITY_TYPE"}}
  ]
}}

**EXAMPLE**:
Message: "Amazon package #123"
Correct:
{{
  "entities": [
    {{"text": "Amazon", "start": 0, "end": 6, "label": "BRAND"}},
    {{"text": "#123", "start": 15, "end": 19, "label": "ORDER_ID"}}
  ]
}}

**SMS MESSAGE TO ANNOTATE**:
```
{message}
```

Think step by step:
1. Find each entity in the message
2. Count characters carefully from position 0
3. Verify: message[start:end] == text

**YOUR ANNOTATION** (JSON only):"""

# ============================================================================
# CLAIM-BASED ANNOTATION PROMPT
# ============================================================================

CLAIM_SYSTEM_PROMPT = """You are an expert NLP annotator specializing in semantic claim extraction for SMS phishing detection.

Your task is to identify and extract CLAIM PHRASES - semantic units that make assertions, create urgency, or direct actions.

Claims are more abstract than entities. They capture the MEANING and INTENT of phishing tactics.

You MUST follow these rules strictly:
1. Annotate complete claim phrases (may span multiple words)
2. Character positions must be exact (start inclusive, end exclusive)
3. Claims can overlap (e.g., "Click now" is both ACTION_CLAIM and URGENCY_CLAIM)
4. Capture implicit claims even without explicit entities
5. Output ONLY valid JSON, nothing else"""

CLAIM_USER_PROMPT = """Extract claim phrases from this SMS message using these EXACT claim types:

**IDENTITY_CLAIM**: Assertions about who sent the message
- Examples: "We are Amazon", "From PayPal", "IRS Department", "Your bank", "Official notification"
- Captures: Sender identity, authority claims, impersonation attempts
- Can be implicit: Just "Amazon:" at start implies identity claim

**DELIVERY_CLAIM**: Assertions about packages, shipments, deliveries
- Examples: "Your package is delayed", "Parcel awaiting delivery", "USPS shipment failed"
- Captures: Delivery status, package issues, shipping problems
- Include: Any mention of package/parcel/delivery status

**FINANCIAL_CLAIM**: Assertions about money, payments, refunds, charges
- Examples: "You won $5000", "Tax refund pending", "Unauthorized charge of $99", "Prize money"
- Captures: Money owed/won, refunds, charges, prizes, rewards
- Include: Financial benefits or threats

**ACCOUNT_CLAIM**: Assertions about account status or security
- Examples: "Your account is suspended", "Account locked", "Security alert", "Unusual activity"
- Captures: Account problems, security issues, access restrictions
- Include: Account status changes, security threats

**URGENCY_CLAIM**: Time-pressure tactics and urgent language
- Examples: "within 24 hours", "Act now", "Expires tonight", "Immediate action required"
- Captures: Deadline pressure, urgency, time constraints
- Include: "now", "urgent", "immediately", "expires", "limited time"

**ACTION_CLAIM**: Required or requested user actions
- Examples: "Click here to verify", "Call 1-800-555-0199", "Reply YES", "Confirm your details"
- Captures: Instructions, commands, calls-to-action
- Include: Imperative verbs + context (not just verb alone)

**VERIFICATION_CLAIM**: Requests to verify, confirm, or validate
- Examples: "Verify your identity", "Confirm your account", "Re-validate payment method"
- Captures: Verification requests, confirmation demands
- Include: Any request to prove/verify/confirm identity or information

**SECURITY_CLAIM**: Security threats or warnings
- Examples: "Suspicious login detected", "Security breach", "Your data is at risk"
- Captures: Security alerts, threats, warnings about compromised security
- Include: Security incidents, breaches, suspicious activity

**REWARD_CLAIM**: Promises of rewards, prizes, or benefits
- Examples: "You've won a prize", "Claim your reward", "Free gift waiting", "Exclusive offer"
- Captures: Prizes, rewards, gifts, special offers
- Include: Contest wins, giveaways, exclusive benefits

**LEGAL_CLAIM**: Legal threats or official demands
- Examples: "Legal action pending", "Court summons", "Tax authority", "Final notice"
- Captures: Legal consequences, official demands, authority threats
- Include: Legal terminology, official language

**SOCIAL_CLAIM**: Social engineering tactics
- Examples: "Your friend sent you", "Family member in trouble", "Someone shared with you"
- Captures: Social pressure, relationships, emotional manipulation
- Include: References to friends, family, social connections

**CREDENTIALS_CLAIM**: Requests for passwords, PINs, or sensitive data
- Examples: "Enter your password", "Provide your PIN", "Confirm CVV", "Social security number"
- Captures: Requests for sensitive credentials or personal information
- Include: Any request for passwords, codes, SSN, credit card details

---

**OUTPUT FORMAT** (JSON only, no markdown):
{{
  "claims": [
    {{"text": "exact claim phrase", "start": 0, "end": 20, "label": "CLAIM_TYPE"}},
    {{"text": "another claim", "start": 25, "end": 45, "label": "CLAIM_TYPE"}}
  ]
}}

**CRITICAL RULES**:
- `start` is the character index where claim begins (0-indexed, inclusive)
- `end` is the character index where claim ends (exclusive)
- Verify: `message[start:end] == text` must be TRUE
- Claims CAN overlap (same text can have multiple claim types)
- If message is benign (HAM), return: {{"claims": []}}
- DO NOT include explanations, only JSON

**SMS MESSAGE TO ANNOTATE**:
```
{message}
```

**YOUR ANNOTATION** (JSON only):"""

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_spans(message: str, annotations: List[Dict], annotation_type: str = "entities") -> bool:
    """Validate that all spans are correct"""
    for ann in annotations:
        text = ann.get('text', '')
        start = ann.get('start', -1)
        end = ann.get('end', -1)
        
        # Check indices
        if start < 0 or end < 0 or start >= end:
            print(f"  WARNING: Invalid indices for '{text}': start={start}, end={end}")
            return False
        
        if end > len(message):
            print(f"  WARNING: End index {end} exceeds message length {len(message)}")
            return False
        
        # Check text match
        extracted = message[start:end]
        if extracted != text:
            print(f"  WARNING: Span mismatch: expected '{text}', got '{extracted}'")
            print(f"           Indices: [{start}:{end}]")
            return False
    
    return True

def fix_json_response(response_text: str) -> str:
    """Clean up JSON response from GPT"""
    # Remove markdown code blocks
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)
    
    # Strip whitespace
    response_text = response_text.strip()
    
    return response_text

# ============================================================================
# ANNOTATION FUNCTIONS
# ============================================================================

def find_span_in_message(message: str, text: str, start_search: int = 0) -> Optional[tuple]:
    """Find the start and end positions of text in message"""
    try:
        start = message.index(text, start_search)
        end = start + len(text)
        return (start, end)
    except ValueError:
        # Try case-insensitive
        try:
            lower_msg = message.lower()
            lower_text = text.lower()
            start = lower_msg.index(lower_text, start_search)
            end = start + len(text)
            return (start, end)
        except ValueError:
            return None

def annotate_entities(message: str, retry_count: int = 3) -> Optional[List[Dict]]:
    """Annotate a single message with entity-based NER"""
    # Simpler prompt - just ask for text and labels, we'll find positions
    simple_prompt = f"""Extract entities from this SMS message. Return JSON with entity text and label only.

Entity types: BRAND, PHONE, URL, ORDER_ID, AMOUNT, DATE, DEADLINE, ACTION_REQUIRED

Output format:
{{
  "entities": [
    {{"text": "entity text exactly as it appears", "label": "ENTITY_TYPE"}}
  ]
}}

SMS: {message}

JSON:"""
    
    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert NLP annotator. Extract entities and return only JSON."},
                    {"role": "user", "content": simple_prompt}
                ],
                temperature=0,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            content = fix_json_response(content)
            
            result = json.loads(content)
            raw_entities = result.get('entities', [])
            
            # Find positions for each entity
            entities = []
            used_positions = set()
            
            for ent in raw_entities:
                text = ent.get('text', '').strip()
                label = ent.get('label', '')
                
                if not text or not label:
                    continue
                
                # Try to find this text in the message
                search_start = 0
                found = False
                
                while search_start < len(message):
                    span = find_span_in_message(message, text, search_start)
                    if span:
                        start, end = span
                        # Check if this position overlaps with already used positions
                        if start not in used_positions:
                            entities.append({
                                "text": text,
                                "start": start,
                                "end": end,
                                "label": label
                            })
                            for i in range(start, end):
                                used_positions.add(i)
                            found = True
                            break
                        else:
                            search_start = start + 1
                    else:
                        break
                
                if not found:
                    print(f"  WARNING: Could not find '{text}' in message")
            
            # Validate spans
            if entities and not validate_spans(message, entities, "entities"):
                if attempt < retry_count - 1:
                    print(f"  Validation failed, retrying... (attempt {attempt + 2}/{retry_count})")
                    time.sleep(1)
                    continue
                else:
                    print(f"  Validation failed after {retry_count} attempts, returning empty")
                    return []
            
            return entities
            
        except json.JSONDecodeError as e:
            print(f"  JSON decode error: {e}")
            if attempt < retry_count - 1:
                time.sleep(1)
                continue
            return []
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retry_count - 1:
                time.sleep(2)
                continue
            return []
    
    return []

def annotate_claims(message: str, label: str, retry_count: int = 3) -> Optional[List[Dict]]:
    """Annotate a single message with claim-based NER"""
    # HAM messages get no claims automatically
    if label == 'ham':
        return []
    
    # Simpler prompt - just ask for text and labels
    simple_claim_prompt = f"""Extract claim phrases from this SMS phishing message. Return JSON with claim text and type only.

Claim types: IDENTITY_CLAIM, DELIVERY_CLAIM, FINANCIAL_CLAIM, ACCOUNT_CLAIM, URGENCY_CLAIM, ACTION_CLAIM, VERIFICATION_CLAIM, SECURITY_CLAIM, REWARD_CLAIM, LEGAL_CLAIM, SOCIAL_CLAIM, CREDENTIALS_CLAIM

Output format:
{{
  "claims": [
    {{"text": "claim phrase exactly as it appears", "label": "CLAIM_TYPE"}}
  ]
}}

SMS: {message}

JSON:"""
    
    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting semantic claims from phishing messages. Return only JSON."},
                    {"role": "user", "content": simple_claim_prompt}
                ],
                temperature=0,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            content = fix_json_response(content)
            
            result = json.loads(content)
            raw_claims = result.get('claims', [])
            
            # Find positions for each claim
            claims = []
            
            for claim in raw_claims:
                text = claim.get('text', '').strip()
                label_type = claim.get('label', '')
                
                if not text or not label_type:
                    continue
                
                # Try to find this text in the message
                span = find_span_in_message(message, text)
                if span:
                    start, end = span
                    claims.append({
                        "text": text,
                        "start": start,
                        "end": end,
                        "label": label_type
                    })
                else:
                    print(f"  WARNING: Could not find claim '{text}' in message")
            
            # Validate spans
            if claims and not validate_spans(message, claims, "claims"):
                if attempt < retry_count - 1:
                    print(f"  Validation failed, retrying... (attempt {attempt + 2}/{retry_count})")
                    time.sleep(1)
                    continue
                else:
                    print(f"  Validation failed after {retry_count} attempts, returning empty")
                    return []
            
            return claims
            
        except json.JSONDecodeError as e:
            print(f"  JSON decode error: {e}")
            if attempt < retry_count - 1:
                time.sleep(1)
                continue
            return []
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retry_count - 1:
                time.sleep(2)
                continue
            return []
    
    return []

# ============================================================================
# LABEL STUDIO FORMAT CONVERSION
# ============================================================================

def convert_to_label_studio_format(task_id: int, message: str, entities: List[Dict], 
                                   meta: Dict, annotation_type: str = "entity") -> Dict:
    """Convert annotations to Label Studio format"""
    
    # Create NER labels in Label Studio format
    result = []
    for ent in entities:
        label_type = "labels" if annotation_type == "entity" else "labels"
        result.append({
            "value": {
                "start": ent['start'],
                "end": ent['end'],
                "text": ent['text'],
                label_type: [ent['label']]
            },
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
        })
    
    task = {
        "id": task_id,
        "data": {
            "text": message
        },
        "annotations": [{
            "id": task_id,
            "result": result,
            "was_cancelled": False,
            "ground_truth": False,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "lead_time": 0,
            "prediction": {},
            "result_count": 0
        }],
        "meta": meta
    }
    
    return task

# ============================================================================
# MAIN ANNOTATION PIPELINE
# ============================================================================

def annotate_dataset(input_path: str, output_entity_path: str, output_claim_path: str, 
                     start_idx: int = 0, limit: Optional[int] = None):
    """Annotate entire dataset with both entity and claim schemas"""
    
    print("="*60)
    print("AUTOMATED ANNOTATION PIPELINE")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output Entity: {output_entity_path}")
    print(f"Output Claim: {output_claim_path}")
    print()
    
    # Load dataset
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} messages")
    
    if limit:
        dataset = dataset[start_idx:start_idx + limit]
        print(f"Processing subset: {len(dataset)} messages (starting from index {start_idx})")
    
    # Track progress
    entity_annotations = []
    claim_annotations = []
    
    total = len(dataset)
    start_time = time.time()
    
    for idx, task in enumerate(dataset, start=1):
        message = task['data']['text']
        meta = task.get('meta', {})
        label = meta.get('label', 'phishing')
        
        print(f"\n[{idx}/{total}] Processing message (label={label})...")
        print(f"  Text: {message[:80]}{'...' if len(message) > 80 else ''}")
        
        # Annotate entities
        print(f"  Annotating entities...")
        entities = annotate_entities(message)
        print(f"    Found {len(entities)} entities")
        
        # Annotate claims
        print(f"  Annotating claims...")
        claims = annotate_claims(message, label)
        print(f"    Found {len(claims)} claims")
        
        # Convert to Label Studio format
        entity_task = convert_to_label_studio_format(
            task_id=task['id'],
            message=message,
            entities=entities,
            meta=meta,
            annotation_type="entity"
        )
        
        claim_task = convert_to_label_studio_format(
            task_id=task['id'],
            message=message,
            entities=claims,
            meta=meta,
            annotation_type="claim"
        )
        
        entity_annotations.append(entity_task)
        claim_annotations.append(claim_task)
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = (total - idx) * avg_time
        
        print(f"  Progress: {idx}/{total} ({idx/total*100:.1f}%)")
        print(f"  Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min")
        
        # Rate limiting (avoid hitting API limits)
        if idx % 10 == 0:
            print(f"  Saving checkpoint...")
            save_annotations(entity_annotations, output_entity_path)
            save_annotations(claim_annotations, output_claim_path)
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    # Final save
    print("\n" + "="*60)
    print("SAVING FINAL ANNOTATIONS")
    print("="*60)
    
    save_annotations(entity_annotations, output_entity_path)
    save_annotations(claim_annotations, output_claim_path)
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("ANNOTATION COMPLETE")
    print("="*60)
    print(f"Total messages: {total}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average time per message: {total_time/total:.1f} seconds")
    print()
    print(f"Entity annotations: {output_entity_path}")
    print(f"  Total entities: {sum(len(t['annotations'][0]['result']) for t in entity_annotations)}")
    print()
    print(f"Claim annotations: {output_claim_path}")
    print(f"  Total claims: {sum(len(t['annotations'][0]['result']) for t in claim_annotations)}")
    print("="*60)

def save_annotations(annotations: List[Dict], output_path: str):
    """Save annotations to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"    Saved {len(annotations)} annotations to {output_path}")
    print(f"    File size: {output_path.stat().st_size / 1024:.1f} KB")

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Paths
    input_path = "data/annotations/balanced_dataset_2000.json"
    output_entity_path = "data/annotations/entity_annotations_2000.json"
    output_claim_path = "data/annotations/claim_annotations_2000.json"
    
    # Run annotation
    annotate_dataset(
        input_path=input_path,
        output_entity_path=output_entity_path,
        output_claim_path=output_claim_path,
        start_idx=0,
        limit=None  # Set to None to process all, or a number to test
    )

if __name__ == "__main__":
    main()

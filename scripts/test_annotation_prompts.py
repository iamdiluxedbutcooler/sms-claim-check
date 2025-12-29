#!/usr/bin/env python3
"""
Test the annotation prompts with sample messages to verify quality.
"""

import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Load prompts from main script
import sys
sys.path.append('scripts')

# Test messages
TEST_MESSAGES = [
    {
        "text": "Amazon: Your package #ABC123 is delayed. Click here to track: bit.ly/track123 or call 1-800-555-0199 within 24h",
        "label": "phishing"
    },
    {
        "text": "URGENT! Your PayPal account has been suspended due to unusual activity. Verify now at paypal-secure.com or your account will be closed permanently.",
        "label": "phishing"
    },
    {
        "text": "Hey, are you free for lunch tomorrow? Let me know!",
        "label": "ham"
    },
    {
        "text": "You have won £5000 in our lottery! Call 09012345678 NOW to claim your prize. Expires today!",
        "label": "phishing"
    }
]

# Import prompts
with open('scripts/automated_annotation.py', 'r') as f:
    script_content = f.read()
    
print("="*80)
print("PROMPT TESTING - ENTITY-BASED ANNOTATIONS")
print("="*80)

for i, msg in enumerate(TEST_MESSAGES, 1):
    text = msg['text']
    label = msg['label']
    
    print(f"\n{'='*80}")
    print(f"TEST MESSAGE {i} (Label: {label})")
    print(f"{'='*80}")
    print(f"Text: {text}")
    print(f"\nCalling OpenAI API (entity annotation)...")
    
    # Use the actual prompt format from the script
    entity_system = """You are an expert NLP annotator specializing in Named Entity Recognition (NER) for SMS phishing detection.

Your task is to extract concrete, verifiable entities from SMS messages with precise character-level spans.

You MUST follow these rules strictly:
1. Annotate minimal spans only (no extra words)
2. Character positions must be exact (start is inclusive, end is exclusive)
3. Exclude trailing/leading punctuation unless part of entity
4. When in doubt, prefer precision over recall
5. Output ONLY valid JSON, nothing else"""

    entity_user = f"""Extract entities from this SMS message using these EXACT entity types:

**BRAND**: Company, organization, or service names
- Examples: "Amazon", "PayPal", "IRS", "USPS", "DHL", "Netflix", "HMRC"

**PHONE**: Phone numbers in any format
- Examples: "1-800-123-4567", "(555) 123-4567", "08001234567"

**URL**: Web links, domains, shortened links
- Examples: "http://amazon.com/verify", "bit.ly/abc123", "amzn.to/xyz"

**ORDER_ID**: Order numbers, tracking IDs, reference codes
- Examples: "#12345", "TRK-9876543210", "Order: ABC123"

**AMOUNT**: Monetary amounts
- Examples: "$50", "£100", "€99.99", "500 GBP"

**DATE**: Specific dates or temporal references (NOT urgent)
- Examples: "12/25/2024", "December 25", "Monday"

**DEADLINE**: Time pressure phrases
- Examples: "within 24 hours", "immediately", "expires today"

**ACTION_REQUIRED**: Imperative action verbs or phrases
- Examples: "Click here", "Call now", "Verify account"

OUTPUT FORMAT (JSON only):
{{
  "entities": [
    {{"text": "exact text", "start": 0, "end": 10, "label": "ENTITY_TYPE"}}
  ]
}}

SMS MESSAGE:
```
{text}
```

YOUR ANNOTATION (JSON only):"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": entity_system},
                {"role": "user", "content": entity_user}
            ],
            temperature=0,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        print(f"\nENTITY ANNOTATIONS:")
        print(json.dumps(result, indent=2))
        
        # Validate spans
        entities = result.get('entities', [])
        print(f"\nVALIDATION:")
        for ent in entities:
            extracted = text[ent['start']:ent['end']]
            match = "[OK]" if extracted == ent['text'] else "[FAIL] MISMATCH"
            print(f"  {match} [{ent['start']}:{ent['end']}] '{ent['text']}' ({ent['label']})")
            if extracted != ent['text']:
                print(f"       Expected: '{ent['text']}'")
                print(f"       Got: '{extracted}'")
        
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "="*80)
print("PROMPT TESTING - CLAIM-BASED ANNOTATIONS")
print("="*80)

for i, msg in enumerate(TEST_MESSAGES, 1):
    text = msg['text']
    label = msg['label']
    
    print(f"\n{'='*80}")
    print(f"TEST MESSAGE {i} (Label: {label})")
    print(f"{'='*80}")
    print(f"Text: {text}")
    
    if label == 'ham':
        print(f"\nSKIPPING (HAM messages get no claims)")
        continue
    
    print(f"\nCalling OpenAI API (claim annotation)...")
    
    claim_system = """You are an expert NLP annotator specializing in semantic claim extraction for SMS phishing detection.

Your task is to identify and extract CLAIM PHRASES - semantic units that make assertions, create urgency, or direct actions.

You MUST follow these rules strictly:
1. Annotate complete claim phrases (may span multiple words)
2. Character positions must be exact
3. Claims can overlap
4. Output ONLY valid JSON, nothing else"""

    claim_user = f"""Extract claim phrases using these claim types:

**IDENTITY_CLAIM**: Who sent the message
**DELIVERY_CLAIM**: Package/shipment assertions
**FINANCIAL_CLAIM**: Money, payments, refunds
**ACCOUNT_CLAIM**: Account status/security
**URGENCY_CLAIM**: Time pressure
**ACTION_CLAIM**: Required actions
**VERIFICATION_CLAIM**: Verification requests
**SECURITY_CLAIM**: Security threats
**REWARD_CLAIM**: Prizes, rewards
**LEGAL_CLAIM**: Legal threats
**SOCIAL_CLAIM**: Social engineering
**CREDENTIALS_CLAIM**: Password/PIN requests

OUTPUT FORMAT (JSON only):
{{
  "claims": [
    {{"text": "exact claim phrase", "start": 0, "end": 20, "label": "CLAIM_TYPE"}}
  ]
}}

SMS MESSAGE:
```
{text}
```

YOUR ANNOTATION (JSON only):"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": claim_system},
                {"role": "user", "content": claim_user}
            ],
            temperature=0,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        print(f"\nCLAIM ANNOTATIONS:")
        print(json.dumps(result, indent=2))
        
        # Validate spans
        claims = result.get('claims', [])
        print(f"\nVALIDATION:")
        for claim in claims:
            extracted = text[claim['start']:claim['end']]
            match = "[OK]" if extracted == claim['text'] else "[FAIL] MISMATCH"
            print(f"  {match} [{claim['start']}:{claim['end']}] '{claim['text']}' ({claim['label']})")
            if extracted != claim['text']:
                print(f"       Expected: '{claim['text']}'")
                print(f"       Got: '{extracted}'")
        
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "="*80)
print("PROMPT TEST COMPLETE")
print("="*80)

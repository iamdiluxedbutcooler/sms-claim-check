"""
Hybrid LLM Prompts for Claim Extraction

This module contains prompts for the two hybrid approaches:
1. Entity-NER + LLM: Extract entities with NER model, then use LLM to structure into claims
2. Claim-NER + LLM: Extract claim phrases with NER model, then use LLM to structure and verify

These are used during INFERENCE, not training.
"""

# ============================================================================
# Approach 3a: Entity-NER + LLM Parsing
# ============================================================================

ENTITY_TO_CLAIM_PROMPT = """You are a claim structuring agent for phishing detection. You receive extracted entities from an SMS message and must construct structured, verifiable claims.

**Input:**
Message: {message}

Extracted Entities:
{entities}

**Your Task:**
Transform these entities into structured, verifiable claims. Each claim should:
1. Combine relevant entities into meaningful assertions
2. Identify what needs to be verified
3. Specify the verification method (check sender domain, verify tracking ID, confirm account status, etc.)
4. Categorize the claim type

**Claim Types:**
- IDENTITY_CLAIM: Who claims to have sent this (brand/organization)
- DELIVERY_CLAIM: Assertions about packages/deliveries
- FINANCIAL_CLAIM: Money, payments, refunds, charges
- ACCOUNT_CLAIM: Account status, security, access
- URGENCY_CLAIM: Time-sensitive assertions
- ACTION_CLAIM: Required user actions

**Output Format:**
{{
  "structured_claims": [
    {{
      "claim_type": "IDENTITY_CLAIM",
      "claim_text": "Message claims to be from Amazon",
      "confidence": "high|medium|low",
      "entities_used": ["BRAND: Amazon"],
      "verification_steps": [
        {{
          "step": "Check sender domain/number",
          "method": "Compare against Amazon's official contact list",
          "data_needed": ["sender_phone_number", "sender_domain"]
        }}
      ],
      "risk_indicators": ["unofficial contact method", "unexpected message"]
    }},
    {{
      "claim_type": "DELIVERY_CLAIM",
      "claim_text": "Package ABC123 is delayed",
      "confidence": "medium",
      "entities_used": ["ORDER_ID: ABC123"],
      "verification_steps": [
        {{
          "step": "Verify tracking number exists",
          "method": "Query Amazon tracking API",
          "data_needed": ["tracking_number", "user_account"]
        }},
        {{
          "step": "Confirm delivery status",
          "method": "Check if package is actually delayed",
          "data_needed": ["tracking_number", "expected_delivery_date"]
        }}
      ],
      "risk_indicators": ["unsolicited notification", "tracking number format unusual"]
    }},
    {{
      "claim_type": "ACTION_CLAIM",
      "claim_text": "User must click link to resolve issue",
      "confidence": "high",
      "entities_used": ["ACTION_REQUIRED: Click", "URL: bit.ly/pkg123"],
      "verification_steps": [
        {{
          "step": "Verify URL legitimacy",
          "method": "Check domain against brand's official domains",
          "data_needed": ["url", "brand_official_domains"]
        }},
        {{
          "step": "Check if link is shortened",
          "method": "Expand shortened URL and verify destination",
          "data_needed": ["shortened_url"]
        }}
      ],
      "risk_indicators": ["shortened URL", "urgent action required", "unusual domain"]
    }},
    {{
      "claim_type": "URGENCY_CLAIM",
      "claim_text": "Action required within 24 hours",
      "confidence": "high",
      "entities_used": ["DEADLINE: within 24h"],
      "verification_steps": [
        {{
          "step": "Verify legitimacy of urgency",
          "method": "Check if brand typically imposes such deadlines",
          "data_needed": ["brand_communication_patterns"]
        }}
      ],
      "risk_indicators": ["artificial urgency", "pressure tactic"]
    }},
    {{
      "claim_type": "FINANCIAL_CLAIM",
      "claim_text": "Payment of $15.99 required",
      "confidence": "medium",
      "entities_used": ["AMOUNT: $15.99"],
      "verification_steps": [
        {{
          "step": "Verify outstanding charges",
          "method": "Check user's account for pending payments",
          "data_needed": ["user_account", "payment_history"]
        }}
      ],
      "risk_indicators": ["unexpected charge", "no prior transaction"]
    }}
  ],
  "overall_risk_assessment": {{
    "risk_level": "HIGH",
    "reasoning": "Multiple high-risk indicators: shortened URL, urgent deadline, unsolicited notification",
    "suspicious_patterns": ["artificial urgency", "unofficial contact", "suspicious URL"],
    "recommendation": "DO NOT CLICK LINK. Contact Amazon directly through official channels."
  }}
}}

Now process the provided message and entities:
"""

# ============================================================================
# Approach 3b: Claim-NER + LLM Parsing
# ============================================================================

CLAIM_TO_STRUCTURED_PROMPT = """You are a claim verification agent for phishing detection. You receive extracted claim phrases from an SMS message and must structure them into verifiable queries.

**Input:**
Message: {message}

Extracted Claims:
{claims}

**Your Task:**
For each extracted claim, provide:
1. Structured representation
2. Verification method
3. Data requirements
4. Risk assessment

**Output Format:**
{{
  "structured_claims": [
    {{
      "original_claim": "Your Amazon package #ABC123 is delayed",
      "claim_type": "DELIVERY_CLAIM",
      "structured_query": {{
        "subject": "Package",
        "subject_identifier": "#ABC123",
        "brand": "Amazon",
        "assertion": "is delayed",
        "implicit_assertions": [
          "User has an Amazon account",
          "User has a pending delivery",
          "Tracking number ABC123 exists",
          "Delivery is associated with user's account"
        ]
      }},
      "verification_steps": [
        {{
          "step": "Authenticate sender",
          "method": "Verify message source is from Amazon",
          "authoritative_sources": ["Amazon official SMS numbers", "Amazon domain"],
          "data_needed": ["sender_phone", "sender_domain"]
        }},
        {{
          "step": "Verify tracking number",
          "method": "Query Amazon tracking system",
          "authoritative_sources": ["Amazon Tracking API", "Amazon customer account"],
          "data_needed": ["tracking_number", "user_amazon_account"]
        }},
        {{
          "step": "Confirm delivery status",
          "method": "Check actual delivery status",
          "authoritative_sources": ["Amazon order history", "Carrier tracking"],
          "data_needed": ["order_number", "expected_delivery_date"]
        }},
        {{
          "step": "Verify user association",
          "method": "Confirm tracking number is linked to user's account",
          "authoritative_sources": ["Amazon account orders"],
          "data_needed": ["user_account", "tracking_number"]
        }}
      ],
      "risk_indicators": [
        "Unsolicited notification",
        "No prior purchase confirmation",
        "Unusual tracking number format"
      ],
      "confidence": "medium",
      "recommendation": "Verify through official Amazon app or website, do not click links in message"
    }},
    {{
      "original_claim": "Click here",
      "claim_type": "ACTION_CLAIM",
      "structured_query": {{
        "action_type": "click",
        "target": "link (URL not in claim text)",
        "purpose": "unstated",
        "urgency": "implied by message context"
      }},
      "verification_steps": [
        {{
          "step": "Verify link destination",
          "method": "Check URL against brand's official domains",
          "authoritative_sources": ["Brand's official website list", "Domain registration records"],
          "data_needed": ["full_url", "brand_official_domains"]
        }},
        {{
          "step": "Check for URL manipulation",
          "method": "Analyze URL for typosquatting, homograph attacks",
          "authoritative_sources": ["URL analysis tools", "Domain reputation services"],
          "data_needed": ["url"]
        }}
      ],
      "risk_indicators": [
        "Generic call-to-action",
        "No explanation of where link leads",
        "Pressure to act without information"
      ],
      "confidence": "high",
      "recommendation": "DO NOT CLICK. High risk of phishing link."
    }},
    {{
      "original_claim": "within 24h",
      "claim_type": "URGENCY_CLAIM",
      "structured_query": {{
        "timeframe": "24 hours",
        "urgency_level": "high",
        "consequence": "implied but not stated",
        "legitimacy": "questionable"
      }},
      "verification_steps": [
        {{
          "step": "Verify urgency legitimacy",
          "method": "Check if brand typically imposes such deadlines",
          "authoritative_sources": ["Brand's official communication policies"],
          "data_needed": ["brand_name", "communication_patterns"]
        }}
      ],
      "risk_indicators": [
        "Artificial time pressure",
        "Common phishing tactic",
        "No legitimate reason for urgency provided"
      ],
      "confidence": "high",
      "recommendation": "Red flag: artificial urgency is a common phishing tactic"
    }}
  ],
  "overall_risk_assessment": {{
    "risk_level": "HIGH",
    "confidence": "high",
    "reasoning": "Multiple high-risk claims including suspicious action request and artificial urgency",
    "phishing_likelihood": "90%",
    "key_red_flags": [
      "Unsolicited notification with urgent action",
      "Generic call-to-action without context",
      "Artificial time pressure",
      "Combination of delivery + urgency + action"
    ],
    "verification_priority": "IMMEDIATE - Do not interact with message",
    "recommended_action": "Delete message. If concerned about delivery, check directly with Amazon through official app/website."
  }}
}}

Now process the provided message and claims:
"""

# ============================================================================
# Batch Processing Prompts
# ============================================================================

def create_entity_to_claim_batch_request(message: str, entities: list, msg_id: str):
    """Create batch API request for Entity-NER + LLM approach"""
    
    # Format entities for prompt
    entities_formatted = []
    for ent in entities:
        entities_formatted.append(f"- {ent['label']}: \"{ent['text']}\"")
    entities_str = "\n".join(entities_formatted)
    
    return {
        "custom_id": f"entity_llm_{msg_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",  # Cheaper for structured parsing
            "messages": [
                {
                    "role": "system",
                    "content": "You are a claim structuring agent that transforms extracted entities into structured, verifiable claims for phishing detection. Always return valid JSON."
                },
                {
                    "role": "user",
                    "content": ENTITY_TO_CLAIM_PROMPT.format(
                        message=message,
                        entities=entities_str
                    )
                }
            ],
            "temperature": 0,
            "max_tokens": 2500,
            "response_format": {"type": "json_object"}
        }
    }


def create_claim_to_structured_batch_request(message: str, claims: list, msg_id: str):
    """Create batch API request for Claim-NER + LLM approach"""
    
    # Format claims for prompt
    claims_formatted = []
    for claim in claims:
        claims_formatted.append(f"- {claim['label']}: \"{claim['text']}\"")
    claims_str = "\n".join(claims_formatted)
    
    return {
        "custom_id": f"claim_llm_{msg_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",  # Cheaper for structured parsing
            "messages": [
                {
                    "role": "system",
                    "content": "You are a claim verification agent that structures extracted claims into verifiable queries for phishing detection. Always return valid JSON."
                },
                {
                    "role": "user",
                    "content": CLAIM_TO_STRUCTURED_PROMPT.format(
                        message=message,
                        claims=claims_str
                    )
                }
            ],
            "temperature": 0,
            "max_tokens": 3000,
            "response_format": {"type": "json_object"}
        }
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example message
    example_message = "URGENT: Your Amazon package #ABC123 is delayed. Click here: bit.ly/pkg123 or call 1-800-555-0199 within 24h to resolve. Cost: $15.99"
    
    # Example entities (from Entity-NER model)
    example_entities = [
        {"text": "URGENT", "label": "DEADLINE"},
        {"text": "Amazon", "label": "BRAND"},
        {"text": "#ABC123", "label": "ORDER_ID"},
        {"text": "Click", "label": "ACTION_REQUIRED"},
        {"text": "bit.ly/pkg123", "label": "URL"},
        {"text": "call", "label": "ACTION_REQUIRED"},
        {"text": "1-800-555-0199", "label": "PHONE"},
        {"text": "within 24h", "label": "DEADLINE"},
        {"text": "$15.99", "label": "AMOUNT"}
    ]
    
    # Example claims (from Claim-NER model)
    example_claims = [
        {"text": "Your Amazon package #ABC123 is delayed", "label": "DELIVERY_CLAIM"},
        {"text": "Click here", "label": "ACTION_CLAIM"},
        {"text": "call 1-800-555-0199", "label": "ACTION_CLAIM"},
        {"text": "within 24h", "label": "URGENCY_CLAIM"},
        {"text": "Cost: $15.99", "label": "FINANCIAL_CLAIM"}
    ]
    
    print("=" * 80)
    print("HYBRID APPROACH 3a: Entity-NER + LLM")
    print("=" * 80)
    request1 = create_entity_to_claim_batch_request(example_message, example_entities, "test_1")
    print(json.dumps(request1, indent=2))
    
    print("\n" + "=" * 80)
    print("HYBRID APPROACH 3b: Claim-NER + LLM")
    print("=" * 80)
    request2 = create_claim_to_structured_batch_request(example_message, example_claims, "test_1")
    print(json.dumps(request2, indent=2))

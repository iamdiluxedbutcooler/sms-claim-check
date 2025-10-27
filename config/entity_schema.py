from typing import Dict, List

ENTITY_TYPES: Dict[str, Dict[str, str]] = {
    "BRAND": {
        "description": "Company, organization, or service names",
        "examples": ["Amazon", "PayPal", "IRS", "USPS", "DHL", "Apple", "Netflix"],
        "color": "#FF6B6B",
    },
    "PHONE": {
        "description": "Phone numbers in any format",
        "examples": ["1-800-123-4567", "(555) 123-4567", "18001234567", "+1-555-123-4567"],
        "color": "#4ECDC4",
    },
    "URL": {
        "description": "Web links (full URLs or shortened links)",
        "examples": ["http://amazon.com/verify", "bit.ly/abc123", "amzn.to/xyz", "suspicious-link.ru"],
        "color": "#45B7D1",
    },
    "ORDER_ID": {
        "description": "Order numbers, tracking IDs, invoice numbers, confirmation codes",
        "examples": ["#12345", "TRK-9876543210", "Invoice INV-2024-001", "Ref: ABC123"],
        "color": "#FFA07A",
    },
    "AMOUNT": {
        "description": "Monetary amounts (with or without currency symbols)",
        "examples": ["$50", "50 USD", "€100", "500.00", "$1,234.56"],
        "color": "#98D8C8",
    },
    "DATE": {
        "description": "Specific dates or relative time references",
        "examples": ["12/25/2024", "tomorrow", "today", "December 25", "in 3 days"],
        "color": "#F7DC6F",
    },
    "DEADLINE": {
        "description": "Time pressure or urgency phrases",
        "examples": ["within 24 hours", "by tonight", "immediately", "before midnight", "expires soon"],
        "color": "#FF6B9D",
    },
    "ACTION_REQUIRED": {
        "description": "Imperative action verbs or action phrases",
        "examples": ["click here", "verify now", "call immediately", "confirm", "update", "respond"],
        "color": "#C44569",
    },
}

ANNOTATION_RULES: List[str] = [
    "Annotate minimal span only",
    "Exclude punctuation unless part of entity",
    "Verify entity externality",
    "Annotate nested entities separately",
    "Distinguish URLs from brand references",
    "Require currency context for AMOUNT",
    "DEADLINE requires urgency; DATE is neutral",
]

TAG_SCHEME = "IOB2"

def get_tag_list() -> List[str]:
    tags = ["O"]
    for entity_type in ENTITY_TYPES.keys():
        tags.append(f"B-{entity_type}")
        tags.append(f"I-{entity_type}")
    return tags

def get_label_studio_labels() -> List[Dict[str, str]]:
    labels = []
    for entity_type, info in ENTITY_TYPES.items():
        labels.append({
            "value": entity_type,
            "background": info["color"],
        })
    return labels

def validate_entity_annotation(text: str, entity_type: str, span_text: str) -> bool:
    if entity_type not in ENTITY_TYPES:
        return False
    
    if span_text != span_text.strip():
        return False
    
    if entity_type == "PHONE":
        if not any(c.isdigit() for c in span_text):
            return False
    
    elif entity_type == "URL":
        if "." not in span_text and "/" not in span_text:
            return False
    
    elif entity_type == "AMOUNT":
        if not any(c.isdigit() for c in span_text):
            return False
    
    return True

if __name__ == "__main__":
    print("Entity Schema for SMS Phishing Detection")
    print("=" * 60)
    print(f"\nTotal Entity Types: {len(ENTITY_TYPES)}")
    print(f"\nBIO Tags ({len(get_tag_list())}): {get_tag_list()}\n")
    
    for entity_type, info in ENTITY_TYPES.items():
        print(f"\n{entity_type}")
        print("-" * 40)
        print(f"Description: {info['description']}")
        print(f"Examples: {', '.join(info['examples'][:3])}")

# Annotation Guidelines for SMS Phishing Detection

Version 1.0
Date: October 25, 2025

## 1. Introduction

This document provides standardized guidelines for annotating SMS messages with token-level entity spans to support Named Entity Recognition (NER) model training for phishing detection.

### 1.1 Annotation Objectives

The annotation process serves to create training data for models that:
- Extract factual claims from SMS messages
- Identify entity patterns indicative of phishing attempts
- Enable automated claim verification against external knowledge sources
- Improve detection of novel phishing attacks

### 1.2 Dataset Composition

Primary dataset: 510 smishing messages
Control dataset: 100 legitimate messages
Supplementary dataset: 50 spam messages

Estimated annotation time: 6 minutes per message

### 1.3 Annotation Interface

Label Studio web interface provides token-level annotation capabilities with keyboard shortcuts for efficient labeling.

## 2. Entity Schema

The annotation schema consists of eight entity types, each representing verifiable information components within SMS messages.

### 2.1 BRAND

Definition: Company, organization, or service provider names

Valid examples:
- Your [BRAND: Amazon] package is ready
- [BRAND: PayPal] account suspended
- [BRAND: IRS] tax refund available
- Message from [BRAND: USPS] regarding delivery

Invalid examples:
- Generic references: "your bank", "the company"
- Descriptive phrases: "online retailer", "shipping company"

Special cases:
- Misspelled brand names should be annotated if recognizable
- Separate brand names from associated URLs

### 2.2 PHONE

Definition: Telephone numbers in any standard format

Valid examples:
- Call [PHONE: 1-800-123-4567] now
- Text [PHONE: (555) 123-4567] to confirm
- Contact us: [PHONE: 18001234567]
- Reply to [PHONE: +1-555-123-4567]

Annotation rules:
- Include formatting characters (dashes, parentheses, dots, spaces)
- Exclude trailing punctuation not part of the number

### 2.3 URL

Definition: Web addresses including full URLs, shortened links, and domain names

Valid examples:
- Visit [URL: http://amazon.com/verify]
- Click [URL: https://bit.ly/abc123]
- Go to [URL: amzn.to/xyz]
- [URL: suspicious-link.ru/account]

Disambiguation from BRAND:
- URL: Used when directing user to visit a link
- BRAND: Used when referencing company name

### 2.4 ORDER_ID

Definition: Order numbers, tracking identifiers, invoice numbers, confirmation codes

Valid examples:
- Order [ORDER_ID: #12345] is ready
- Tracking: [ORDER_ID: 1Z999AA10123456784]
- Invoice [ORDER_ID: INV-2024-001]
- Confirmation code: [ORDER_ID: ABC123]

Include alphanumeric codes and standard prefixes.

### 2.5 AMOUNT

Definition: Monetary values with or without currency symbols

Valid examples:
- Refund of [AMOUNT: $50.00]
- Pay [AMOUNT: 50 USD]
- Total: [AMOUNT: €100]
- [AMOUNT: $1,234.56] has been charged

Annotation rules:
- Include currency symbols and codes
- Exclude non-monetary numeric values

### 2.6 DATE

Definition: Temporal references without urgency connotation

Valid examples:
- Delivery on [DATE: 12/25/2024]
- Shipped [DATE: tomorrow]
- Arriving [DATE: today]
- Expected [DATE: in 3 days]

Distinction from DEADLINE: DATE entities represent neutral temporal information.

### 2.7 DEADLINE

Definition: Time references with urgency or pressure connotation

Valid examples:
- Respond [DEADLINE: within 24 hours]
- Act [DEADLINE: by tonight]
- Click [DEADLINE: immediately]
- Expires [DEADLINE: before midnight]

Key indicators: "immediately", "urgent", "now", "ASAP", "within X hours"

### 2.8 ACTION_REQUIRED

Definition: Imperative action verbs or directive phrases

Valid examples:
- [ACTION_REQUIRED: Click here] to verify
- [ACTION_REQUIRED: Call] us immediately
- [ACTION_REQUIRED: Verify now]
- [ACTION_REQUIRED: Confirm] your account

Exclusions:
- Past tense verbs
- Passive voice constructions
- Neutral informational statements

## 3. Core Annotation Principles

### 3.1 Minimal Span Principle

Annotate only the entity itself, excluding surrounding context.

Correct: Your [BRAND: Amazon] package
Incorrect: [BRAND: Your Amazon package]

### 3.2 Punctuation Handling

Exclude trailing punctuation unless integral to the entity.

Correct: Visit [URL: example.com].
Incorrect: Visit [URL: example.com.]

Exception: Punctuation within entity boundaries (e.g., phone number formatting)

### 3.3 Verification Criterion

Apply the test: "Can this entity be verified against external sources?"
If yes, annotate. If no, skip.

### 3.4 Nested Entities

When entities overlap semantically, annotate each separately.

Example: [BRAND: Amazon] order [ORDER_ID: #12345]

### 3.5 Context-Dependent Classification

The same text string may receive different labels based on usage context.

Visit [URL: amazon.com] (directive to access link)
Your [BRAND: Amazon] account (reference to company)

## 4. Annotation Examples

### 4.1 Phishing Message Example

Message text:
"Your Amazon account has been suspended. Verify now at http://amzn-verify.tk or call 1-800-FAKE-NUM within 24 hours."

Annotations:
- [BRAND: Amazon]
- [ACTION_REQUIRED: Verify]
- [URL: http://amzn-verify.tk]
- [ACTION_REQUIRED: call]
- [PHONE: 1-800-FAKE-NUM]
- [DEADLINE: within 24 hours]

### 4.2 Legitimate Message Example

Message text:
"Your Amazon order #112-3456789 has shipped and will arrive Dec 25. Track it at amazon.com/orders"

Annotations:
- [BRAND: Amazon]
- [ORDER_ID: #112-3456789]
- [DATE: Dec 25]
- [URL: amazon.com/orders]

Note: Absence of ACTION_REQUIRED and DEADLINE entities indicates neutral tone.

## 5. Edge Cases and Special Situations

### 5.1 Misspelled Brand Names

Annotate recognizable misspellings as BRAND entities.

Examples: "Amaz0n", "PayPaI" (capital I for lowercase l)

### 5.2 Generic versus Specific References

Specific, verifiable entities should be annotated; generic references should not.

Annotate: [BRAND: Bank of America]
Skip: "your bank"

### 5.3 Ambiguous Numeric Values

Context determines classification.

Pay [AMOUNT: $50] (monetary context)
Order [ORDER_ID: #50] (identifier context)

### 5.4 URL Fragments

Annotate partial URLs and domain names when used directively.

Visit [URL: amazon.com]
Click [URL: bit.ly/abc]

### 5.5 Compound Action Phrases

Annotate the minimal imperative component.

[ACTION_REQUIRED: Click here] to verify now
Alternative: [ACTION_REQUIRED: Click] here to verify now

## 6. Quality Assurance

### 6.1 Self-Review Checklist

Per message review:
- All brand names identified
- All contact information (phone, URL) marked
- Transaction identifiers annotated
- Monetary values tagged
- Temporal references classified correctly
- Imperative verbs marked
- No annotation overlaps
- Minimal span principle applied
- Punctuation handled correctly

### 6.2 Periodic Quality Checks

Every 100 messages:
- Export annotations
- Run quality_check.py script
- Review identified issues
- Adjust annotation strategy if needed

### 6.3 Common Errors

Frequent mistakes to avoid:
- Including trailing punctuation in entity spans
- Annotating generic instead of specific references
- Confusing DATE and DEADLINE entities
- Exceeding minimal span boundaries
- Omitting nested entities

## 7. Annotation Workflow

Standard procedure:
1. Read complete message
2. Identify all entity candidates
3. Apply annotation schema systematically
4. Verify minimal span compliance
5. Check for nested entities
6. Add notes for ambiguous cases
7. Submit annotation

## 8. Technical Specifications

### 8.1 Output Format

Annotations are exported to:
- CoNLL format for BERT-based NER training
- JSON format for custom model architectures

### 8.2 BIO Tagging Scheme

The system uses IOB2 (Inside-Outside-Beginning) tagging:
- B-ENTITY: Beginning of entity
- I-ENTITY: Inside entity (continuation)
- O: Outside any entity

### 8.3 Inter-Annotator Agreement

For quality control, a subset of messages may be multiply annotated to calculate agreement metrics.

## 9. Dataset Split Strategy

Annotated messages will be divided:
- Training set: 67%
- Validation set: 17%
- Test set: 16%

This split maintains class distribution across phishing, legitimate, and spam categories.

## 10. References

Dataset source: Mendeley SMS Dataset
Annotation tool: Label Studio
Entity schema: config/entity_schema.py
Quality validation: scripts/quality_check.py

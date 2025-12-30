from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SlotSchema:
    name: str
    description: str
    value_type: str = "string"


CLAIM_TYPE_SCHEMAS: dict[str, list[SlotSchema]] = {
    "ACTION_CLAIM": [
        SlotSchema("action_type", "Type of action requested (call, click, visit, reply, download, etc.)", "string"),
        SlotSchema("phone_number", "Phone number to call if specified", "string"),
        SlotSchema("url", "URL to visit if specified", "string"),
        SlotSchema("purpose", "Stated purpose or reason for the action", "string"),
    ],
    "URGENCY_CLAIM": [
        SlotSchema("time_frame", "Explicit time constraint (24 hours, immediately, today, etc.)", "string"),
        SlotSchema("urgency_level", "Level of urgency (low, medium, high, critical)", "string"),
        SlotSchema("consequence", "What happens if action not taken", "string"),
    ],
    "REWARD_CLAIM": [
        SlotSchema("reward_type", "Type of reward (cashback, prize, discount, points, gift, etc.)", "string"),
        SlotSchema("amount", "Monetary amount if specified", "float"),
        SlotSchema("currency", "Currency symbol or code", "string"),
        SlotSchema("sponsor", "Entity offering the reward", "string"),
        SlotSchema("condition", "Condition to receive reward", "string"),
    ],
    "FINANCIAL_CLAIM": [
        SlotSchema("transaction_type", "Type of transaction (payment, refund, charge, withdrawal, etc.)", "string"),
        SlotSchema("amount", "Monetary amount", "float"),
        SlotSchema("currency", "Currency symbol or code", "string"),
        SlotSchema("account_type", "Type of account (bank, credit card, payment app, etc.)", "string"),
        SlotSchema("status", "Status of transaction (pending, completed, failed, etc.)", "string"),
    ],
    "ACCOUNT_CLAIM": [
        SlotSchema("account_type", "Type of account (banking, email, social media, shopping, etc.)", "string"),
        SlotSchema("account_name", "Name or brand of the account", "string"),
        SlotSchema("status", "Account status (suspended, locked, compromised, expired, etc.)", "string"),
        SlotSchema("reason", "Reason for status change", "string"),
    ],
    "DELIVERY_CLAIM": [
        SlotSchema("service_type", "Type of delivery (package, flight, shipment, mail, etc.)", "string"),
        SlotSchema("carrier", "Delivery carrier or service (USPS, FedEx, airline, etc.)", "string"),
        SlotSchema("tracking_id", "Tracking or reference number", "string"),
        SlotSchema("flight_number", "Flight number if applicable", "string"),
        SlotSchema("origin_city", "Origin location", "string"),
        SlotSchema("destination_city", "Destination location", "string"),
        SlotSchema("date", "Delivery or travel date", "string"),
        SlotSchema("status", "Delivery status (delayed, failed, arrived, cancelled, etc.)", "string"),
    ],
    "VERIFICATION_CLAIM": [
        SlotSchema("verification_type", "What needs verification (identity, account, email, phone, etc.)", "string"),
        SlotSchema("verification_method", "How to verify (code, link, call, document, etc.)", "string"),
        SlotSchema("entity", "Who is requesting verification", "string"),
        SlotSchema("reason", "Reason verification is needed", "string"),
    ],
    "IDENTITY_CLAIM": [
        SlotSchema("brand", "Brand or organization name", "string"),
        SlotSchema("claim_type", "Type of identity claim (from, on behalf of, representing, etc.)", "string"),
        SlotSchema("authenticity_indicators", "Any provided authentication details", "string"),
    ],
    "SOCIAL_CLAIM": [
        SlotSchema("social_context", "Social engineering tactic used", "string"),
        SlotSchema("relationship", "Claimed relationship (friend, colleague, family, etc.)", "string"),
        SlotSchema("request", "What is being requested", "string"),
    ],
    "LEGAL_CLAIM": [
        SlotSchema("legal_action", "Type of legal action (lawsuit, warrant, fine, suspension, etc.)", "string"),
        SlotSchema("authority", "Legal authority involved (court, police, IRS, etc.)", "string"),
        SlotSchema("violation", "Alleged violation or issue", "string"),
        SlotSchema("consequence", "Consequence of non-compliance", "string"),
    ],
    "SECURITY_CLAIM": [
        SlotSchema("security_issue", "Type of security issue (breach, unauthorized access, suspicious activity, etc.)", "string"),
        SlotSchema("affected_system", "What system is affected", "string"),
        SlotSchema("severity", "Severity level", "string"),
        SlotSchema("action_required", "What action is needed", "string"),
    ],
    "CREDENTIALS_CLAIM": [
        SlotSchema("credential_type", "Type of credential (password, PIN, SSN, account number, etc.)", "string"),
        SlotSchema("status", "Status of credential (expired, compromised, needs update, etc.)", "string"),
        SlotSchema("request", "What is being requested", "string"),
    ],
    "OTHER_CLAIM": [
        SlotSchema("claim_description", "Description of the claim", "string"),
        SlotSchema("context", "Additional context", "string"),
    ],
}


def get_slot_schema(claim_type: str) -> list[SlotSchema]:
    return CLAIM_TYPE_SCHEMAS.get(claim_type, CLAIM_TYPE_SCHEMAS["OTHER_CLAIM"])


def get_slot_names(claim_type: str) -> list[str]:
    return [slot.name for slot in get_slot_schema(claim_type)]


def format_schema_for_prompt(claim_type: str) -> str:
    slots = get_slot_schema(claim_type)
    lines = [f"Slots for {claim_type}:"]
    for slot in slots:
        lines.append(f"  - {slot.name}: {slot.description}")
    return "\n".join(lines)


def validate_parsed_claim(claim_type: str, slots: dict[str, any]) -> tuple[bool, list[str]]:
    valid_slots = set(get_slot_names(claim_type))
    provided_slots = set(slots.keys())
    
    unknown_slots = provided_slots - valid_slots
    
    if unknown_slots:
        errors = [f"Unknown slots for {claim_type}: {', '.join(unknown_slots)}"]
        return False, errors
    
    return True, []

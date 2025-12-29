#!/usr/bin/env python3
"""
Apply manual review decisions from terminal output to JSON
"""

import json
from pathlib import Path

# Parse the review log
review_log = """
ACCEPT: REWARD_CLAIM - 'your exclusive offer is ready for redemption'
EDITED: URGENCY_CLAIM - 'within 24 hours'
ACCEPT: URGENCY_CLAIM - 'within 24 hours'
ACCEPT: ACTION_CLAIM - 'to claim your reward'
ACCEPT: REWARD_CLAIM - 'ur awarded either a yrs supply of CDs from Virgin '
ACCEPT: ACTION_CLAIM - 'Call 09061104283'
ACCEPT: REWARD_CLAIM - 'Congratulations! You've been chosen to receive a £'
ACCEPT: ACTION_CLAIM - 'Call us now at 08001234567 to claim your prize!'
ACCEPT: REWARD_CLAIM - 'Congratulations! Your exclusive discount code RP17'
ACCEPT: URGENCY_CLAIM - 'Claim it now to enjoy your savings.'
ACCEPT: ACTION_CLAIM - 'To opt-out, reply STOP.'
ACCEPT: ACTION_CLAIM - 'Visit www.regalportfolio.co.uk'
ACCEPT: ACTION_CLAIM - 'call Customer Services at 0871 720 5546.'
ACCEPT: REWARD_CLAIM - 'Congratulations! You have been selected to receive'
ACCEPT: URGENCY_CLAIM - 'Act fast!'
ACCEPT: ACTION_CLAIM - 'Please call 09061221067 from a landline to confirm'
ACCEPT: URGENCY_CLAIM - 'If you didn't make this order, click here immediat'
ACCEPT: ACTION_CLAIM - 'click here immediately: http://192.168.1.10:8080/v'
ACCEPT: REWARD_CLAIM - 'Congratulations! You've received 500 complimentary'
ACCEPT: ACTION_CLAIM - 'To activate, please call us at 0800-123-4567 now!'
ACCEPT: URGENCY_CLAIM - 'Don't miss out!'
ACCEPT: REWARD_CLAIM - 'Congratulations! You've been selected to receive a'
ACCEPT: URGENCY_CLAIM - 'Please call us at 09061221067 from a landline with'
ACCEPT: ACTION_CLAIM - 'to confirm your delivery.'
ACCEPT: ACTION_CLAIM - 'Please call our customer service representative on'
ACCEPT: ACTION_CLAIM - 'Call 08715203652'
ACCEPT: URGENCY_CLAIM - 'Expires 29/10/0'
ACCEPT: ACTION_CLAIM - 'To access your refund, follow the steps required.'
REJECT: URGENCY_CLAIM - 'http:/bit.do/Claim-Tax'
ACCEPT: ACTION_CLAIM - 'please complete your kyc'
ACCEPT: ACTION_CLAIM - 'Call 08715203677'
ACCEPT: URGENCY_CLAIM - 'Expires 24/10/04'
ACCEPT: ACTION_CLAIM - 'please reply STOP to 87239'
ACCEPT: URGENCY_CLAIM - 'contact customer support at 0870 803 4412 immediat'
ACCEPT: ACTION_CLAIM - 'View your image now at https://picsgallery.com/vie'
ACCEPT: URGENCY_CLAIM - 'Don't miss it!'
ACCEPT: URGENCY_CLAIM - 'to regain access immediately!'
REJECT: URGENCY_CLAIM - 'Am free all next week'
ACCEPT: ACTION_CLAIM - 'Chat now 2 sort time'
ACCEPT: REWARD_CLAIM - 'Unlock your FREE 3-month streaming subscription!'
ACCEPT: ACTION_CLAIM - 'Reply NOW to 12345 to claim your access!'
ACCEPT: URGENCY_CLAIM - 'Limited time only!'
ACCEPT: REWARD_CLAIM - 'Congratulations! You've been randomly selected to '
ACCEPT: ACTION_CLAIM - 'Call 08001234567 to claim now!'
ACCEPT: ACTION_CLAIM - 'Click:http://23.254.215.52 to fill out your Applic'
REJECT: URGENCY_CLAIM - 'due to the CoronaVirus pandemic'
ACCEPT: ACTION_CLAIM - 'To unlock please update security details'
REJECT: URGENCY_CLAIM - 'Should your tone not arrive please call customer s'
ACCEPT: ACTION_CLAIM - 'please call customer services 09065989180'
ACCEPT: URGENCY_CLAIM - 'URGENT'
ACCEPT: REWARD_CLAIM - 'Today is your lucky day!'
ACCEPT: ACTION_CLAIM - 'log onto http://www.urawinner.com'
ACCEPT: REWARD_CLAIM - 'there is a fantastic surprise awaiting you'
REJECT: REWARD_CLAIM - 'Holiday for 25 years.'
ACCEPT: ACTION_CLAIM - 'Call 08715203694'
ACCEPT: URGENCY_CLAIM - 'Expires 31/10/04'
ACCEPT: URGENCY_CLAIM - 'URGENT!'
ACCEPT: REWARD_CLAIM - 'WON a £2000 Bonus Caller prize'
ACCEPT: URGENCY_CLAIM - 'This is the 2nd attempt to reach you!'
ACCEPT: ACTION_CLAIM - 'Call 09066368753 ASAP!'
REJECT: REWARD_CLAIM - 'Congrats!'
ACCEPT: ACTION_CLAIM - 'Call 09066382422'
ACCEPT: REWARD_CLAIM - 'Congratulations! You've won a FREE high-definition'
ACCEPT: ACTION_CLAIM - 'To claim your prize, simply text CLICK to 12345.'
ACCEPT: REWARD_CLAIM - 'you have W0N a guaranteed cash prize of Rs.5,00,00'
ACCEPT: ACTION_CLAIM - 'Please call our customer service representative on'
ACCEPT: URGENCY_CLAIM - 'This exclusive offer expires soon'
ACCEPT: ACTION_CLAIM - 'call MobilesDirect at 0800 093 8766 now'
ACCEPT: REWARD_CLAIM - 'Buy any HP pavilion notebook & get a Kodak camera'
ACCEPT: REWARD_CLAIM - 'Claim a 200 shopping spree'
ACCEPT: URGENCY_CLAIM - 'just call 08717895698 now!'
CHANGED TYPE: REWARD_CLAIM - 'Have you won!'
ACCEPT: REWARD_CLAIM - 'Have you won!'
ACCEPT: ACTION_CLAIM - 'Claim a 200 shopping spree, just call 08717895698 '
ACCEPT: REWARD_CLAIM - 'Free Msg: Ringtone!'
ACCEPT: REWARD_CLAIM - 'Congratulations! You've been selected for an exclu'
ACCEPT: ACTION_CLAIM - 'Call us at 08081234567 and use code WIN123.'
ACCEPT: URGENCY_CLAIM - 'Act fast, offer ends in 12 hours!'
ACCEPT: URGENCY_CLAIM - 'Please verify your payment info immediately'
ACCEPT: URGENCY_CLAIM - 'risk suspension'
ACCEPT: ACTION_CLAIM - 'verify your payment info'
ACCEPT: REWARD_CLAIM - 'todays vodafone numbers ending with 0089(my last f'
ACCEPT: ACTION_CLAIM - 'If your number matches please call 09063442151 to '
ACCEPT: REWARD_CLAIM - 'Lucky draw scheme'
REJECT: REWARD_CLAIM - 'every month lucky member will be given Rs3600 gold'
ACCEPT: ACTION_CLAIM - 'Call-9008097492'
ACCEPT: REWARD_CLAIM - 'Congratulations! You've been selected for a specia'
ACCEPT: URGENCY_CLAIM - 'Claim your reward now!'
ACCEPT: ACTION_CLAIM - 'Visit: http://bit.ly/AmazonReward2025'
ACCEPT: REWARD_CLAIM - 'Your 4* Costa Del Sol Holiday or £5000 await colle'
ACCEPT: ACTION_CLAIM - 'Call 09050090044 Now toClaim'
ACCEPT: URGENCY_CLAIM - 'URGENT!'
ACCEPT: REWARD_CLAIM - 'Your Mobile number has been awarded with a £2000 p'
ACCEPT: ACTION_CLAIM - 'Call 09061790321 from land line.'
ACCEPT: ACTION_CLAIM - 'Claim 3030.'
ACCEPT: URGENCY_CLAIM - 'Valid 12hrs only'
ACCEPT: REWARD_CLAIM - 'Congratulations! Your Netflix account has been sel'
ACCEPT: ACTION_CLAIM - 'Click here to verify your details: bit.ly/netflix-'
ACCEPT: URGENCY_CLAIM - 'Act fast!'
ACCEPT: URGENCY_CLAIM - 'If this transaction was not authorized, please ver'
ACCEPT: ACTION_CLAIM - 'please verify immediately'
ACCEPT: REWARD_CLAIM - 'Congratulations! You've been selected for a cash r'
ACCEPT: URGENCY_CLAIM - 'Call us now at 0800 123 4567 by 10pm to claim your'
ACCEPT: ACTION_CLAIM - 'Call us now at 0800 123 4567'
ACCEPT: URGENCY_CLAIM - 'Urgent!'
ACCEPT: REWARD_CLAIM - 'Your ABTA complimentary 4* Tenerife Holiday or £50'
ACCEPT: ACTION_CLAIM - 'Please call 09061743811 from landline'
ACCEPT: REWARD_CLAIM - 'Congratulations! You've won an all-expenses-paid t'
ACCEPT: ACTION_CLAIM - 'Call 0800-123-4567 now to claim your prize.'
ACCEPT: REWARD_CLAIM - 'you are awarded with a £2000 Bonus Prize'
ACCEPT: ACTION_CLAIM - 'call 09066364529'
ACCEPT: REWARD_CLAIM - 'Win over £1000 in our quiz'
ACCEPT: REWARD_CLAIM - 'take your partner on the trip of a lifetime'
ACCEPT: ACTION_CLAIM - 'Send GO to 83600 now'
REJECT: ACTION_CLAIM - 'We have received a request from you for purchasing'
REJECT: URGENCY_CLAIM - 'You have 1 new message.'
ACCEPT: ACTION_CLAIM - 'Call 0207-083-6089'
ACCEPT: URGENCY_CLAIM - 'URGENT!'
ACCEPT: REWARD_CLAIM - 'you have won a £800 prize GUARANTEED'
ACCEPT: ACTION_CLAIM - 'Call 09050003092 from land line'
ACCEPT: URGENCY_CLAIM - 'Valid 12hrs only'
ACCEPT: ACTION_CLAIM - 'Log in via the secure link'
ACCEPT: URGENCY_CLAIM - 'to avoid account suspension'
ACCEPT: ACTION_CLAIM - 'Call 08719899230'
ACCEPT: URGENCY_CLAIM - 'Expires 07/11/04'
ACCEPT: REWARD_CLAIM - 'you have W0N a guaranteed cash prize of Rs.5,00,00'
ACCEPT: ACTION_CLAIM - 'Please call our customer service representative on'
ACCEPT: REWARD_CLAIM - 'Congratulations! As a valued customer of TravelLux'
ACCEPT: ACTION_CLAIM - 'Call 0800-123-4567 now and mention code LOUNGE2 to'
ACCEPT: URGENCY_CLAIM - 'now'
ACCEPT: URGENCY_CLAIM - 'Confirm your identity now to avoid account suspens'
ACCEPT: ACTION_CLAIM - 'Click here: https://secure-amazon-verification.com'
ACCEPT: ACTION_CLAIM - 'please complete your kyc'
ACCEPT: URGENCY_CLAIM - 'contact customer care 8536074310'
ACCEPT: URGENCY_CLAIM - 'please contact us urgently'
ACCEPT: REWARD_CLAIM - 'You have been selected for a complimentary 4* luxu'
REJECT: REWARD_CLAIM - 'or £1000 cash'
ACCEPT: ACTION_CLAIM - 'please contact us urgently at 09063440451 from a l'
REJECT: URGENCY_CLAIM - 'Today could change everything for you!'
ACCEPT: REWARD_CLAIM - 'to claim your exclusive prize'
ACCEPT: URGENCY_CLAIM - 'Act fast!'
ACCEPT: ACTION_CLAIM - 'Visit http://www'
ACCEPT: URGENCY_CLAIM - 'contact our delivery team at 07046744435 immediate'
ACCEPT: ACTION_CLAIM - 'to confirm arrangements'
ACCEPT: URGENCY_CLAIM - 'Reply of call 08000930705 Now'
ACCEPT: URGENCY_CLAIM - 'Now'
ACCEPT: ACTION_CLAIM - 'call you re your reply to our sms'
REJECT: URGENCY_CLAIM - 'You have an important customer service announcemen'
ACCEPT: ACTION_CLAIM - 'Call FREEPHONE 0800 542 0826 now!'
ACCEPT: URGENCY_CLAIM - 'Reactivate now to avoid interruption.'
ACCEPT: ACTION_CLAIM - 'Click here: http://bit.ly/reactivateNetflix ASAP!'
ACCEPT: URGENCY_CLAIM - 'URGENT!'
ACCEPT: REWARD_CLAIM - 'Last weekends draw shows that you w0n a Rs.2,00,00'
ACCEPT: ACTION_CLAIM - 'Call 6299257179'
ACCEPT: URGENCY_CLAIM - 'Valid 12hrs only'
ACCEPT: ACTION_CLAIM - 'to claim just call 09050002312'
ACCEPT: URGENCY_CLAIM - 'Cant guess who?'
ACCEPT: ACTION_CLAIM - 'CALL 09058095107 NOW all will be revealed.'
ACCEPT: ACTION_CLAIM - 'To claim, reply YES now.'
ACCEPT: URGENCY_CLAIM - 'ALERT!'
ACCEPT: REWARD_CLAIM - 'Congratulations! Your Netflix account has been ran'
ACCEPT: ACTION_CLAIM - 'To claim, please visit: netflix-upgrade2024.com an'
ACCEPT: URGENCY_CLAIM - 'Final Chance!'
ACCEPT: REWARD_CLAIM - 'Claim ur £150 worth of discount vouchers today!'
ACCEPT: ACTION_CLAIM - 'Text YES to 85023 now!'
REJECT: URGENCY_CLAIM - 'find out who they R'
ACCEPT: ACTION_CLAIM - 'call on 09058094565'
ACCEPT: URGENCY_CLAIM - 'Urgent'
ACCEPT: REWARD_CLAIM - 'T&Cs SAE award'
ACCEPT: ACTION_CLAIM - 'Please call 09066612661 from landline'
ACCEPT: URGENCY_CLAIM - 'Click here: bit.ly/NetflixSecure NOW!'
ACCEPT: ACTION_CLAIM - 'Click here: bit'
"""

# Parse decisions
decisions = []
edits = {}
type_changes = {}

for line in review_log.strip().split('\n'):
    if not line.strip():
        continue
    
    parts = line.split(' - ', 1)
    if len(parts) != 2:
        continue
    
    action_type = parts[0].split(': ')
    if len(action_type) != 2:
        continue
    
    action = action_type[0].strip()
    claim_type = action_type[1].strip()
    claim_text = parts[1].strip().strip("'")
    
    decisions.append({
        'action': action,
        'type': claim_type,
        'text': claim_text
    })

print("="*70)
print("APPLYING MANUAL REVIEW DECISIONS")
print("="*70)
print(f"\nParsed {len(decisions)} decisions from log")

# Load data
input_file = Path('data/annotations/claim_annotations_2000_clean.json')
with open(input_file, 'r') as f:
    data = json.load(f)

# Apply decisions in order
current_decision_idx = 0
removed_count = 0
edited_count = 0
type_changed_count = 0

# Track which claim we're at globally
global_claim_idx = 0

for entry_idx, entry in enumerate(data):
    if not entry.get('annotations') or not entry['annotations']:
        continue
    
    annotations = entry['annotations'][0]
    if 'result' not in annotations or not annotations['result']:
        continue
    
    # Filter relevant claims (URGENCY, ACTION, REWARD)
    target_labels = ['URGENCY_CLAIM', 'ACTION_CLAIM', 'REWARD_CLAIM']
    
    new_results = []
    for result_idx, result in enumerate(annotations['result']):
        value = result.get('value', {})
        labels = value.get('labels', [])
        
        if not labels or labels[0] not in target_labels:
            # Keep non-target claims as-is
            new_results.append(result)
            continue
        
        claim_text = value.get('text', '')
        claim_type = labels[0]
        
        # Match with decision
        if current_decision_idx < len(decisions):
            decision = decisions[current_decision_idx]
            
            # Check if this claim matches the decision (by text prefix)
            if claim_text.startswith(decision['text']) or decision['text'].startswith(claim_text[:min(len(claim_text), 40)]):
                
                if decision['action'] == 'REJECT':
                    removed_count += 1
                    print(f"  Removed: {claim_type} - '{claim_text[:50]}'")
                    current_decision_idx += 1
                    continue  # Skip adding this result
                
                elif decision['action'] == 'EDITED':
                    # Next decision should be the edit
                    result['value']['text'] = decision['text']
                    edited_count += 1
                    print(f"  Edited: '{claim_text[:30]}' -> '{decision['text'][:30]}'")
                    current_decision_idx += 1
                
                elif decision['action'] == 'CHANGED TYPE':
                    # Change type
                    old_type = labels[0]
                    new_type = decision['type']
                    result['value']['labels'] = [new_type]
                    type_changed_count += 1
                    print(f"  Type changed: {old_type} -> {new_type}")
                    current_decision_idx += 1
                
                else:  # ACCEPT
                    current_decision_idx += 1
                
                new_results.append(result)
            else:
                # Doesn't match, keep as-is
                new_results.append(result)
        else:
            # No more decisions, keep remaining
            new_results.append(result)
    
    annotations['result'] = new_results
    annotations['result_count'] = len(new_results)

# Save
output_file = Path('data/annotations/claim_annotations_2000_reviewed.json')
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Removed: {removed_count}")
print(f"Edited: {edited_count}")
print(f"Type changed: {type_changed_count}")
print(f"Decisions applied: {current_decision_idx}/{len(decisions)}")
print(f"\nSaved to: {output_file}")
print(f"{'='*70}")

#!/usr/bin/env python3
"""
Simple UI to review IDENTITY_CLAIM annotations in claim_annotations_2000.json
Controls:
- LEFT ARROW or A: Reject (remove this claim)
- RIGHT ARROW or D: Accept (keep this claim)
- E: Edit the claim text
- ESC: Save and exit
"""

import pygame
import json
import sys
from pathlib import Path

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
BG_COLOR = (240, 240, 245)
CARD_COLOR = (255, 255, 255)
TEXT_COLOR = (40, 40, 40)
ACCENT_COLOR = (70, 130, 180)
REJECT_COLOR = (220, 60, 60)
ACCEPT_COLOR = (60, 180, 80)
EDIT_COLOR = (255, 165, 0)
SHADOW_COLOR = (200, 200, 200)

# Fonts
FONT_LARGE = pygame.font.Font(None, 42)
FONT_MEDIUM = pygame.font.Font(None, 32)
FONT_SMALL = pygame.font.Font(None, 26)
FONT_TINY = pygame.font.Font(None, 22)

class IdentityClaimReviewer:
    def __init__(self, json_path):
        self.json_path = Path(json_path)
        self.backup_path = self.json_path.with_suffix('.backup.json')
        
        # Load data
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Extract identity claims
        self.claims_to_review = []
        self.extract_identity_claims()
        
        self.current_index = 0
        self.decisions = {}  # {(entry_id, annotation_idx, result_idx): {'action': 'accept'|'reject'|'edit', 'new_text': str}}
        
        # Edit mode
        self.edit_mode = False
        self.edit_text = ""
        self.cursor_pos = 0
        
        # Setup display
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Identity Claim Reviewer")
        self.clock = pygame.time.Clock()
        
    def extract_identity_claims(self):
        """Extract all IDENTITY_CLAIM annotations"""
        for entry_idx, entry in enumerate(self.data):
            entry_id = entry.get('id', f'entry_{entry_idx}')
            text = entry.get('data', {}).get('text', '')
            
            if 'annotations' in entry and len(entry['annotations']) > 0:
                annotations = entry['annotations'][0]
                if 'result' in annotations:
                    for result_idx, result in enumerate(annotations['result']):
                        labels = result.get('value', {}).get('labels', [])
                        if 'IDENTITY_CLAIM' in labels:
                            claim_text = result.get('value', {}).get('text', '')
                            self.claims_to_review.append({
                                'entry_idx': entry_idx,
                                'entry_id': entry_id,
                                'annotation_idx': 0,
                                'result_idx': result_idx,
                                'full_text': text,
                                'claim_text': claim_text,
                                'start': result.get('value', {}).get('start', 0),
                                'end': result.get('value', {}).get('end', 0)
                            })
        
        print(f"Found {len(self.claims_to_review)} IDENTITY_CLAIM annotations to review")
    
    def draw_wrapped_text(self, surface, text, rect, font, color, center=False):
        """Draw text with word wrapping"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            test_surface = font.render(test_line, True, color)
            if test_surface.get_width() <= rect.width - 40:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        y_offset = rect.y + 20
        for line in lines:
            text_surface = font.render(line, True, color)
            if center:
                x = rect.x + (rect.width - text_surface.get_width()) // 2
            else:
                x = rect.x + 20
            surface.blit(text_surface, (x, y_offset))
            y_offset += font.get_height() + 5
    
    def draw_card(self):
        """Draw the current claim card"""
        self.screen.fill(BG_COLOR)
        
        if self.current_index >= len(self.claims_to_review):
            # All done!
            title = FONT_LARGE.render("All Claims Reviewed! 🎉", True, ACCENT_COLOR)
            self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 100))
            
            accepted = sum(1 for d in self.decisions.values() if d['action'] == 'accept')
            rejected = sum(1 for d in self.decisions.values() if d['action'] == 'reject')
            edited = sum(1 for d in self.decisions.values() if d['action'] == 'edit')
            
            stats_text = f"Accepted: {accepted} | Edited: {edited} | Rejected: {rejected}"
            stats = FONT_MEDIUM.render(stats_text, True, TEXT_COLOR)
            self.screen.blit(stats, (WINDOW_WIDTH // 2 - stats.get_width() // 2, 200))
            
            instruction = FONT_SMALL.render("Press ESC to save and exit", True, TEXT_COLOR)
            self.screen.blit(instruction, (WINDOW_WIDTH // 2 - instruction.get_width() // 2, 300))
            return
        
        claim = self.claims_to_review[self.current_index]
        
        # Progress
        progress_text = f"Claim {self.current_index + 1} / {len(self.claims_to_review)}"
        progress_surface = FONT_SMALL.render(progress_text, True, ACCENT_COLOR)
        self.screen.blit(progress_surface, (20, 20))
        
        # Progress bar
        progress_pct = (self.current_index / len(self.claims_to_review))
        bar_width = WINDOW_WIDTH - 40
        bar_height = 10
        pygame.draw.rect(self.screen, SHADOW_COLOR, (20, 60, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, ACCENT_COLOR, (20, 60, int(bar_width * progress_pct), bar_height), border_radius=5)
        
        # Main card - shadow
        card_rect = pygame.Rect(40, 110, WINDOW_WIDTH - 80, WINDOW_HEIGHT - 220)
        pygame.draw.rect(self.screen, SHADOW_COLOR, card_rect.move(5, 5), border_radius=15)
        
        # Main card - white background
        pygame.draw.rect(self.screen, CARD_COLOR, card_rect, border_radius=15)
        pygame.draw.rect(self.screen, ACCENT_COLOR, card_rect, width=3, border_radius=15)
        
        # Title
        title = FONT_MEDIUM.render("Identity Claim to Review:", True, ACCENT_COLOR)
        self.screen.blit(title, (60, 130))
        
        # Claim text (highlighted)
        claim_rect = pygame.Rect(60, 180, WINDOW_WIDTH - 120, 100)
        pygame.draw.rect(self.screen, (255, 248, 220), claim_rect, border_radius=8)
        pygame.draw.rect(self.screen, ACCENT_COLOR, claim_rect, width=2, border_radius=8)
        
        claim_label = FONT_TINY.render("IDENTITY_CLAIM:", True, ACCENT_COLOR)
        self.screen.blit(claim_label, (70, 190))
        
        claim_text_surface = FONT_SMALL.render(f'"{claim["claim_text"]}"', True, TEXT_COLOR)
        self.screen.blit(claim_text_surface, (70, 220))
        
        # Full message context
        context_label = FONT_TINY.render("Full Message Context:", True, TEXT_COLOR)
        self.screen.blit(context_label, (60, 310))
        
        context_rect = pygame.Rect(60, 340, WINDOW_WIDTH - 120, 180)
        self.draw_wrapped_text(self.screen, claim['full_text'], context_rect, FONT_SMALL, TEXT_COLOR)
        
        # Controls at bottom
        controls_y = WINDOW_HEIGHT - 90
        
        if self.edit_mode:
            # Edit mode - show text input
            edit_box_rect = pygame.Rect(60, controls_y - 50, WINDOW_WIDTH - 120, 40)
            pygame.draw.rect(self.screen, (255, 255, 255), edit_box_rect, border_radius=5)
            pygame.draw.rect(self.screen, EDIT_COLOR, edit_box_rect, width=3, border_radius=5)
            
            # Draw edit text with cursor
            edit_display = FONT_SMALL.render(self.edit_text, True, TEXT_COLOR)
            self.screen.blit(edit_display, (edit_box_rect.x + 10, edit_box_rect.y + 8))
            
            # Draw cursor
            cursor_x = edit_box_rect.x + 10 + FONT_SMALL.size(self.edit_text[:self.cursor_pos])[0]
            pygame.draw.line(self.screen, EDIT_COLOR, 
                           (cursor_x, edit_box_rect.y + 5), 
                           (cursor_x, edit_box_rect.y + 35), 2)
            
            # Instructions
            inst1 = FONT_TINY.render("Edit the claim text | ENTER to save | ESC to cancel", True, EDIT_COLOR)
            self.screen.blit(inst1, (60, controls_y - 80))
            
            # Save button (Enter)
            save_rect = pygame.Rect(250, controls_y, 220, 60)
            pygame.draw.rect(self.screen, ACCEPT_COLOR, save_rect, border_radius=10)
            save_text1 = FONT_MEDIUM.render("SAVE (ENTER)", True, (255, 255, 255))
            save_text2 = FONT_TINY.render("Accept edited text", True, (255, 255, 255))
            self.screen.blit(save_text1, (save_rect.centerx - save_text1.get_width() // 2, controls_y + 10))
            self.screen.blit(save_text2, (save_rect.centerx - save_text2.get_width() // 2, controls_y + 38))
            
            # Cancel button (ESC)
            cancel_rect = pygame.Rect(530, controls_y, 220, 60)
            pygame.draw.rect(self.screen, REJECT_COLOR, cancel_rect, border_radius=10)
            cancel_text1 = FONT_MEDIUM.render("CANCEL (ESC)", True, (255, 255, 255))
            cancel_text2 = FONT_TINY.render("Discard changes", True, (255, 255, 255))
            self.screen.blit(cancel_text1, (cancel_rect.centerx - cancel_text1.get_width() // 2, controls_y + 10))
            self.screen.blit(cancel_text2, (cancel_rect.centerx - cancel_text2.get_width() // 2, controls_y + 38))
        else:
            # Normal mode - show accept/reject/edit buttons
            # Reject button
            reject_rect = pygame.Rect(60, controls_y, 220, 60)
            pygame.draw.rect(self.screen, REJECT_COLOR, reject_rect, border_radius=10)
            reject_text1 = FONT_MEDIUM.render("← REJECT (A)", True, (255, 255, 255))
            reject_text2 = FONT_TINY.render("Remove claim", True, (255, 255, 255))
            self.screen.blit(reject_text1, (reject_rect.centerx - reject_text1.get_width() // 2, controls_y + 10))
            self.screen.blit(reject_text2, (reject_rect.centerx - reject_text2.get_width() // 2, controls_y + 38))
            
            # Edit button
            edit_rect = pygame.Rect(320, controls_y, 220, 60)
            pygame.draw.rect(self.screen, EDIT_COLOR, edit_rect, border_radius=10)
            edit_text1 = FONT_MEDIUM.render("EDIT (E)", True, (255, 255, 255))
            edit_text2 = FONT_TINY.render("Modify text", True, (255, 255, 255))
            self.screen.blit(edit_text1, (edit_rect.centerx - edit_text1.get_width() // 2, controls_y + 10))
            self.screen.blit(edit_text2, (edit_rect.centerx - edit_text2.get_width() // 2, controls_y + 38))
            
            # Accept button
            accept_rect = pygame.Rect(720, controls_y, 220, 60)
            pygame.draw.rect(self.screen, ACCEPT_COLOR, accept_rect, border_radius=10)
            accept_text1 = FONT_MEDIUM.render("ACCEPT (D) →", True, (255, 255, 255))
            accept_text2 = FONT_TINY.render("Keep as-is", True, (255, 255, 255))
            self.screen.blit(accept_text1, (accept_rect.centerx - accept_text1.get_width() // 2, controls_y + 10))
            self.screen.blit(accept_text2, (accept_rect.centerx - accept_text2.get_width() // 2, controls_y + 38))
    
    def enter_edit_mode(self):
        """Enter edit mode for current claim"""
        if self.current_index < len(self.claims_to_review):
            claim = self.claims_to_review[self.current_index]
            self.edit_mode = True
            self.edit_text = claim['claim_text']
            self.cursor_pos = len(self.edit_text)
    
    def save_edit(self):
        """Save edited text and move to next claim"""
        if self.edit_text.strip():
            claim = self.claims_to_review[self.current_index]
            key = (claim['entry_idx'], claim['annotation_idx'], claim['result_idx'])
            self.decisions[key] = {
                'action': 'edit',
                'new_text': self.edit_text.strip()
            }
            self.current_index += 1
        self.edit_mode = False
        self.edit_text = ""
        self.cursor_pos = 0
    
    def cancel_edit(self):
        """Cancel edit mode"""
        self.edit_mode = False
        self.edit_text = ""
        self.cursor_pos = 0
    
    def make_decision(self, decision):
        """Record decision for current claim"""
        if self.current_index < len(self.claims_to_review):
            claim = self.claims_to_review[self.current_index]
            key = (claim['entry_idx'], claim['annotation_idx'], claim['result_idx'])
            self.decisions[key] = {'action': decision}
            self.current_index += 1
    
    def save_changes(self):
        """Save the updated JSON with rejected claims removed"""
        # Create backup
        with open(self.backup_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"Backup saved to: {self.backup_path}")
        
        # Apply decisions in reverse order to maintain indices
        for (entry_idx, annotation_idx, result_idx), decision in sorted(self.decisions.items(), reverse=True):
            action = decision['action']
            
            try:
                annotations = self.data[entry_idx]['annotations'][annotation_idx]
                
                if action == 'reject':
                    # Remove the claim
                    del annotations['result'][result_idx]
                    print(f"  Removed claim from entry {self.data[entry_idx].get('id')}")
                    
                elif action == 'edit':
                    # Update the claim text
                    new_text = decision['new_text']
                    old_text = annotations['result'][result_idx]['value']['text']
                    
                    # Update the text and recalculate span if needed
                    annotations['result'][result_idx]['value']['text'] = new_text
                    # Note: We keep start/end positions unchanged for now
                    # In production, you might want to find the new position in the full text
                    
                    print(f"  Edited claim in entry {self.data[entry_idx].get('id')}")
                    print(f"    Old: '{old_text}'")
                    print(f"    New: '{new_text}'")
                    
            except (IndexError, KeyError) as e:
                print(f"Warning: Could not process claim at entry {entry_idx}: {e}")
        
        # Save updated data
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        
        accepted = sum(1 for d in self.decisions.values() if d['action'] == 'accept')
        rejected = sum(1 for d in self.decisions.values() if d['action'] == 'reject')
        edited = sum(1 for d in self.decisions.values() if d['action'] == 'edit')
        
        print(f"\n✅ Changes saved!")
        print(f"   Accepted: {accepted}")
        print(f"   Edited: {edited}")
        print(f"   Rejected: {rejected}")
        print(f"   Updated file: {self.json_path}")
    
    def run(self):
        """Main event loop"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if self.edit_mode:
                        # Edit mode controls
                        if event.key == pygame.K_ESCAPE:
                            self.cancel_edit()
                        
                        elif event.key == pygame.K_RETURN:
                            self.save_edit()
                        
                        elif event.key == pygame.K_BACKSPACE:
                            if self.cursor_pos > 0:
                                self.edit_text = self.edit_text[:self.cursor_pos-1] + self.edit_text[self.cursor_pos:]
                                self.cursor_pos -= 1
                        
                        elif event.key == pygame.K_DELETE:
                            if self.cursor_pos < len(self.edit_text):
                                self.edit_text = self.edit_text[:self.cursor_pos] + self.edit_text[self.cursor_pos+1:]
                        
                        elif event.key == pygame.K_LEFT:
                            self.cursor_pos = max(0, self.cursor_pos - 1)
                        
                        elif event.key == pygame.K_RIGHT:
                            self.cursor_pos = min(len(self.edit_text), self.cursor_pos + 1)
                        
                        elif event.key == pygame.K_HOME:
                            self.cursor_pos = 0
                        
                        elif event.key == pygame.K_END:
                            self.cursor_pos = len(self.edit_text)
                        
                        elif event.unicode and event.unicode.isprintable():
                            self.edit_text = self.edit_text[:self.cursor_pos] + event.unicode + self.edit_text[self.cursor_pos:]
                            self.cursor_pos += 1
                    
                    else:
                        # Normal mode controls
                        if event.key == pygame.K_ESCAPE:
                            self.save_changes()
                            running = False
                        
                        elif event.key in (pygame.K_LEFT, pygame.K_a):
                            self.make_decision('reject')
                        
                        elif event.key in (pygame.K_RIGHT, pygame.K_d):
                            self.make_decision('accept')
                        
                        elif event.key == pygame.K_e:
                            self.enter_edit_mode()
            
            self.draw_card()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

def main():
    json_path = Path(__file__).parent / 'data' / 'annotations' / 'claim_annotations_2000.json'
    
    if not json_path.exists():
        print(f"Error: Could not find {json_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("IDENTITY CLAIM REVIEWER")
    print("=" * 60)
    print("\nControls:")
    print("  ← or A : Reject (remove this claim)")
    print("  E      : Edit (modify the claim text)")
    print("  → or D : Accept (keep this claim as-is)")
    print("  ESC    : Save and exit")
    print("\nEdit Mode Controls:")
    print("  Type to edit text")
    print("  ENTER  : Save edited text")
    print("  ESC    : Cancel edit")
    print("\nStarting review...")
    print("-" * 60)
    
    reviewer = IdentityClaimReviewer(json_path)
    reviewer.run()

if __name__ == '__main__':
    main()

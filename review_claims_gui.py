#!/usr/bin/env python3
"""
Interactive GUI tool to manually review and accept/reject claim annotations
Uses pygame for UI
"""

import pygame
import json
import sys
from pathlib import Path

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1400, 900
BG_COLOR = (240, 240, 245)
TEXT_COLOR = (20, 20, 20)
HIGHLIGHT_COLOR = (255, 215, 0)
ACCEPT_COLOR = (76, 175, 80)
REJECT_COLOR = (244, 67, 54)
BUTTON_COLOR = (33, 150, 243)
PANEL_COLOR = (255, 255, 255)
EDIT_COLOR = (255, 152, 0)
WARNING_COLOR = (156, 39, 176)

# Fonts
FONT_LARGE = pygame.font.Font(None, 36)
FONT_MEDIUM = pygame.font.Font(None, 28)
FONT_SMALL = pygame.font.Font(None, 24)

class ClaimReviewer:
    def __init__(self, data_file):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Claim Annotation Reviewer")
        self.clock = pygame.time.Clock()
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Filter for claims to review (URGENCY, ACTION, REWARD only)
        self.review_items = []
        self.build_review_list()
        
        self.current_index = 0
        self.decisions = {}  # {(entry_idx, result_idx): 'accept' or 'reject'}
        self.edits = {}  # {(entry_idx, result_idx): {'text': new_text, 'type': new_type}}
        
        # Edit mode state
        self.edit_mode = False
        self.edit_text = ""
        self.change_type_mode = False
        self.available_types = ['URGENCY_CLAIM', 'ACTION_CLAIM', 'REWARD_CLAIM', 'FINANCIAL_CLAIM', 
                               'ACCOUNT_CLAIM', 'DELIVERY_CLAIM', 'VERIFICATION_CLAIM', 'OTHER_CLAIM']
        
    def build_review_list(self):
        """Build list of claims to review"""
        target_labels = ['URGENCY_CLAIM', 'ACTION_CLAIM', 'REWARD_CLAIM']
        
        for entry_idx, entry in enumerate(self.data):
            if not entry.get('annotations') or not entry['annotations']:
                continue
            
            annotations = entry['annotations'][0]
            if 'result' not in annotations or not annotations['result']:
                continue
            
            for result_idx, result in enumerate(annotations['result']):
                value = result.get('value', {})
                labels = value.get('labels', [])
                
                if labels and labels[0] in target_labels:
                    self.review_items.append({
                        'entry_idx': entry_idx,
                        'result_idx': result_idx,
                        'entry_id': entry.get('id'),
                        'message': entry['data']['text'],
                        'claim_text': value.get('text', ''),
                        'claim_type': labels[0],
                        'start': value.get('start'),
                        'end': value.get('end')
                    })
        
        print(f"Total claims to review: {len(self.review_items)}")
    
    def wrap_text(self, text, font, max_width):
        """Wrap text to fit within max_width"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font.size(test_line)[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def draw_button(self, text, x, y, width, height, color, text_color=TEXT_COLOR):
        """Draw a button"""
        pygame.draw.rect(self.screen, color, (x, y, width, height), border_radius=5)
        pygame.draw.rect(self.screen, (200, 200, 200), (x, y, width, height), 2, border_radius=5)
        text_surf = FONT_SMALL.render(text, True, text_color)
        text_rect = text_surf.get_rect(center=(x + width//2, y + height//2))
        self.screen.blit(text_surf, text_rect)
        return pygame.Rect(x, y, width, height)
    
    def draw(self):
        """Draw the UI"""
        self.screen.fill(BG_COLOR)
        
        if self.current_index >= len(self.review_items):
            # Show completion screen
            self.draw_completion_screen()
            return
        
        if self.change_type_mode:
            self.draw_change_type_screen()
            return
        
        if self.edit_mode:
            self.draw_edit_screen()
            return
        
        item = self.review_items[self.current_index]
        
        # Progress bar
        progress = (self.current_index / len(self.review_items)) * WIDTH
        pygame.draw.rect(self.screen, BUTTON_COLOR, (0, 0, progress, 10))
        
        # Title
        title = FONT_LARGE.render(f"Review Claim {self.current_index + 1} / {len(self.review_items)}", True, TEXT_COLOR)
        self.screen.blit(title, (20, 30))
        
        # Entry ID
        id_text = FONT_SMALL.render(f"ID: {item['entry_id']}", True, (100, 100, 100))
        self.screen.blit(id_text, (20, 70))
        
        # Message panel
        pygame.draw.rect(self.screen, PANEL_COLOR, (20, 110, WIDTH-40, 200), border_radius=10)
        pygame.draw.rect(self.screen, (200, 200, 200), (20, 110, WIDTH-40, 200), 2, border_radius=10)
        
        msg_label = FONT_MEDIUM.render("Full Message:", True, TEXT_COLOR)
        self.screen.blit(msg_label, (40, 120))
        
        # Wrap message text
        message_lines = self.wrap_text(item['message'], FONT_SMALL, WIDTH-80)
        y_offset = 155
        for line in message_lines[:6]:  # Show max 6 lines
            text_surf = FONT_SMALL.render(line, True, TEXT_COLOR)
            self.screen.blit(text_surf, (40, y_offset))
            y_offset += 30
        
        # Claim panel
        pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, (20, 330, WIDTH-40, 180), border_radius=10)
        pygame.draw.rect(self.screen, (200, 150, 0), (20, 330, WIDTH-40, 180), 3, border_radius=10)
        
        claim_label = FONT_MEDIUM.render(f"Claim Type: {item['claim_type']}", True, TEXT_COLOR)
        self.screen.blit(claim_label, (40, 345))
        
        claim_text_label = FONT_SMALL.render("Claim Text:", True, TEXT_COLOR)
        self.screen.blit(claim_text_label, (40, 385))
        
        # Wrap claim text
        claim_lines = self.wrap_text(item['claim_text'], FONT_MEDIUM, WIDTH-80)
        y_offset = 420
        for line in claim_lines[:3]:
            text_surf = FONT_MEDIUM.render(line, True, TEXT_COLOR)
            self.screen.blit(text_surf, (40, y_offset))
            y_offset += 35
        
        # Position info
        pos_text = FONT_SMALL.render(f"Position: [{item['start']}:{item['end']}]", True, (100, 100, 100))
        self.screen.blit(pos_text, (40, y_offset + 10))
        
        # Action buttons
        button_y = 550
        button_width = 250
        button_height = 50
        button_spacing = 20
        
        self.accept_btn = self.draw_button("A - ACCEPT", 40, button_y, button_width, button_height, ACCEPT_COLOR, (255, 255, 255))
        self.reject_btn = self.draw_button("R - REJECT", 40 + button_width + button_spacing, button_y, button_width, button_height, REJECT_COLOR, (255, 255, 255))
        self.skip_btn = self.draw_button("S - SKIP", 40 + (button_width + button_spacing) * 2, button_y, button_width, button_height, (150, 150, 150), (255, 255, 255))
        
        button_y += button_height + button_spacing
        self.edit_btn = self.draw_button("E - EDIT TEXT", 40, button_y, button_width, button_height, EDIT_COLOR, (255, 255, 255))
        self.change_type_btn = self.draw_button("T - CHANGE TYPE", 40 + button_width + button_spacing, button_y, button_width, button_height, WARNING_COLOR, (255, 255, 255))
        self.quit_btn = self.draw_button("Q - QUIT & SAVE", 40 + (button_width + button_spacing) * 2, button_y, button_width, button_height, BUTTON_COLOR, (255, 255, 255))
        
        # Decision counter
        accepted = sum(1 for d in self.decisions.values() if d == 'accept')
        rejected = sum(1 for d in self.decisions.values() if d == 'reject')
        
        stats_y = 750
        stats = [
            f"Accepted: {accepted}",
            f"Rejected: {rejected}",
            f"Remaining: {len(self.review_items) - self.current_index}"
        ]
        
        for i, stat in enumerate(stats):
            text_surf = FONT_MEDIUM.render(stat, True, TEXT_COLOR)
            self.screen.blit(text_surf, (40, stats_y + i * 35))
        
        pygame.display.flip()
    
    def draw_edit_screen(self):
        """Draw edit text screen"""
        self.screen.fill(BG_COLOR)
        
        item = self.review_items[self.current_index]
        
        title = FONT_LARGE.render("Edit Claim Text", True, EDIT_COLOR)
        self.screen.blit(title, (40, 40))
        
        # Original text
        orig_label = FONT_MEDIUM.render("Original:", True, TEXT_COLOR)
        self.screen.blit(orig_label, (40, 120))
        
        orig_lines = self.wrap_text(item['claim_text'], FONT_SMALL, WIDTH-80)
        y = 160
        for line in orig_lines[:3]:
            text_surf = FONT_SMALL.render(line, True, (100, 100, 100))
            self.screen.blit(text_surf, (40, y))
            y += 30
        
        # Edit box
        edit_label = FONT_MEDIUM.render("New Text (type and press ENTER):", True, TEXT_COLOR)
        self.screen.blit(edit_label, (40, 280))
        
        # Input box
        pygame.draw.rect(self.screen, (255, 255, 255), (40, 320, WIDTH-80, 100), border_radius=5)
        pygame.draw.rect(self.screen, EDIT_COLOR, (40, 320, WIDTH-80, 100), 3, border_radius=5)
        
        # Wrap edit text
        edit_lines = self.wrap_text(self.edit_text, FONT_MEDIUM, WIDTH-100)
        y = 340
        for line in edit_lines[:2]:
            text_surf = FONT_MEDIUM.render(line, True, TEXT_COLOR)
            self.screen.blit(text_surf, (50, y))
            y += 40
        
        # Instructions
        inst = [
            "Type the corrected claim text",
            "Press ENTER to save edit",
            "Press ESC to cancel"
        ]
        y = 480
        for line in inst:
            text_surf = FONT_SMALL.render(line, True, (100, 100, 100))
            self.screen.blit(text_surf, (40, y))
            y += 35
        
        pygame.display.flip()
    
    def draw_change_type_screen(self):
        """Draw change claim type screen"""
        self.screen.fill(BG_COLOR)
        
        item = self.review_items[self.current_index]
        
        title = FONT_LARGE.render("Change Claim Type", True, WARNING_COLOR)
        self.screen.blit(title, (40, 40))
        
        # Current type
        current_label = FONT_MEDIUM.render(f"Current: {item['claim_type']}", True, TEXT_COLOR)
        self.screen.blit(current_label, (40, 120))
        
        # Claim text
        claim_label = FONT_SMALL.render(f"Claim: '{item['claim_text'][:80]}...'", True, (100, 100, 100))
        self.screen.blit(claim_label, (40, 160))
        
        # Type selection buttons
        inst_label = FONT_MEDIUM.render("Select new type (press number):", True, TEXT_COLOR)
        self.screen.blit(inst_label, (40, 220))
        
        y = 270
        for i, claim_type in enumerate(self.available_types, 1):
            color = BUTTON_COLOR if claim_type != item['claim_type'] else (150, 150, 150)
            self.draw_button(f"{i} - {claim_type}", 40, y, 600, 45, color, (255, 255, 255))
            y += 55
        
        # Instructions
        inst = FONT_SMALL.render("Press ESC to cancel", True, (100, 100, 100))
        self.screen.blit(inst, (40, y + 20))
        
        pygame.display.flip()
    
    def draw_completion_screen(self):
        """Draw completion screen"""
        self.screen.fill(BG_COLOR)
        
        title = FONT_LARGE.render("Review Complete!", True, ACCEPT_COLOR)
        self.screen.blit(title, (WIDTH//2 - 150, 200))
        
        accepted = sum(1 for d in self.decisions.values() if d == 'accept')
        rejected = sum(1 for d in self.decisions.values() if d == 'reject')
        
        stats = [
            f"Total reviewed: {len(self.decisions)}",
            f"Accepted: {accepted}",
            f"Rejected: {rejected}",
            "",
            "Press S to save and quit",
            "Press Q to quit without saving"
        ]
        
        y_offset = 300
        for stat in stats:
            text_surf = FONT_MEDIUM.render(stat, True, TEXT_COLOR)
            self.screen.blit(text_surf, (WIDTH//2 - text_surf.get_width()//2, y_offset))
            y_offset += 50
        
        pygame.display.flip()
    
    def handle_decision(self, decision):
        """Handle accept/reject decision"""
        if self.current_index < len(self.review_items):
            item = self.review_items[self.current_index]
            key = (item['entry_idx'], item['result_idx'])
            self.decisions[key] = decision
            print(f"{decision.upper()}: {item['claim_type']} - '{item['claim_text'][:50]}'")
            self.current_index += 1
    
    def save_decisions(self):
        """Save decisions and update dataset"""
        print("\nSaving decisions...")
        
        # Apply edits and remove rejected claims
        removed_count = 0
        edited_count = 0
        type_changed_count = 0
        
        for entry_idx, entry in enumerate(self.data):
            if not entry.get('annotations') or not entry['annotations']:
                continue
            
            annotations = entry['annotations'][0]
            if 'result' not in annotations or not annotations['result']:
                continue
            
            # Filter and update results
            new_results = []
            for result_idx, result in enumerate(annotations['result']):
                key = (entry_idx, result_idx)
                
                # Check if rejected
                if key in self.decisions and self.decisions[key] == 'reject':
                    removed_count += 1
                    continue
                
                # Apply edits
                if key in self.edits:
                    edits = self.edits[key]
                    
                    if 'text' in edits:
                        # Update text
                        old_text = result['value']['text']
                        new_text = edits['text']
                        result['value']['text'] = new_text
                        
                        # Recalculate position (keep start, adjust end)
                        result['value']['end'] = result['value']['start'] + len(new_text)
                        edited_count += 1
                        print(f"  Edited: '{old_text[:30]}...' -> '{new_text[:30]}...'")
                    
                    if 'type' in edits:
                        # Update type
                        old_type = result['value']['labels'][0]
                        new_type = edits['type']
                        result['value']['labels'] = [new_type]
                        type_changed_count += 1
                        print(f"  Type changed: {old_type} -> {new_type}")
                
                new_results.append(result)
            
            annotations['result'] = new_results
            annotations['result_count'] = len(new_results)
        
        # Save to new file
        output_file = Path('data/annotations/claim_annotations_2000_reviewed.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved reviewed dataset:")
        print(f"  Removed: {removed_count} rejected claims")
        print(f"  Edited: {edited_count} claim texts")
        print(f"  Type changed: {type_changed_count} claims")
        print(f"  Saved to: {output_file}")
        print(f"\nUse this file for training: claim_annotations_2000_reviewed.json")
    
    def run(self):
        """Main loop"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Auto-save before quitting
                    print("\nWindow closed - auto-saving...")
                    self.save_decisions()
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    # Edit mode handling
                    if self.edit_mode:
                        if event.key == pygame.K_RETURN:
                            # Save edit
                            if self.edit_text.strip():
                                item = self.review_items[self.current_index]
                                key = (item['entry_idx'], item['result_idx'])
                                if key not in self.edits:
                                    self.edits[key] = {}
                                self.edits[key]['text'] = self.edit_text.strip()
                                # Update review item display
                                item['claim_text'] = self.edit_text.strip()
                                print(f"EDITED: {item['claim_type']} - '{self.edit_text[:50]}'")
                            self.edit_mode = False
                            self.edit_text = ""
                        
                        elif event.key == pygame.K_ESCAPE:
                            # Cancel edit
                            self.edit_mode = False
                            self.edit_text = ""
                        
                        elif event.key == pygame.K_BACKSPACE:
                            self.edit_text = self.edit_text[:-1]
                        
                        else:
                            # Add character
                            if event.unicode and len(self.edit_text) < 300:
                                self.edit_text += event.unicode
                    
                    # Change type mode handling
                    elif self.change_type_mode:
                        if event.key == pygame.K_ESCAPE:
                            self.change_type_mode = False
                        
                        elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, 
                                          pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8]:
                            # Select type
                            type_idx = int(event.unicode) - 1
                            if 0 <= type_idx < len(self.available_types):
                                item = self.review_items[self.current_index]
                                new_type = self.available_types[type_idx]
                                key = (item['entry_idx'], item['result_idx'])
                                if key not in self.edits:
                                    self.edits[key] = {}
                                self.edits[key]['type'] = new_type
                                # Update review item display
                                item['claim_type'] = new_type
                                print(f"CHANGED TYPE: {new_type} - '{item['claim_text'][:50]}'")
                                self.change_type_mode = False
                    
                    # Normal mode handling
                    else:
                        if event.key == pygame.K_q:
                            # Quit
                            running = False
                        
                        elif event.key == pygame.K_a:
                            # Accept
                            self.handle_decision('accept')
                        
                        elif event.key == pygame.K_r:
                            # Reject
                            self.handle_decision('reject')
                        
                        elif event.key == pygame.K_s:
                            if self.current_index >= len(self.review_items):
                                # Save and quit
                                self.save_decisions()
                                running = False
                            else:
                                # Skip
                                self.current_index += 1
                        
                        elif event.key == pygame.K_e:
                            # Edit mode
                            if self.current_index < len(self.review_items):
                                item = self.review_items[self.current_index]
                                self.edit_text = item['claim_text']
                                self.edit_mode = True
                        
                        elif event.key == pygame.K_t:
                            # Change type mode
                            if self.current_index < len(self.review_items):
                                self.change_type_mode = True
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.edit_mode and not self.change_type_mode:
                        mouse_pos = event.pos
                        
                        # Check button clicks
                        if hasattr(self, 'accept_btn') and self.accept_btn.collidepoint(mouse_pos):
                            self.handle_decision('accept')
                        elif hasattr(self, 'reject_btn') and self.reject_btn.collidepoint(mouse_pos):
                            self.handle_decision('reject')
                        elif hasattr(self, 'skip_btn') and self.skip_btn.collidepoint(mouse_pos):
                            if self.current_index < len(self.review_items):
                                self.current_index += 1
                        elif hasattr(self, 'edit_btn') and self.edit_btn.collidepoint(mouse_pos):
                            if self.current_index < len(self.review_items):
                                item = self.review_items[self.current_index]
                                self.edit_text = item['claim_text']
                                self.edit_mode = True
                        elif hasattr(self, 'change_type_btn') and self.change_type_btn.collidepoint(mouse_pos):
                            if self.current_index < len(self.review_items):
                                self.change_type_mode = True
                        elif hasattr(self, 'quit_btn') and self.quit_btn.collidepoint(mouse_pos):
                            running = False
            
            self.draw()
            self.clock.tick(30)
        
        pygame.quit()

if __name__ == '__main__':
    data_file = Path('data/annotations/claim_annotations_2000_clean.json')
    
    if not data_file.exists():
        print(f"Error: {data_file} not found!")
        sys.exit(1)
    
    print("="*70)
    print("CLAIM ANNOTATION REVIEWER")
    print("="*70)
    print("\nLoading dataset...")
    
    reviewer = ClaimReviewer(data_file)
    
    print("\nControls:")
    print("  A = Accept (keep claim)")
    print("  R = Reject (remove claim)")
    print("  S = Skip to next")
    print("  Q = Quit and save")
    print("\nStarting review...")
    print("="*70)
    
    reviewer.run()

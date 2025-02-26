# ui_components.py
"""
UI Components for the Tetris AI Training Visualizer.

This module provides reusable UI components like buttons, sliders, checkboxes,
and tab managers for the pygame-based training visualization interface.
These components handle their own rendering and event processing.
"""

import pygame

class Button:
    """Interactive button for the UI"""
    def __init__(self, text, x, y, width, height, color, hover_color, action=None):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.action = action
        self.font = pygame.font.SysFont('Arial', 16)
    
    def draw(self, screen):
        mouse = pygame.mouse.get_pos()
        
        # Check if mouse is over button
        if self.is_over(mouse):
            self.current_color = self.hover_color
        else:
            self.current_color = self.color
        
        # Draw button rectangle
        pygame.draw.rect(screen, self.current_color, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (200, 200, 200), (self.x, self.y, self.width, self.height), 1)
        
        # Draw text
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(self.x + self.width/2, self.y + self.height/2))
        screen.blit(text_surf, text_rect)
    
    def is_over(self, pos):
        return self.x < pos[0] < self.x + self.width and self.y < pos[1] < self.y + self.height
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_over(pygame.mouse.get_pos()) and self.action:
                self.action()
                return True
        return False


class Slider:
    """Interactive slider for adjusting values"""
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label, action=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.action = action
        self.dragging = False
        self.font = pygame.font.SysFont('Arial', 16)
    
    def draw(self, screen):
        # Draw slider track
        pygame.draw.rect(screen, (80, 80, 80), (self.x, self.y, self.width, self.height))
        
        # Calculate handle position
        handle_x = self.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.width
        
        # Draw slider handle
        handle_width = 10
        pygame.draw.rect(screen, (200, 200, 200), 
                        (handle_x - handle_width/2, self.y - 5, handle_width, self.height + 10))
        
        # Draw label and value
        label_text = self.font.render(f"{self.label}: {self.value:.1f}", True, (255, 255, 255))
        screen.blit(label_text, (self.x, self.y - 20))
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_over(pygame.mouse.get_pos()):
                self.dragging = True
                self.update_value(pygame.mouse.get_pos()[0])
                if self.action:
                    self.action(self.value)
                return True
        
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.update_value(pygame.mouse.get_pos()[0])
            if self.action:
                self.action(self.value)
            return True
        
        return False
    
    def is_over(self, pos):
        return self.x < pos[0] < self.x + self.width and self.y - 5 < pos[1] < self.y + self.height + 5
    
    def update_value(self, x_pos):
        relative_x = max(0, min(x_pos - self.x, self.width))
        self.value = self.min_val + (relative_x / self.width) * (self.max_val - self.min_val)
        self.value = round(self.value * 10) / 10  # Round to 1 decimal place


class CheckBox:
    """Interactive checkbox for toggling options"""
    def __init__(self, x, y, size, label, initial_state=False, action=None):
        self.x = x
        self.y = y
        self.size = size
        self.label = label
        self.checked = initial_state
        self.action = action
        self.font = pygame.font.SysFont('Arial', 16)
    
    def draw(self, screen):
        # Draw checkbox
        pygame.draw.rect(screen, (200, 200, 200), (self.x, self.y, self.size, self.size), 1)
        
        # Draw checkmark if checked
        if self.checked:
            inner_margin = self.size // 4
            pygame.draw.rect(screen, (200, 200, 200), 
                            (self.x + inner_margin, self.y + inner_margin, 
                             self.size - 2*inner_margin, self.size - 2*inner_margin))
        
        # Draw label
        label_text = self.font.render(self.label, True, (255, 255, 255))
        screen.blit(label_text, (self.x + self.size + 10, self.y + self.size // 2 - 8))
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_over(pygame.mouse.get_pos()):
                self.checked = not self.checked
                if self.action:
                    self.action(self.checked)
                return True
        return False
    
    def is_over(self, pos):
        return (self.x < pos[0] < self.x + self.size and 
                self.y < pos[1] < self.y + self.size)


class TabManager:
    """Manages tab-based interface"""
    def __init__(self, x, y, width, tab_height):
        self.x = x
        self.y = y
        self.width = width
        self.tab_height = tab_height
        self.tabs = []
        self.active_tab = 0
        self.font = pygame.font.SysFont('Arial', 16)
    
    def add_tab(self, title):
        self.tabs.append(title)
        return len(self.tabs) - 1
    
    def draw(self, screen):
        # Calculate tab width
        tab_width = self.width // max(1, len(self.tabs))
        
        # Draw tabs
        for i, tab in enumerate(self.tabs):
            # Determine colors based on active state
            if i == self.active_tab:
                bg_color = (60, 60, 60)
                text_color = (255, 255, 255)
            else:
                bg_color = (40, 40, 40)
                text_color = (200, 200, 200)
            
            # Draw tab background
            pygame.draw.rect(screen, bg_color, 
                            (self.x + i * tab_width, self.y, tab_width, self.tab_height))
            pygame.draw.rect(screen, (100, 100, 100), 
                            (self.x + i * tab_width, self.y, tab_width, self.tab_height), 1)
            
            # Draw tab title
            text_surf = self.font.render(tab, True, text_color)
            text_rect = text_surf.get_rect(center=(self.x + i * tab_width + tab_width/2, 
                                                  self.y + self.tab_height/2))
            screen.blit(text_surf, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            tab_width = self.width // max(1, len(self.tabs))
            
            for i in range(len(self.tabs)):
                tab_rect = pygame.Rect(self.x + i * tab_width, self.y, tab_width, self.tab_height)
                if tab_rect.collidepoint(mouse_pos):
                    self.active_tab = i
                    return True
        
        return False
    
    def get_active_tab(self):
        return self.active_tab

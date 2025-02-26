# plotting.py
"""
Plotting utilities for the Tetris AI Training Visualizer.

This module provides classes for creating and updating plots to visualize 
training metrics such as loss, reward, and score using pygame.
"""

import pygame

class PlotData:
    """Helper class for plotting data"""
    def __init__(self, title, max_points, color):
        self.title = title
        self.data = []
        self.max_points = max_points
        self.color = color
        self.min_value = 0
        self.max_value = 1
    
    def update_data(self, new_data):
        """Update the plot data"""
        self.data = new_data
        if len(self.data) > 0:
            self.min_value = min(self.data)
            self.max_value = max(self.data)
            
            # Ensure range is never zero to avoid division issues
            if self.min_value == self.max_value:
                self.min_value -= 0.1
                self.max_value += 0.1
    
    def draw(self, surface, x, y, width, height):
        """Draw the plot on the given surface"""
        # Draw border and background
        pygame.draw.rect(surface, (30, 30, 30), (x, y, width, height))
        pygame.draw.rect(surface, (70, 70, 70), (x, y, width, height), 1)
        
        # Draw title
        font = pygame.font.SysFont('Arial', 16)
        title_text = font.render(self.title, True, (255, 255, 255))
        surface.blit(title_text, (x + 5, y + 5))
        
        # Draw grid lines
        plot_x = x + 10
        plot_y = y + 30
        plot_width = width - 20
        plot_height = height - 50
        
        # Background grid
        for i in range(5):
            # Horizontal grid lines
            line_y = plot_y + i * (plot_height / 4)
            pygame.draw.line(surface, (50, 50, 50), 
                            (plot_x, line_y), 
                            (plot_x + plot_width, line_y), 1)
            
            # Vertical grid lines
            line_x = plot_x + i * (plot_width / 4)
            pygame.draw.line(surface, (50, 50, 50), 
                            (line_x, plot_y), 
                            (line_x, plot_y + plot_height), 1)
        
        # Draw min/max values
        if len(self.data) > 0:
            min_text = font.render(f"Min: {self.min_value:.2f}", True, (200, 200, 200))
            max_text = font.render(f"Max: {self.max_value:.2f}", True, (200, 200, 200))
            
            surface.blit(min_text, (x + 5, y + height - 20))
            surface.blit(max_text, (x + width - 100, y + height - 20))
        
        # Draw the plot if we have data
        if len(self.data) > 1:
            # Calculate points
            points = []
            for i, value in enumerate(self.data):
                px = plot_x + (i / (len(self.data) - 1)) * plot_width
                # Normalize and invert y (since pygame origin is top-left)
                normalized = (value - self.min_value) / (self.max_value - self.min_value)
                py = plot_y + plot_height - normalized * plot_height
                points.append((px, py))
            
            # Draw the line connecting points
            if len(points) > 1:
                pygame.draw.lines(surface, self.color, False, points, 2)
                
                # Draw small circles at each data point
                for point in points[::max(1, len(points)//20)]:  # Show a subset of points
                    pygame.draw.circle(surface, self.color, (int(point[0]), int(point[1])), 3)

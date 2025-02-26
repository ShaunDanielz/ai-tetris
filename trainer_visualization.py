# trainer_visualization.py
"""
Tetris AI Training Visualization Interface.

This module provides a visual interface for the Tetris DQN training process,
displaying the game board, training metrics, and controls for interacting with
the training process. It integrates the TetrisEnv, DQNAgent, and UI components
to create a comprehensive training dashboard.
"""

import os
import sys
import time
import json
import random
import pygame
import numpy as np
from datetime import datetime
import tensorflow as tf

from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
from ui_components import Button, Slider, CheckBox, TabManager
from plotting import PlotData
from utils import convert_model_to_tfjs
from config import VIBRANT_COLORS, BLACK, WHITE, GRAY, GRID_COLOR, TFJS_AVAILABLE

class VisualTetrisDQNTrainer:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        
        # Screen dimensions
        self.window_width = 1280
        self.window_height = 950
        self.tetris_width = 300
        self.tetris_height = 600
        self.block_size = 30
        
        # Set up the display
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Enhanced Tetris DQN Training Visualizer")
        
        # UI state
        self.tab_manager = TabManager(400, 50, 800, 30)
        self.tab_manager.add_tab("Training")
        self.tab_manager.add_tab("Metrics")
        self.tab_manager.add_tab("Network")
        self.tab_manager.add_tab("Board Analysis")
        self.tab_manager.add_tab("Performance")
        self.tab_manager.add_tab("Web Export")
        
        # Font
        self.font = pygame.font.SysFont('Arial', 16)
        self.big_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 12)
        
        # Create environment and agent
        self.env = TetrisEnv()
        self.state_size = 204  # 20x10 board + 4 for piece encoding
        self.action_size = 40  # 10 positions x 4 rotations
        self.agent = DQNAgent(self.state_size, self.action_size)
        
        # Auto-save setting
        self.auto_save_enabled = True
        
        # Training parameters
        self.episodes = 100000
        self.target_update_freq = 10
        self.training_active = False
        self.training_speed = 1.0  # Default speed multiplier
        self.frame_delay = 50  # ms between frames
        self.show_heatmap = False
        self.show_q_values = False
        self.show_ghost_piece = True
        self.step_mode = False
        self.do_step = False
        self.frame_skip = 0
        self.current_frame = 0
        
        # Training stats
        self.training_start_time = None
        self.total_training_time = 0
        self.episodes_per_second = 0
        self.steps_per_second = 0
        self.total_steps = 0
        self.training_sessions = 0
        
        # Current game state
        self.state = None
        self.episode = 0
        self.step_count = 0
        self.player = None  # Added for tracking current piece
        
        # UI elements
        self.ui_elements = []
        self._setup_ui()
        
        # Performance metrics plots
        self.plot_loss = PlotData("Loss", 100, (255, 50, 50))
        self.plot_epsilon = PlotData("Epsilon", 100, (50, 150, 255))
        self.plot_reward = PlotData("Reward", 100, (50, 255, 50))
        self.plot_score = PlotData("Score", 100, (255, 255, 50))
        
        # Time tracking
        self.last_update_time = time.time()
        self.last_fps_update = time.time()
        self.frame_count = 0
        self.fps = 0

        # Web export settings
        self.auto_export_web = True  # Auto export to web when saving
        self.web_export_status = "No exports yet"
        self.last_web_export_time = None
        
        # Try to load latest model at startup
        self.load_latest_model()

    def toggle_auto_web_export(self, value):
        """Toggle automatic export to web"""
        self.auto_export_web = value

    def export_for_web(self):
        """Export the current model for web use"""
        if not TFJS_AVAILABLE:
            self.web_export_status = "Error: tensorflowjs not installed"
            print("TensorFlow.js conversion tools not installed. Use 'pip install tensorflowjs' to install.")
            return
        
        # Create output directory
        output_dir = "models/tetris_dqn/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full model (not just weights)
        model_path = f"{output_dir}tetris_dqn_full_{self.episode}.h5"
        self.agent.save_full_model(model_path)
        
        # Convert to TensorFlow.js format
        web_output_dir = f"{output_dir}web_model_{self.episode}/"
        
        success = convert_model_to_tfjs(model_path, web_output_dir)
        
        if success:
            self.web_export_status = f"Successfully exported model for web at episode {self.episode}"
            self.last_web_export_time = time.time()
            print(f"Model exported for web use at {web_output_dir}")
        else:
            self.web_export_status = "Export failed. See console for details."    
        
    def _setup_ui(self):
        """Set up all UI elements"""
        # Training controls
        self.ui_elements.append(Button("Start Training", 420, 100, 120, 30, 
                                       (0, 120, 0), (0, 180, 0), 
                                       self.toggle_training))
        
        self.ui_elements.append(Button("Step", 550, 100, 80, 30, 
                                      (0, 80, 120), (0, 120, 180), 
                                      self.do_single_step))
        
        self.ui_elements.append(Button("Save Model", 640, 100, 120, 30, 
                                      (120, 0, 120), (180, 0, 180), 
                                      self.save_model))
        
        self.ui_elements.append(Button("Load Model", 770, 100, 120, 30, 
                                      (120, 80, 0), (180, 120, 0), 
                                      self.load_model))
        
        self.ui_elements.append(Button("Reset", 900, 100, 80, 30, 
                                      (120, 0, 0), (180, 0, 0), 
                                      self.reset_training))
        
        # Adjusted placement (next to "Reset")
        self.ui_elements.append(Button("Export for Web", 990, 100, 140, 30, 
                               (0, 100, 180), (0, 140, 220), 
                               self.export_for_web))

        
        # Training speed slider
        self.ui_elements.append(Slider(420, 160, 250, 10, 0.5, 10.0, 1.0, 
                                      "Training Speed", self.set_training_speed))
        
        # Epsilon slider
        self.ui_elements.append(Slider(700, 160, 250, 10, 0.01, 1.0, 1.0, 
                                      "Exploration Rate (ε)", self.set_epsilon))
        
        # Visualization options
        self.ui_elements.append(CheckBox(420, 200, 20, "Show Heatmap", False, 
                                        self.toggle_heatmap))
        
        self.ui_elements.append(CheckBox(600, 200, 20, "Show Q-Values", False, 
                                        self.toggle_q_values))
        
        self.ui_elements.append(CheckBox(800, 200, 20, "Show Ghost Piece", True, 
                                        self.toggle_ghost_piece))
        
        self.ui_elements.append(CheckBox(420, 230, 20, "Step Mode", False, 
                                        self.toggle_step_mode))
        
        # Auto-save checkbox
        self.ui_elements.append(CheckBox(600, 230, 20, "Auto-Save", True, 
                                       self.toggle_autosave))
        
        # Frame skip slider (for speed) - FIXED: Moved to a new line
        self.ui_elements.append(Slider(420, 300, 250, 10, 0, 10, 0, 
                               "Frame Skip", self.set_frame_skip))
        
        # Auto-export to web checkbox
        self.ui_elements.append(CheckBox(800, 230, 20, "Auto-Export for Web", True, 
                                        self.toggle_auto_web_export))      

    
    def set_training_speed(self, value):
        """Set the training speed multiplier"""
        self.training_speed = value
    
    def set_epsilon(self, value):
        """Set the agent's exploration rate"""
        self.agent.epsilon = value
    
    def toggle_heatmap(self, value):
        """Toggle heatmap visualization"""
        self.show_heatmap = value
    
    def toggle_q_values(self, value):
        """Toggle Q-values visualization"""
        self.show_q_values = value
    
    def toggle_ghost_piece(self, value):
        """Toggle ghost piece visualization"""
        self.show_ghost_piece = value
    
    def toggle_step_mode(self, value):
        """Toggle step-by-step mode"""
        self.step_mode = value
        if value:
            self.training_active = False
            self.update_ui_button_text("Start Training", "Resume")
    
    def toggle_autosave(self, value):
        """Toggle automatic model saving"""
        self.auto_save_enabled = value
    
    def do_single_step(self):
        """Perform a single training step"""
        self.do_step = True
    
    def set_frame_skip(self, value):
        """Set number of frames to skip between renders"""
        self.frame_skip = int(value)
    
    def update_ui_button_text(self, button_text, new_text):
        """Update button text on UI"""
        for element in self.ui_elements:
            if isinstance(element, Button) and element.text == button_text:
                element.text = new_text
                break
    
    def reset_game(self):
        """Reset the environment for a new episode"""
        self.state = self.env.reset()
        self.step_count = 0
        
        # FIXED: Make sure the current player state reflects the new piece
        if not self.env.game_over:
            # Update player to show the new piece properly
            tetro_shape = self.env.current_piece
            tetro_color = self.env.piece_colors.get(np.max(tetro_shape), VIBRANT_COLORS[0])
            self.player = {
                'pos': self.env.current_pos.copy(),
                'tetromino': tetro_shape.copy(),
                'color': tetro_color,
                'collided': False
            }
    
    def save_model_auto(self):
        """Auto-save model with fixed name for continuity"""
        # Create output directory
        output_dir = "models/tetris_dqn/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training state alongside model
        training_state = {
            'episode': self.episode,
            'epsilon': self.agent.epsilon,
            'max_score': self.agent.max_score,
            'timestamp': time.time()
        }
        
        # First save to a temporary file (to prevent corruption if interrupted)
        temp_model_path = f"{output_dir}tetris_dqn_latest.temp.h5"
        temp_state_path = f"{output_dir}training_state.temp.json"
        
        try:
            # Save model
            self.agent.save(temp_model_path)
            
            # Save state
            with open(temp_state_path, 'w') as f:
                json.dump(training_state, f)
            
            # Now rename to final files
            final_model_path = f"{output_dir}tetris_dqn_latest.h5"
            final_state_path = f"{output_dir}training_state.json"
            
            # On Windows, we need to remove the destination files first
            if os.path.exists(final_model_path):
                os.remove(final_model_path)
            if os.path.exists(final_state_path):
                os.remove(final_state_path)
                
            os.rename(temp_model_path, final_model_path)
            os.rename(temp_state_path, final_state_path)
            
            print(f"Model auto-saved at episode {self.episode}")

            # Add after successful model save
            if self.auto_export_web and TFJS_AVAILABLE:
                # Save full model (not just weights) for conversion
                full_model_path = f"{output_dir}tetris_dqn_latest_full.h5"
                self.agent.save_full_model(full_model_path)
                
                # Convert to TensorFlow.js format
                web_output_dir = f"{output_dir}web_model_latest/"
                success = convert_model_to_tfjs(full_model_path, web_output_dir)
                
                if success:
                    self.web_export_status = f"Auto-exported for web at episode {self.episode}"
                    self.last_web_export_time = time.time()
                    print(f"Model auto-exported for web use at {web_output_dir}")
                else:
                    self.web_export_status = "Auto-export failed. See console for details."
            
        except Exception as e:
            print(f"Error during auto-save: {e}")
            # Clean up temp files if they exist
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            if os.path.exists(temp_state_path):
                os.remove(temp_state_path)
    
    def load_latest_model(self):
        """Attempt to load the latest model and training state"""
        output_dir = "models/tetris_dqn/"
        model_path = f"{output_dir}tetris_dqn_latest.h5"
        state_path = f"{output_dir}training_state.json"
        
        if os.path.exists(model_path) and os.path.exists(state_path):
            try:
                # Load model weights
                self.agent.load(model_path)
                
                # Load training state
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                # Restore training state
                self.episode = state['episode']
                self.agent.epsilon = state['epsilon']
                self.agent.max_score = state['max_score']
                
                print(f"Loaded model from episode {self.episode} with epsilon {self.agent.epsilon:.4f}")
                return True
                
            except Exception as e:
                print(f"Error loading saved model: {e}")
                return False
        else:
            print("No saved model found. Starting fresh training.")
            return False
    
    def save_milestone(self):
        """Save a milestone version every 1000 episodes"""
        if self.episode % 1000 == 0 and self.episode > 0:
            output_dir = "models/tetris_dqn/milestones/"
            os.makedirs(output_dir, exist_ok=True)
            
            milestone_path = f"{output_dir}tetris_dqn_ep{self.episode}.h5"
            self.agent.save(milestone_path)
            print(f"Milestone saved at episode {self.episode}")
    
    def render_tetris(self):
        """Draw the Tetris game board"""
        # Calculate the position for the Tetris board
        tetris_x = 50
        tetris_y = 50
        
        # Draw the game board background
        pygame.draw.rect(self.screen, BLACK, (tetris_x, tetris_y, self.tetris_width, self.tetris_height))
        
        # Draw the grid lines
        for x in range(self.env.width + 1):
            pygame.draw.line(
                self.screen, GRID_COLOR,
                (tetris_x + x * self.block_size, tetris_y),
                (tetris_x + x * self.block_size, tetris_y + self.tetris_height)
            )
        
        for y in range(self.env.height + 1):
            pygame.draw.line(
                self.screen, GRID_COLOR,
                (tetris_x, tetris_y + y * self.block_size),
                (tetris_x + self.tetris_width, tetris_y + y * self.block_size)
            )
        
        # Draw heatmap if enabled
        if self.show_heatmap:
            heatmap = self.env.get_heatmap_data()
            max_heat = max(1.0, np.max(heatmap))  # Avoid division by zero
            
            for y in range(self.env.height):
                for x in range(self.env.width):
                    heat_value = heatmap[y][x] / max_heat
                    if heat_value > 0:
                        # Red intensity based on heat value
                        red = int(255 * min(1.0, heat_value))
                        pygame.draw.rect(
                            self.screen, (red, 0, 0),
                            (tetris_x + x * self.block_size, tetris_y + y * self.block_size,
                             self.block_size, self.block_size),
                            0
                        )
        
        # Get the board with current piece
        board = self.env.get_board_with_piece()
        
        # Draw ghost piece if enabled
        if self.show_ghost_piece and not self.env.game_over:
            ghost_x = self.env.ghost_position['x']
            ghost_y = self.env.ghost_position['y']
            
            for y in range(len(self.env.current_piece)):
                for x in range(len(self.env.current_piece[y])):
                    if self.env.current_piece[y][x] != 0:
                        piece_value = self.env.current_piece[y][x]
                        
                        # Get color and make translucent
                        if piece_value in self.env.piece_colors:
                            color = self.env.piece_colors[piece_value]
                            ghost_color = (color[0], color[1], color[2])
                        else:
                            ghost_color = (100, 100, 100)
                        
                        # Draw ghost piece as outline
                        pygame.draw.rect(
                            self.screen, ghost_color,
                            (tetris_x + (ghost_x + x) * self.block_size, 
                             tetris_y + (ghost_y + y) * self.block_size,
                             self.block_size, self.block_size),
                            1  # Just the outline
                        )
        
        # Draw the blocks
        for y in range(self.env.height):
            for x in range(self.env.width):
                if board[y][x] != 0:
                    piece_value = board[y][x]
                    
                    # Use the color assigned to this piece
                    if piece_value in self.env.piece_colors:
                        color = self.env.piece_colors[piece_value]
                    else:
                        # Default color if not found
                        color = random.choice(VIBRANT_COLORS)
                        self.env.piece_colors[piece_value] = color
                    
                    pygame.draw.rect(
                        self.screen, color,
                        (tetris_x + x * self.block_size, tetris_y + y * self.block_size,
                         self.block_size, self.block_size)
                    )
                    
                    # Draw block border
                    pygame.draw.rect(
                        self.screen, GRAY,
                        (tetris_x + x * self.block_size, tetris_y + y * self.block_size,
                         self.block_size, self.block_size), 1
                    )
        
        # Draw particles
        for particle in self.env.active_particles:
            x_pos = tetris_x + particle['x'] * self.block_size
            y_pos = tetris_y + particle['y'] * self.block_size
            
            # Fade out based on remaining lifetime
            alpha = int(255 * (particle['lifetime'] / 1.5))
            if alpha <= 0:
                continue
            
            # Draw particle
            pygame.draw.circle(
                self.screen, particle['color'],
                (int(x_pos), int(y_pos)),
                int(particle['size'])
            )
    
    def render_metrics(self):
        """Draw the training metrics"""
        # Game statistics area
        stats_x = 50
        stats_y = self.tetris_height + 70
        
        # Draw game stats
        pygame.draw.rect(self.screen, (30, 30, 30), 
                        (stats_x, stats_y, 300, 130))
        
        # Draw title
        title = self.big_font.render("Game Stats", True, WHITE)
        self.screen.blit(title, (stats_x + 10, stats_y + 10))
        
        # Draw stats
        stats_text = [
            f"Score: {self.env.score}",
            f"Lines: {self.env.lines_cleared}",
            f"Episode: {self.episode}/{self.episodes}",
            f"Max Score: {self.agent.max_score}",
            f"Avg Score (100): {self.agent.avg_score_100:.1f}"
        ]
        
        for i, text in enumerate(stats_text):
            text_surf = self.font.render(text, True, WHITE)
            self.screen.blit(text_surf, (stats_x + 15, stats_y + 40 + i * 20))
            
        # Draw training stats
        train_x = stats_x
        train_y = stats_y + 140
        
        # Background
        pygame.draw.rect(self.screen, (30, 30, 30), 
                        (train_x, train_y, 300, 100))
        
        # Title
        title = self.font.render("Training Status", True, WHITE)
        self.screen.blit(title, (train_x + 10, train_y + 10))
        
        # Status text
        active_text = "Active" if self.training_active else "Paused"
        status_text = self.font.render(f"Status: {active_text}", True, 
                                      (100, 255, 100) if self.training_active else (255, 100, 100))
        self.screen.blit(status_text, (train_x + 15, train_y + 35))
        
        # Duration
        if self.training_start_time:
            duration = time.time() - self.training_start_time + self.total_training_time
        else:
            duration = self.total_training_time
        
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        time_text = self.font.render(f"Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}", True, WHITE)
        self.screen.blit(time_text, (train_x + 15, train_y + 55))
        
        # Speed
        speed_text = self.font.render(f"FPS: {self.fps:.1f}", True, WHITE)
        self.screen.blit(speed_text, (train_x + 15, train_y + 75))
    
    def render_training_tab(self):
        """Render content for the Training tab"""
        # Already rendered via UI elements and metrics
        pass
    
    def render_metrics_tab(self):
        """Render content for the Metrics tab"""
        metrics_x = 420
        
        checkbox_y_position = 250  # Y position of the last row of checkboxes
        checkbox_height = 30  # Estimated height of checkboxes
        gap = 50  # Add some spacing for clarity

        metrics_y = checkbox_y_position + checkbox_height + gap  # Adjusted Y position
        metrics_x = 420  # Keep X position the same

        
        # Draw plots
        plot_width = 360
        plot_height = 140
        
        # Loss plot
        if len(self.agent.loss_history) > 0:
            self.plot_loss.update_data(self.agent.loss_history[-100:])
            self.plot_loss.draw(self.screen, metrics_x, metrics_y, plot_width, plot_height)
        
        # Epsilon plot
        if len(self.agent.epsilon_history) > 0:
            self.plot_epsilon.update_data(self.agent.epsilon_history[-100:])
            self.plot_epsilon.draw(self.screen, metrics_x + plot_width + 20, metrics_y, plot_width, plot_height)
        
        # Reward plot
        if len(self.agent.reward_history) > 0:
            self.plot_reward.update_data(self.agent.reward_history[-100:])
            self.plot_reward.draw(self.screen, metrics_x, metrics_y + plot_height + 20, plot_width, plot_height)
        
        # Score plot
        if len(self.agent.score_history) > 0:
            self.plot_score.update_data(self.agent.score_history[-100:])
            self.plot_score.draw(self.screen, metrics_x + plot_width + 20, metrics_y + plot_height + 20, plot_width, plot_height)
    
    def render_network_tab(self):
        """Render content for the Network tab"""
        network_x = 420
        
        checkbox_y_position = 250  # Y position of the last checkbox row
        checkbox_height = 30  # Estimated height of checkboxes
        gap = 50  # Additional spacing

        network_y = checkbox_y_position + checkbox_height + gap  # Move it down

        
        # Draw neural network structure
        nn_width = 740
        nn_height = 300
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), (network_x, network_y, nn_width, nn_height))
        
        # Draw layers
        layer_sizes = [self.state_size, 128, 64, self.action_size]
        layer_names = ["Input", "Hidden 1", "Hidden 2", "Output"]
        layer_colors = [(50, 150, 255), (50, 200, 50), (200, 200, 50), (255, 100, 50)]
        
        # Calculate positions
        layer_width = nn_width / len(layer_sizes)
        
        for i, (size, name, color) in enumerate(zip(layer_sizes, layer_names, layer_colors)):
            # Draw layer label
            label = self.font.render(f"{name} ({size})", True, WHITE)
            self.screen.blit(label, (network_x + i * layer_width + 20, network_y + 10))
            
            # Draw nodes (limit to max 10 visible nodes)
            visible_nodes = min(10, size)
            node_spacing = min(20, nn_height / (visible_nodes + 1))
            node_size = min(15, node_spacing * 0.8)
            
            for j in range(visible_nodes):
                node_y = network_y + 40 + j * node_spacing
                pygame.draw.circle(self.screen, color, 
                                  (int(network_x + i * layer_width + layer_width/2), int(node_y)), 
                                  int(node_size))
            
            # Add ellipsis if more nodes exist
            if size > visible_nodes:
                pygame.draw.circle(self.screen, color, 
                                  (int(network_x + i * layer_width + layer_width/2), 
                                   int(network_y + 40 + (visible_nodes+1) * node_spacing)), 
                                  int(node_size/2))
                pygame.draw.circle(self.screen, color, 
                                  (int(network_x + i * layer_width + layer_width/2), 
                                   int(network_y + 40 + (visible_nodes+2) * node_spacing)), 
                                  int(node_size/2))
                pygame.draw.circle(self.screen, color, 
                                  (int(network_x + i * layer_width + layer_width/2), 
                                   int(network_y + 40 + (visible_nodes+3) * node_spacing)), 
                                  int(node_size/2))
            
            # Draw connections to next layer (except for output layer)
            if i < len(layer_sizes) - 1:
                next_visible_nodes = min(10, layer_sizes[i+1])
                next_node_spacing = min(20, nn_height / (next_visible_nodes + 1))
                
                for j in range(visible_nodes):
                    node_y = network_y + 40 + j * node_spacing
                    
                    for k in range(next_visible_nodes):
                        next_node_y = network_y + 40 + k * next_node_spacing
                        
                        # Draw line with very low alpha for visualization
                        pygame.draw.line(self.screen, (100, 100, 100, 10), 
                                        (network_x + i * layer_width + layer_width/2, node_y),
                                        (network_x + (i+1) * layer_width + layer_width/2, next_node_y),
                                        1)
        
        # Render Q-values if available
        if self.show_q_values and self.agent.last_q_values is not None:
            q_data = self.agent.get_q_values_visualization()
            
            if q_data:
                # Draw Q-value distribution
                q_values_x = network_x
                q_values_y = network_y + nn_height + 20
                
                # Title
                title = self.font.render("Q-Values for Last Action", True, WHITE)
                self.screen.blit(title, (q_values_x, q_values_y))
                
                # Draw top 5 actions
                sorted_q = sorted(q_data, key=lambda x: x['q_value'], reverse=True)[:5]
                
                for i, data in enumerate(sorted_q):
                    # Determine color (green for best action, white for others)
                    color = (100, 255, 100) if data['is_best'] else WHITE
                    
                    # Action description
                    action_text = f"Rot={data['rotation']}, Pos={data['position']}"
                    q_text = f"Q={data['q_value']:.2f}"
                    
                    # Render text
                    action_surf = self.font.render(action_text, True, color)
                    q_surf = self.font.render(q_text, True, color)
                    
                    self.screen.blit(action_surf, (q_values_x + 20, q_values_y + 30 + i * 25))
                    self.screen.blit(q_surf, (q_values_x + 200, q_values_y + 30 + i * 25))
    
    def render_board_analysis_tab(self):
        """Render content for the Board Analysis tab"""
        analysis_x = 420
        
        checkbox_y_position = 250  # Y position of the last checkbox row
        checkbox_height = 30  # Estimated height of checkboxes
        gap = 50  # Additional spacing

        analysis_y = checkbox_y_position + checkbox_height + gap  # Move it down
    
        
        # Board metrics
        heights = self.env.get_heights()
        holes = self.env.count_holes()
        bumpiness = self.env.get_bumpiness()
        
        # Draw board analysis
        metrics_width = 360
        metrics_height = 210
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (analysis_x, analysis_y, metrics_width, metrics_height))
        
        # Title
        title = self.font.render("Board Analysis", True, WHITE)
        self.screen.blit(title, (analysis_x + 10, analysis_y + 10))
        
        # Metrics
        metrics_text = [
            f"Holes: {holes}",
            f"Bumpiness: {bumpiness}",
            f"Avg Height: {sum(heights)/len(heights):.2f}",
            f"Max Height: {max(heights)}",
            f"Last Action: {self.agent.get_action_explanation(self.agent.last_action)}",
            f"Move Type: {'Random (Explore)' if self.agent.was_random_action else 'Model (Exploit)'}"
        ]
        
        for i, text in enumerate(metrics_text):
            text_surf = self.font.render(text, True, WHITE)
            self.screen.blit(text_surf, (analysis_x + 15, analysis_y + 40 + i * 25))
        
        # Column heights visualization
        heights_x = analysis_x + metrics_width + 20
        heights_y = analysis_y
        heights_width = 360
        heights_height = 210
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (heights_x, heights_y, heights_width, heights_height))
        
        # Title
        title = self.font.render("Column Heights", True, WHITE)
        self.screen.blit(title, (heights_x + 10, heights_y + 10))
        
        # Draw column bars
        bar_margin = 10
        bar_width = (heights_width - 2 * bar_margin) / len(heights)
        max_bar_height = heights_height - 50
        
        for i, height in enumerate(heights):
            # Calculate bar properties
            bar_height = min(max_bar_height, (height / self.env.height) * max_bar_height)
            
            # Draw bar
            pygame.draw.rect(self.screen, VIBRANT_COLORS[i % len(VIBRANT_COLORS)], 
                            (heights_x + bar_margin + i * bar_width, 
                             heights_y + heights_height - 30 - bar_height, 
                             bar_width - 2, bar_height))
            
            # Draw height value
            if height > 0:
                height_text = self.small_font.render(str(height), True, WHITE)
                text_x = heights_x + bar_margin + i * bar_width + bar_width/2 - height_text.get_width()/2
                text_y = heights_y + heights_height - 30 - bar_height - 15
                self.screen.blit(height_text, (text_x, text_y))
            
            # Draw column number
            col_text = self.small_font.render(str(i), True, WHITE)
            self.screen.blit(col_text, (heights_x + bar_margin + i * bar_width + bar_width/2 - 4, 
                                      heights_y + heights_height - 20))
    
    def render_performance_tab(self):
        """Render content for the Performance tab"""
        perf_x = 420
        
        checkbox_y_position = 250  # Y position of the last checkbox row
        checkbox_height = 30  # Estimated height of checkboxes
        gap = 50  # Additional spacing

        perf_y = checkbox_y_position + checkbox_height + gap  # Move it down


        panel_width = 360
        panel_height = 250  # Increased from 200 to 230 for better text fit
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (perf_x, perf_y, panel_width, panel_height))
        
        # Title
        title = self.big_font.render("Performance Metrics", True, WHITE)
        self.screen.blit(title, (perf_x + 10, perf_y + 10))
        
        # Show GPU status
        from utils import setup_gpu
        device_info = setup_gpu()
        using_gpu = device_info['using_gpu']
        gpu_status = f"GPU: {'Enabled' if using_gpu else 'Disabled (using CPU)'}"
        gpu_text = self.font.render(gpu_status, True, (100, 255, 100) if using_gpu else (255, 100, 100))
        self.screen.blit(gpu_text, (perf_x + 15, perf_y + 45))
        
        # Display GPU details if available
        if using_gpu and device_info['gpu_names']:
            for i, name in enumerate(device_info['gpu_names']):
                gpu_name = self.small_font.render(f"GPU {i+1}: {name}", True, WHITE)
                self.screen.blit(gpu_name, (perf_x + 15, perf_y + 70 + i * 20))
            
            mem_growth = self.small_font.render(
                f"Memory Growth: {'Enabled' if device_info['memory_growth_enabled'] else 'Disabled'}", 
                True, WHITE)
            self.screen.blit(mem_growth, (perf_x + 15, perf_y + 70 + len(device_info['gpu_names']) * 20))
        
        # Timing information
        y_offset = perf_y + 140  # Increased from 120 to 140 for spacing
        
        # Get model performance stats
        stats = self.agent.get_performance_stats()
        
        timing_text = [
            f"Inference Time: {stats['avg_inference_time']:.2f}ms (max: {stats['max_inference_time']:.2f}ms)",
            f"Training Time: {stats['avg_train_time']:.2f}ms (max: {stats['max_train_time']:.2f}ms)",
            f"Episodes/sec: {self.episodes_per_second:.2f}",
            f"Total Steps: {self.total_steps}"
        ]
        
        for i, text in enumerate(timing_text):
            text_surf = self.font.render(text, True, WHITE)
            self.screen.blit(text_surf, (perf_x + 15, y_offset + i * 25))
        
        # Draw second panel
        perf2_x = perf_x + panel_width + 20
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (perf2_x, perf_y, panel_width, panel_height))
        
        # Title
        title = self.font.render("Training Resources", True, WHITE)
        self.screen.blit(title, (perf2_x + 10, perf_y + 10))
        
        # Memory usage
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
        
        memory_text = self.font.render(f"Memory Usage: {memory_usage:.1f} MB", True, WHITE)
        self.screen.blit(memory_text, (perf2_x + 15, perf_y + 45))
        
        # Experience replay buffer size
        buffer_text = self.font.render(f"Replay Buffer: {len(self.agent.memory)}/{self.agent.memory.maxlen}", True, WHITE)
        self.screen.blit(buffer_text, (perf2_x + 15, perf_y + 70))
        
        # Buffer memory usage (estimate)
        if len(self.agent.memory) > 0:
            # Estimate size of one experience in memory
            import sys
            sample = self.agent.memory[0]
            sample_size = (
                sys.getsizeof(sample[0]) +  # state
                sys.getsizeof(sample[1]) +  # action
                sys.getsizeof(sample[2]) +  # reward
                sys.getsizeof(sample[3]) +  # next_state
                sys.getsizeof(sample[4])    # done
            )
            buffer_memory = (sample_size * len(self.agent.memory)) / (1024 * 1024)  # in MB
            
            buffer_mem_text = self.font.render(f"Buffer Memory: {buffer_memory:.1f} MB", True, WHITE)
            self.screen.blit(buffer_mem_text, (perf2_x + 15, perf_y + 95))
    
    def render_web_export_tab(self):
        """Render content for the Web Export tab"""
        export_x = 420
        # Increase export_y to move the panel downward (e.g., from 290 to 350 or higher)
        export_y = 350  # Adjusted to provide more space below the checkboxes
        panel_width = 740
        panel_height = 300
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (export_x, export_y, panel_width, panel_height))
        
        # Title
        title = self.big_font.render("Web Export Settings", True, WHITE)
        self.screen.blit(title, (export_x + 10, export_y + 10))
        
        # Check if tensorflowjs is available
        if not TFJS_AVAILABLE:
            error_text = self.font.render(
                "TensorFlow.js converter not installed. Install with: pip install tensorflowjs", 
                True, (255, 100, 100))
            self.screen.blit(error_text, (export_x + 15, export_y + 50))
            return
        
        # Web export information
        info_text = [
            "Export your model to use in the HTML implementation:",
            "",
            "1. Train your model to achieve good performance",
            "2. Save the model using 'Save Model' or auto-save feature",
            "3. Export for web using 'Export for Web' button",
            "4. The model will be converted to TensorFlow.js format",
            "5. Use the exported model in the HTML implementation by uploading the model.json file",
            "",
            "The HTML implementation can be found in the 'models/tetris_dqn/web_model_latest/' directory",
            "after exporting.",
            "",
            f"Last export: {datetime.fromtimestamp(self.last_web_export_time).strftime('%Y-%m-%d %H:%M:%S') if self.last_web_export_time else 'Never'}"
        ]
        
        for i, text in enumerate(info_text):
            text_surf = self.font.render(text, True, WHITE)
            self.screen.blit(text_surf, (export_x + 15, export_y + 50 + i * 20))

    def render_ui_elements(self):
        """Draw all UI elements"""
        for element in self.ui_elements:
            element.draw(self.screen)
    
    def render_current_tab(self):
        """Render content for the currently selected tab"""
        active_tab = self.tab_manager.get_active_tab()
        
        if active_tab == 0:  # Training tab
            self.render_training_tab()
        elif active_tab == 1:  # Metrics tab
            self.render_metrics_tab()
        elif active_tab == 2:  # Network tab
            self.render_network_tab()
        elif active_tab == 3:  # Board Analysis tab
            self.render_board_analysis_tab()
        elif active_tab == 4:  # Performance tab
            self.render_performance_tab()
        elif active_tab == 5:  # Web Export tab
            self.render_web_export_tab()
    
    def render(self):
        """Render the entire visualization"""
        # Skip frames if needed
        if self.frame_skip > 0:
            self.current_frame = (self.current_frame + 1) % (self.frame_skip + 1)
            if self.current_frame != 0 and self.training_active:
                return
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
        
        # Clear the screen
        self.screen.fill((30, 30, 30))
        
        # Draw tab bar
        self.tab_manager.draw(self.screen)
        
        # Draw components
        self.render_tetris()
        self.render_metrics()
        self.render_ui_elements()
        self.render_current_tab()
        
        # Update the display
        pygame.display.flip()
    
    def train_step(self):
        """Perform one step of training"""
        if (not self.training_active and not self.do_step) or (self.step_mode and not self.do_step):
            if self.env.game_over:
                # Auto-save model every 100 episodes if enabled
                if self.auto_save_enabled and self.episode > 0:
                    if self.episode % 100 == 0:
                        self.save_model_auto()
                    if self.episode % 1000 == 0:
                        self.save_milestone()
                
                # Update agent score
                self.agent.update_score(self.env.score)
                
                # Start a new episode
                self.reset_game()
                self.episode += 1
                
                # FIXED: Make sure we can see the new game state
                self.render()
                
                # Update target network periodically
                if self.episode % self.target_update_freq == 0:
                    self.agent.update_target_model()
                    print(f"Episode {self.episode}: Target network updated")
                
                # Print progress
                if self.episode % 10 == 0:
                    print(f"Episode {self.episode}/{self.episodes}, " +
                        f"Score: {self.env.score}, " +
                        f"Epsilon: {self.agent.epsilon:.4f}, " +
                        f"Memory: {len(self.agent.memory)}/{self.agent.memory.maxlen}")
            
            # Reset do_step flag
            self.do_step = False
            return
        
        # Select action
        action = self.agent.act(self.state)
        
        # Take action
        next_state, reward, done, info = self.env.step(action)
        
        # Remember the experience
        self.agent.remember(self.state, action, reward, next_state, done)
        
        # Move to the next state
        self.state = next_state
        self.step_count += 1
        self.total_steps += 1
        
        # Train the model
        if len(self.agent.memory) > self.agent.batch_size:
            self.agent.replay(self.agent.batch_size)
        
        # Handle game over when training is active
        if self.env.game_over:
            # Update agent score
            self.agent.update_score(self.env.score)
            
            # Start a new episode
            self.reset_game()
            self.episode += 1
            
            # Update target network periodically
            if self.episode % self.target_update_freq == 0:
                self.agent.update_target_model()
                print(f"Episode {self.episode}: Target network updated")
                
            # Print progress
            if self.episode % 10 == 0:
                print(f"Episode {self.episode}/{self.episodes}, " +
                    f"Score: {self.env.score}, " +
                    f"Epsilon: {self.agent.epsilon:.4f}, " +
                    f"Memory: {len(self.agent.memory)}/{self.agent.memory.maxlen}")
            
            # Auto-save if enabled
            if self.auto_save_enabled and self.episode > 0:
                if self.episode % 100 == 0:
                    self.save_model_auto()
                if self.episode % 1000 == 0:
                    self.save_milestone()
        
        # Reset do_step flag
        self.do_step = False
    
    def toggle_training(self):
        """Toggle training state"""
        self.training_active = not self.training_active
        
        # Update button text
        if self.training_active:
            self.update_ui_button_text("Start Training", "Pause Training")
            self.update_ui_button_text("Resume", "Pause Training")
            
            # Record start time if first start
            if self.training_start_time is None:
                self.training_start_time = time.time()
                self.training_sessions += 1
        else:
            # Add to total training time
            if self.training_start_time is not None:
                self.total_training_time += time.time() - self.training_start_time
                self.training_start_time = None
            
            self.update_ui_button_text("Pause Training", "Resume")
    
    def save_model(self):
        """Save the model weights"""
        # Create output directory
        output_dir = "models/tetris_dqn/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = f"{output_dir}tetris_dqn_weights_{self.episode}.h5"
        self.agent.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load the model weights"""
        # Use Pygame file dialog (compatible with all platforms)
        # For simplicity, we'll just use terminal input here
        model_path = input("Enter the path to the model weights file: ")
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"File not found: {model_path}")
    
    def reset_training(self):
        """Reset training completely"""
        # Stop training if active
        self.training_active = False
        
        # Reset environment and agent
        self.env = TetrisEnv()
        self.agent = DQNAgent(self.state_size, self.action_size)
        
        # Reset episode counter
        self.episode = 0
        
        # Reset game
        self.reset_game()
        
        # Reset timing stats
        self.training_start_time = None
        self.total_training_time = 0
        self.episodes_per_second = 0
        self.steps_per_second = 0
        self.total_steps = 0
        
        # Update UI button text
        self.update_ui_button_text("Pause Training", "Start Training")
        self.update_ui_button_text("Resume", "Start Training")
        
        print("Training reset")
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Handle UI element events
            handled = False
            for element in self.ui_elements:
                if element.handle_event(event):
                    handled = True
                    break
            
            # Handle tab manager events
            if not handled:
                self.tab_manager.handle_event(event)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                if event.key == pygame.K_SPACE:
                    # Toggle training
                    self.toggle_training()
                
                if event.key == pygame.K_s:
                    # Save model
                    self.save_model()
                
                if event.key == pygame.K_r:
                    # Reset training
                    self.reset_training()
                
                if event.key == pygame.K_n:
                    # Single step (when in step mode)
                    self.do_single_step()
        
        return True
    
    def run(self):
        """Main loop for the visualization"""
        running = True
        clock = pygame.time.Clock()
        
        self.reset_game()
        self.last_update_time = time.time()
        self.last_fps_update = time.time()
        
        # Track episode time for calculating episodes per second
        last_episode = self.episode
        last_episode_time = time.time()
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Calculate delta time
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
            
            # Update particle effects
            self.env.update_particles(dt)
            
            # Process training steps - FIXED: Only train when appropriate
            if self.training_active or self.do_step:
                self.train_step()
            
            # Calculate episodes per second
            if self.episode > last_episode:
                episodes_elapsed = self.episode - last_episode
                time_elapsed = current_time - last_episode_time
                
                if time_elapsed > 0:
                    self.episodes_per_second = 0.9 * self.episodes_per_second + 0.1 * (episodes_elapsed / time_elapsed)
                
                last_episode = self.episode
                last_episode_time = current_time
            
            # Render everything
            self.render()
            
            # Control frame rate
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

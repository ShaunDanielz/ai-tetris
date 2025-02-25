import numpy as np
import tensorflow as tf
import random
import time
import os
import sys
import pygame
import threading
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
from datetime import datetime

# Configure GPU usage
def setup_gpu():
    """Configure GPU usage and return information about available devices"""
    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    device_info = {
        'gpus_available': len(gpus),
        'gpu_names': [],
        'using_gpu': False,
        'memory_growth_enabled': False
    }
    
    if gpus:
        try:
            # Get GPU details
            for gpu in gpus:
                device_info['gpu_names'].append(gpu.name)
            
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            device_info['memory_growth_enabled'] = True
            
            # By default, TensorFlow will use GPU if available, so we just need to confirm
            device_info['using_gpu'] = True
            
            print(f"GPU acceleration enabled. Found {len(gpus)} GPU(s):")
            for i, gpu_name in enumerate(device_info['gpu_names']):
                print(f"  {i+1}. {gpu_name}")
                
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            print("Falling back to CPU")
            # Disable all GPUs if there was an error
            tf.config.set_visible_devices([], 'GPU')
            device_info['using_gpu'] = False
    else:
        print("No GPU found. Using CPU for training (slower).")
        print("Available CPUs:", len(cpus))
    
    return device_info

# Initialize GPU/CPU configuration
device_info = setup_gpu()

# Define colorblind-friendly colors
COLORBLIND_COLORS = [
    (0, 114, 178),    # Blue
    (230, 159, 0),    # Orange/Amber
    (0, 158, 115)     # Green
]

# More vibrant versions for UI
VIBRANT_COLORS = [
    (0, 144, 238),    # Bright Blue
    (255, 179, 0),    # Bright Orange
    (0, 198, 145)     # Bright Green
]

# Tetromino shapes
TETROMINOS = {
    'I': {
        'shape': np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
    },
    'J': {
        'shape': np.array([
            [0, 0, 0],
            [2, 2, 2],
            [0, 0, 2]
        ])
    },
    'L': {
        'shape': np.array([
            [0, 0, 0],
            [3, 3, 3],
            [3, 0, 0]
        ])
    },
    'O': {
        'shape': np.array([
            [4, 4],
            [4, 4]
        ])
    },
    'S': {
        'shape': np.array([
            [0, 0, 0],
            [0, 5, 5],
            [5, 5, 0]
        ])
    },
    'T': {
        'shape': np.array([
            [0, 0, 0],
            [6, 6, 6],
            [0, 6, 0]
        ])
    },
    'Z': {
        'shape': np.array([
            [0, 0, 0],
            [7, 7, 0],
            [0, 7, 7]
        ])
    }
}

class TetrisEnv:
    def __init__(self):
        self.width = 10
        self.height = 20
        self.board = None
        self.current_piece = None
        self.current_pos = None
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.piece_colors = {}  # Maps piece values to colors
        self.active_particles = []  # For line clear animations
        self.ghost_position = {'x': 0, 'y': 0}  # For ghost piece (landing preview)
        self.reset()
    
    def reset(self):
        """Reset the game state"""
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.piece_colors = {}  # Reset piece colors
        self.active_particles = []
        self._new_piece()
        return self.get_state()
    
    def _new_piece(self):
        """Generate a new random piece"""
        tetromino_keys = list(TETROMINOS.keys())
        random_key = random.choice(tetromino_keys)
        self.current_piece = TETROMINOS[random_key]['shape'].copy()
        
        # Start position
        self.current_pos = {
            'x': self.width // 2 - len(self.current_piece[0]) // 2,
            'y': 0
        }
        
        # Assign a random color to this piece
        piece_value = np.max(self.current_piece)
        if piece_value > 0 and piece_value not in self.piece_colors:
            self.piece_colors[piece_value] = random.choice(COLORBLIND_COLORS)
        
        # Update ghost piece position
        self._update_ghost_position()
        
        # Check if game is over (collision on spawn)
        if self._check_collision(0, 0):
            self.game_over = True
    
    def _check_collision(self, dx=0, dy=0):
        """Check if the current piece would collide at the given offset"""
        for y in range(len(self.current_piece)):
            for x in range(len(self.current_piece[y])):
                if self.current_piece[y][x] == 0:
                    continue
                
                new_x = self.current_pos['x'] + x + dx
                new_y = self.current_pos['y'] + y + dy
                
                # Check if out of bounds
                if (new_x < 0 or new_x >= self.width or
                    new_y >= self.height):
                    return True
                
                # Check if collision with placed pieces
                if new_y >= 0 and self.board[new_y][new_x] != 0:
                    return True
        
        return False
    
    def _update_ghost_position(self):
        """Update the ghost piece position (shows where piece will land)"""
        # Start at current position
        self.ghost_position = {'x': self.current_pos['x'], 'y': self.current_pos['y']}
        
        # Drop as far as possible
        while not self._check_collision(0, self.ghost_position['y'] - self.current_pos['y'] + 1):
            self.ghost_position['y'] += 1
    
    def _rotate_piece(self, clockwise=True):
        """Rotate the current piece"""
        # Make a copy of the current piece
        original_piece = self.current_piece.copy()
        
        # Transpose
        self.current_piece = np.transpose(self.current_piece)
        
        # Reverse rows/cols
        if clockwise:
            for i in range(len(self.current_piece)):
                self.current_piece[i] = self.current_piece[i][::-1]
        else:
            self.current_piece = self.current_piece[::-1]
        
        # Check if rotation causes collision
        if self._check_collision():
            # Revert rotation
            self.current_piece = original_piece
            return False
        
        # Update ghost position after rotation
        self._update_ghost_position()
        return True
    
    def _merge_piece(self):
        """Merge the current piece into the board"""
        for y in range(len(self.current_piece)):
            for x in range(len(self.current_piece[y])):
                if self.current_piece[y][x] != 0:
                    self.board[self.current_pos['y'] + y][self.current_pos['x'] + x] = self.current_piece[y][x]
    
    def _clear_lines(self):
        """Clear completed lines and return the number cleared"""
        lines_cleared = 0
        lines_to_clear = []
        
        # Find lines to clear
        for y in range(self.height):
            if all(self.board[y]):
                lines_to_clear.append(y)
                lines_cleared += 1
        
        # Create particles for cleared lines
        for y in lines_to_clear:
            for x in range(self.width):
                # Create 2 particles per block for a more dense effect
                for _ in range(2):
                    self._add_particle(x, y, self.board[y][x])
        
        # Remove the lines
        for y in lines_to_clear:
            # Remove the line
            self.board = np.delete(self.board, y, axis=0)
            # Add a new empty line at the top
            self.board = np.vstack([np.zeros(self.width, dtype=int), self.board])
        
        return lines_cleared
    
    def _add_particle(self, x, y, piece_value):
        """Add a particle effect at the given position"""
        color = self.piece_colors.get(piece_value, COLORBLIND_COLORS[0])
        
        # Random velocity for the particle
        vx = random.uniform(-2, 2)
        vy = random.uniform(-5, -2)  # Up direction
        
        # Create particle (x, y, vx, vy, color, lifetime)
        self.active_particles.append({
            'x': x, 
            'y': y, 
            'vx': vx, 
            'vy': vy, 
            'color': color, 
            'lifetime': random.uniform(0.5, 1.5),
            'size': random.uniform(2, 6)
        })
    
    def update_particles(self, dt):
        """Update particle positions and remove expired ones"""
        gravity = 9.8
        
        for particle in self.active_particles[:]:
            # Update position
            particle['x'] += particle['vx'] * dt
            particle['y'] += particle['vy'] * dt
            
            # Apply gravity
            particle['vy'] += gravity * dt
            
            # Reduce lifetime
            particle['lifetime'] -= dt
            
            # Remove expired particles
            if particle['lifetime'] <= 0:
                self.active_particles.remove(particle)
    
    def get_heights(self):
        """Get the height of each column"""
        heights = []
        for x in range(self.width):
            y = 0
            while y < self.height and self.board[y][x] == 0:
                y += 1
            heights.append(self.height - y)
        return heights
    
    def count_holes(self):
        """Count holes (empty cells with blocks above them)"""
        holes = 0
        for x in range(self.width):
            block_found = False
            for y in range(self.height):
                if self.board[y][x] != 0:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes
    
    def get_bumpiness(self):
        """Calculate the bumpiness (sum of differences between adjacent columns)"""
        heights = self.get_heights()
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness
    
    def get_state(self):
        """Convert the game state to a feature vector"""
        # Flatten the board
        flat_board = (self.board > 0).astype(float).flatten()
        
        # Encode current piece type (one-hot)
        piece_type = -1
        for i, key in enumerate(TETROMINOS.keys()):
            if np.array_equal(TETROMINOS[key]['shape'], self.current_piece):
                piece_type = i
                break
        
        piece_encoding = np.zeros(4)
        if piece_type >= 0:
            # Binary encoding
            piece_encoding[0] = 1 if (piece_type & 1) else 0
            piece_encoding[1] = 1 if (piece_type & 2) else 0
            piece_encoding[2] = 1 if (piece_type & 4) else 0
            piece_encoding[3] = self.current_pos['x'] / self.width  # Normalized x position
        
        return np.concatenate([flat_board, piece_encoding])
    
    def step(self, action):
        """
        Take an action in the environment
        
        Action is an integer 0-39:
        - rotation: action // 10 (0-3)
        - position: action % 10 (0-9)
        
        Returns:
        - next_state: new state after action
        - reward: reward for the action
        - done: whether the game is over
        - info: additional information
        """
        if self.game_over:
            return self.get_state(), -100, True, {}
        
        # Save the initial state for reward calculation
        initial_board = self.board.copy()
        initial_heights = self.get_heights()
        initial_holes = self.count_holes()
        
        # Decode action
        rotation = (action // 10) % 4
        position = action % 10
        
        # Perform rotation
        for _ in range(rotation):
            self._rotate_piece()
        
        # Move to target x position
        current_x = self.current_pos['x']
        target_x = position
        
        # Adjust for piece width
        piece_width = len(self.current_piece[0])
        if target_x + piece_width > self.width:
            target_x = self.width - piece_width
        
        # Move horizontally
        dx = target_x - current_x
        if dx < 0:
            for _ in range(abs(dx)):
                if not self._check_collision(-1, 0):
                    self.current_pos['x'] -= 1
                    self._update_ghost_position()
                else:
                    break
        elif dx > 0:
            for _ in range(dx):
                if not self._check_collision(1, 0):
                    self.current_pos['x'] += 1
                    self._update_ghost_position()
                else:
                    break
        
        # Drop the piece all the way down
        while not self._check_collision(0, 1):
            self.current_pos['y'] += 1
        
        # Merge piece and check for cleared lines
        self._merge_piece()
        lines_cleared = self._clear_lines()
        
        # Update score
        if lines_cleared > 0:
            self.score += (1 << lines_cleared) * 100  # 100, 200, 400, 800
            self.lines_cleared += lines_cleared
        
        # Get new piece
        self._new_piece()
        
        # Calculate reward
        reward = self._calculate_reward(initial_board, initial_heights, initial_holes, lines_cleared)
        
        return self.get_state(), reward, self.game_over, {'lines_cleared': lines_cleared}
    
    def _calculate_reward(self, initial_board, initial_heights, initial_holes, lines_cleared):
        """Calculate reward based on state change"""
        reward = 0
        
        # Base reward for surviving
        reward += 0.1
        
        # Reward for clearing lines
        if lines_cleared > 0:
            reward += (1 << lines_cleared) * 10  # 10, 20, 40, 80
        
        # Penalty for game over
        if self.game_over:
            reward -= 100
            return reward
        
        # Calculate changes in board metrics
        current_heights = self.get_heights()
        current_holes = self.count_holes()
        
        # Penalize increase in height
        avg_height_before = sum(initial_heights) / len(initial_heights)
        avg_height_after = sum(current_heights) / len(current_heights)
        if avg_height_after > avg_height_before:
            reward -= (avg_height_after - avg_height_before) * 0.5
        
        # Penalize creating holes
        if current_holes > initial_holes:
            reward -= (current_holes - initial_holes) * 2
        
        return reward
    
    def get_board_with_piece(self):
        """Return a copy of the board with the current piece included"""
        board_with_piece = self.board.copy()
        
        # Add current piece to board
        for y in range(len(self.current_piece)):
            for x in range(len(self.current_piece[y])):
                if self.current_piece[y][x] != 0:
                    py = self.current_pos['y'] + y
                    px = self.current_pos['x'] + x
                    if 0 <= py < self.height and 0 <= px < self.width:
                        board_with_piece[py][px] = self.current_piece[y][x]
        
        return board_with_piece
    
    def get_heatmap_data(self):
        """Generate a heatmap showing dangerous areas (higher is more dangerous)"""
        heatmap = np.zeros((self.height, self.width))
        
        # Get heights of each column
        heights = []
        for x in range(self.width):
            y = 0
            while y < self.height and self.board[y][x] == 0:
                y += 1
            heights.append(y)
        
        # Add danger to holes (empty cells with blocks above them)
        for x in range(self.width):
            block_found = False
            for y in range(self.height):
                if self.board[y][x] != 0:
                    block_found = True
                elif block_found:
                    heatmap[y][x] += 5  # Holes are dangerous
        
        # Add danger to tall columns
        avg_height = sum(heights) / len(heights)
        for x in range(self.width):
            if heights[x] < avg_height:  # Lower value means higher stack
                danger = (avg_height - heights[x]) * 0.5
                for y in range(self.height):
                    if y >= heights[x]:
                        heatmap[y][x] += danger
        
        # Add danger to bumpy areas
        for x in range(1, self.width):
            bumpiness = abs(heights[x] - heights[x-1])
            for y in range(self.height):
                if min(heights[x], heights[x-1]) <= y < max(heights[x], heights[x-1]):
                    heatmap[y][x] += bumpiness * 0.3
        
        return heatmap


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Metrics tracking
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        self.score_history = []
        self.avg_score_100 = 0
        self.avg_reward_100 = 0
        self.max_score = 0
        self.total_games = 0
        self.last_action = None
        self.was_random_action = False
        self.last_q_values = None
        
        # Track performance
        self.inference_times = []  # Track time taken for predictions
        self.train_times = []      # Track time taken for training
    
    def _build_model(self):
        """Build a deep neural network model"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Update reward history
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]
        
        # Update average reward
        if len(self.reward_history) > 0:
            self.avg_reward_100 = np.mean(self.reward_history[-100:])
    
    def act(self, state, training=True):
        """Return action based on current state"""
        self.was_random_action = False
        
        if training and np.random.rand() <= self.epsilon:
            self.was_random_action = True
            action = random.randrange(self.action_size)
            self.last_q_values = None
        else:
            # Measure inference time
            start_time = time.time()
            
            act_values = self.model.predict(state.reshape(1, -1), verbose=0)
            
            # Record inference time
            end_time = time.time()
            self.inference_times.append(end_time - start_time)
            
            self.last_q_values = act_values[0]
            action = np.argmax(act_values[0])
        
        self.last_action = action
        return action
    
    def replay(self, batch_size):
        """Train the model with experiences from memory"""
        if len(self.memory) < batch_size:
            return None
        
        # Measure training time
        start_time = time.time()
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values for current states
        target = self.model.predict(states, verbose=0)
        
        # Get Q-values for next states from target model
        target_next = self.target_model.predict(next_states, verbose=0)
        
        # Update target Q-values with Bellman equation
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
        
        # Train the model
        history = self.model.fit(states, target, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Record training time
        end_time = time.time()
        self.train_times.append(end_time - start_time)
        
        # Update loss history
        self.loss_history.append(loss)
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon_history.append(self.epsilon)
        
        return loss
    
    def update_score(self, score):
        """Update score metrics"""
        self.score_history.append(score)
        if len(self.score_history) > 1000:
            self.score_history = self.score_history[-1000:]
        
        self.total_games += 1
        
        if score > self.max_score:
            self.max_score = score
        
        self.avg_score_100 = np.mean(self.score_history[-100:])
    
    def get_action_explanation(self, action):
        """Provide explanation of what an action does"""
        rotation = action // 10
        position = action % 10
        return f"Rotate {rotation} times, move to column {position}"
    
    def get_q_values_visualization(self):
        """Get visualization data for Q-values"""
        if self.last_q_values is None:
            return None
        
        q_data = []
        for i, q_value in enumerate(self.last_q_values):
            rotation = i // 10
            position = i % 10
            q_data.append({
                'action': i,
                'rotation': rotation,
                'position': position,
                'q_value': q_value,
                'is_best': (i == self.last_action)
            })
        
        return q_data
    
    def get_performance_stats(self):
        """Get performance statistics for model inference and training"""
        stats = {}
        
        if self.inference_times:
            stats['avg_inference_time'] = np.mean(self.inference_times[-100:]) * 1000  # in ms
            stats['max_inference_time'] = np.max(self.inference_times[-100:]) * 1000   # in ms
        else:
            stats['avg_inference_time'] = 0
            stats['max_inference_time'] = 0
            
        if self.train_times:
            stats['avg_train_time'] = np.mean(self.train_times[-100:]) * 1000  # in ms
            stats['max_train_time'] = np.max(self.train_times[-100:]) * 1000   # in ms
        else:
            stats['avg_train_time'] = 0
            stats['max_train_time'] = 0
            
        return stats
    
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name):
        """Save model weights"""
        self.model.save_weights(name)


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


class VisualTetrisDQNTrainer:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        
        # Screen dimensions
        self.window_width = 1280
        self.window_height = 800
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
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.GRID_COLOR = (50, 50, 50)
        
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
        
        # Try to load latest model at startup
        self.load_latest_model()
    
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
        self.ui_elements.append(CheckBox(600, 230, 20, "Auto-Save Every 100 Episodes", True, 
                                       self.toggle_autosave))
        
        # Frame skip slider (for speed)
        self.ui_elements.append(Slider(800, 230, 250, 10, 0, 10, 0, 
                                      "Frame Skip", self.set_frame_skip))
    
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
                import json
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
                    import json
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
        pygame.draw.rect(self.screen, self.BLACK, (tetris_x, tetris_y, self.tetris_width, self.tetris_height))
        
        # Draw the grid lines
        for x in range(self.env.width + 1):
            pygame.draw.line(
                self.screen, self.GRID_COLOR,
                (tetris_x + x * self.block_size, tetris_y),
                (tetris_x + x * self.block_size, tetris_y + self.tetris_height)
            )
        
        for y in range(self.env.height + 1):
            pygame.draw.line(
                self.screen, self.GRID_COLOR,
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
                        color = random.choice(COLORBLIND_COLORS)
                        self.env.piece_colors[piece_value] = color
                    
                    pygame.draw.rect(
                        self.screen, color,
                        (tetris_x + x * self.block_size, tetris_y + y * self.block_size,
                         self.block_size, self.block_size)
                    )
                    
                    # Draw block border
                    pygame.draw.rect(
                        self.screen, self.GRAY,
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
        title = self.big_font.render("Game Stats", True, self.WHITE)
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
            text_surf = self.font.render(text, True, self.WHITE)
            self.screen.blit(text_surf, (stats_x + 15, stats_y + 40 + i * 20))
            
        # Draw training stats
        train_x = stats_x
        train_y = stats_y + 140
        
        # Background
        pygame.draw.rect(self.screen, (30, 30, 30), 
                        (train_x, train_y, 300, 100))
        
        # Title
        title = self.font.render("Training Status", True, self.WHITE)
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
        
        time_text = self.font.render(f"Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}", True, self.WHITE)
        self.screen.blit(time_text, (train_x + 15, train_y + 55))
        
        # Speed
        speed_text = self.font.render(f"FPS: {self.fps:.1f}", True, self.WHITE)
        self.screen.blit(speed_text, (train_x + 15, train_y + 75))
    
    def render_training_tab(self):
        """Render content for the Training tab"""
        # Already rendered via UI elements and metrics
        pass
    
    def render_metrics_tab(self):
        """Render content for the Metrics tab"""
        metrics_x = 420
        metrics_y = 260
        
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
        network_y = 260
        
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
            label = self.font.render(f"{name} ({size})", True, self.WHITE)
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
                title = self.font.render("Q-Values for Last Action", True, self.WHITE)
                self.screen.blit(title, (q_values_x, q_values_y))
                
                # Draw top 5 actions
                sorted_q = sorted(q_data, key=lambda x: x['q_value'], reverse=True)[:5]
                
                for i, data in enumerate(sorted_q):
                    # Determine color (green for best action, white for others)
                    color = (100, 255, 100) if data['is_best'] else self.WHITE
                    
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
        analysis_y = 260
        
        # Board metrics
        heights = self.env.get_heights()
        holes = self.env.count_holes()
        bumpiness = self.env.get_bumpiness()
        
        # Draw board analysis
        metrics_width = 360
        metrics_height = 180
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (analysis_x, analysis_y, metrics_width, metrics_height))
        
        # Title
        title = self.font.render("Board Analysis", True, self.WHITE)
        self.screen.blit(title, (analysis_x + 10, analysis_y + 10))
        
        # Metrics
        metrics_text = [
            f"Holes: {holes}",
            f"Bumpiness: {bumpiness}",
            f"Avg Height: {sum(heights)/len(heights):.2f}",
            f"Max Height: {max(heights)}",
            f"Last Action: {self.agent.get_action_explanation(self.agent.last_action) if self.agent.last_action is not None else 'None'}",
            f"Move Type: {'Random (Explore)' if self.agent.was_random_action else 'Model (Exploit)'}"
        ]
        
        for i, text in enumerate(metrics_text):
            text_surf = self.font.render(text, True, self.WHITE)
            self.screen.blit(text_surf, (analysis_x + 15, analysis_y + 40 + i * 25))
        
        # Column heights visualization
        heights_x = analysis_x + metrics_width + 20
        heights_y = analysis_y
        heights_width = 360
        heights_height = 180
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (heights_x, heights_y, heights_width, heights_height))
        
        # Title
        title = self.font.render("Column Heights", True, self.WHITE)
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
                height_text = self.small_font.render(str(height), True, self.WHITE)
                text_x = heights_x + bar_margin + i * bar_width + bar_width/2 - height_text.get_width()/2
                text_y = heights_y + heights_height - 30 - bar_height - 15
                self.screen.blit(height_text, (text_x, text_y))
            
            # Draw column number
            col_text = self.small_font.render(str(i), True, self.WHITE)
            self.screen.blit(col_text, (heights_x + bar_margin + i * bar_width + bar_width/2 - 4, 
                                      heights_y + heights_height - 20))
    
    def render_performance_tab(self):
        """Render content for the Performance tab"""
        perf_x = 420
        perf_y = 260
        panel_width = 360
        panel_height = 200
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (perf_x, perf_y, panel_width, panel_height))
        
        # Title
        title = self.big_font.render("Performance Metrics", True, self.WHITE)
        self.screen.blit(title, (perf_x + 10, perf_y + 10))
        
        # Show GPU status
        using_gpu = device_info['using_gpu']
        gpu_status = f"GPU: {'Enabled' if using_gpu else 'Disabled (using CPU)'}"
        gpu_text = self.font.render(gpu_status, True, (100, 255, 100) if using_gpu else (255, 100, 100))
        self.screen.blit(gpu_text, (perf_x + 15, perf_y + 45))
        
        # Display GPU details if available
        if using_gpu and device_info['gpu_names']:
            for i, name in enumerate(device_info['gpu_names']):
                gpu_name = self.small_font.render(f"GPU {i+1}: {name}", True, self.WHITE)
                self.screen.blit(gpu_name, (perf_x + 15, perf_y + 70 + i * 20))
            
            mem_growth = self.small_font.render(
                f"Memory Growth: {'Enabled' if device_info['memory_growth_enabled'] else 'Disabled'}", 
                True, self.WHITE)
            self.screen.blit(mem_growth, (perf_x + 15, perf_y + 70 + len(device_info['gpu_names']) * 20))
        
        # Timing information
        y_offset = perf_y + 120
        
        # Get model performance stats
        stats = self.agent.get_performance_stats()
        
        timing_text = [
            f"Inference Time: {stats['avg_inference_time']:.2f}ms (max: {stats['max_inference_time']:.2f}ms)",
            f"Training Time: {stats['avg_train_time']:.2f}ms (max: {stats['max_train_time']:.2f}ms)",
            f"Episodes/sec: {self.episodes_per_second:.2f}",
            f"Total Steps: {self.total_steps}"
        ]
        
        for i, text in enumerate(timing_text):
            text_surf = self.font.render(text, True, self.WHITE)
            self.screen.blit(text_surf, (perf_x + 15, y_offset + i * 25))
        
        # Draw second panel
        perf2_x = perf_x + panel_width + 20
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (perf2_x, perf_y, panel_width, panel_height))
        
        # Title
        title = self.font.render("Training Resources", True, self.WHITE)
        self.screen.blit(title, (perf2_x + 10, perf_y + 10))
        
        # Memory usage
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
        
        memory_text = self.font.render(f"Memory Usage: {memory_usage:.1f} MB", True, self.WHITE)
        self.screen.blit(memory_text, (perf2_x + 15, perf_y + 45))
        
        # Experience replay buffer size
        buffer_text = self.font.render(f"Replay Buffer: {len(self.agent.memory)}/{self.agent.memory.maxlen}", True, self.WHITE)
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
            
            buffer_mem_text = self.font.render(f"Buffer Memory: {buffer_memory:.1f} MB", True, self.WHITE)
            self.screen.blit(buffer_mem_text, (perf2_x + 15, perf_y + 95))
    
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
            
            # Process training steps
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


if __name__ == "__main__":
    # Create and run the visualizer
    visualizer = VisualTetrisDQNTrainer()
    visualizer.run()

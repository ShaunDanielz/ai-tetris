# tetris_env.py
"""
Tetris Environment for AI Training.

This module implements a Tetris game environment designed for reinforcement
learning. It handles the game state, rules, piece movement, collision detection,
and provides observation data for the AI agent. The environment follows a similar
interface to OpenAI Gym environments with step() and reset() methods.
"""

import numpy as np
import random
from config import TETROMINOS, COLORBLIND_COLORS

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

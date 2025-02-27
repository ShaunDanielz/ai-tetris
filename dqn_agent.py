# dqn_agent.py
"""
Deep Q-Network Agent for Tetris.

This module implements a DQN agent that learns to play Tetris through
reinforcement learning. It handles neural network creation, experience 
replay, action selection, and model training. The agent uses epsilon-greedy
exploration and maintains a target network for stable training.
"""

import os
import time
import random  # Make sure this import is present
import numpy as np
import tensorflow as tf
from collections import deque

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

    def save_full_model(self, name):
        """Save the full model (architecture + weights)"""
        model_dir = os.path.dirname(name)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model.save(name)
        print(f"Full model saved to {name}")    
    
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
            action = np.random.randint(self.action_size)
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
        
        # FIX: Use random.sample instead of np.random.choice
        minibatch = random.sample(list(self.memory), batch_size)
        
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
        if action is None:
            return "No action"
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

    def save_full_model_for_web(self, name):
        """Save the full model optimized for web export"""
        model_dir = os.path.dirname(name)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Set the optimizer to be serializable
        self.model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        
        # Save the model
        self.model.save(name, save_format='h5')
        
        # Create a metadata file with information to help with web loading
        metadata = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "input_shape": [None, self.state_size],
            "output_shape": [None, self.action_size],
            "version": "1.0"
        }
        
        metadata_path = name.replace(".h5", "_metadata.json")
        with open(metadata_path, "w") as f:
            import json
            json.dump(metadata, f, indent=2)
        
        print(f"Full model saved to {name}")
        print(f"Model metadata saved to {metadata_path}")
        return name
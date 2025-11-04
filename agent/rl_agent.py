"""
Reinforcement Learning Agent using TensorFlow
Implements DQN (Deep Q-Network) for adaptive QoS management
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import logging
from typing import Tuple, List

LOG = logging.getLogger(__name__)


class DQNAgent:
    """
    Deep Q-Network agent for QoS management
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000,
                 batch_size: int = 32, update_target_every: int = 100):
        """
        Initialize DQN agent
        
        Args:
            state_size: Size of state vector
            action_size: Size of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial epsilon for epsilon-greedy exploration
            epsilon_min: Minimum epsilon value
            epsilon_decay: Epsilon decay rate
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            update_target_every: Steps between target network updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        
        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        
        # Training statistics
        self.training_step = 0
        self.loss_history = []
        
        LOG.info(f"DQN Agent initialized: state_size={state_size}, action_size={action_size}")
    
    def _build_model(self) -> keras.Model:
        """
        Build Deep Q-Network model
        
        Returns:
            Keras model
        """
        # Use Input layer instead of input_shape to avoid deprecation warnings
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_network(self):
        """
        Update target network weights from main network
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (use epsilon-greedy)
            
        Returns:
            Action index
        """
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Predict Q-values
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self) -> float:
        """
        Train agent on a batch of experiences
        
        Returns:
            Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q-values
        current_q_values = self.model.predict(states, verbose=0)
        
        # Next Q-values from target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Compute target Q-values
        target_q_values = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train model
        history = self.model.fit(states, target_q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.update_target_every == 0:
            self.update_target_network()
            LOG.info(f"Target network updated at step {self.training_step}")
        
        return loss
    
    def save(self, filepath: str):
        """
        Save model weights
        
        Args:
            filepath: Path to save model
        """
        self.model.save_weights(filepath)
        LOG.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model weights
        
        Args:
            filepath: Path to load model from
        """
        try:
            self.model.load_weights(filepath)
            self.update_target_network()
            LOG.info(f"Model loaded from {filepath}")
        except Exception as e:
            LOG.warning(f"Failed to load model: {e}")
    
    def get_stats(self) -> dict:
        """
        Get training statistics
        
        Returns:
            Dictionary with training stats
        """
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_steps': self.training_step,
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0.0
        }


class PPOAgent:
    """
    Proximal Policy Optimization agent (alternative to DQN)
    """
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.0003, gamma: float = 0.99,
                 clip_ratio: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        """
        Initialize PPO agent
        
        Args:
            state_size: Size of state vector
            action_size: Size of action space
            learning_rate: Learning rate
            gamma: Discount factor
            clip_ratio: PPO clip ratio
            value_coef: Value function coefficient
            entropy_coef: Entropy coefficient
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Build actor-critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        LOG.info(f"PPO Agent initialized: state_size={state_size}, action_size={action_size}")
    
    def _build_actor(self) -> keras.Model:
        """
        Build actor network (policy)
        """
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.action_size, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        
        return model
    
    def _build_critic(self) -> keras.Model:
        """
        Build critic network (value function)
        """
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        
        return model
    
    def act(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Sample action from policy
        
        Returns:
            Action index and action probability
        """
        probs = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
        action = np.random.choice(self.action_size, p=probs)
        return action, probs[action]
    
    def save(self, filepath: str):
        """Save models"""
        self.actor.save_weights(filepath + '_actor.h5')
        self.critic.save_weights(filepath + '_critic.h5')
    
    def load(self, filepath: str):
        """Load models"""
        self.actor.load_weights(filepath + '_actor.h5')
        self.critic.load_weights(filepath + '_critic.h5')




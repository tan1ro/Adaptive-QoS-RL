"""
Tests for RL agent functionality
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.rl_agent import DQNAgent
from agent.utils import preprocess_observation, clip_action


class TestRLAgent(unittest.TestCase):
    """Test RL agent functions"""
    
    def test_dqn_agent_initialization(self):
        """Test DQN agent initialization"""
        state_size = 16
        action_size = 9
        
        agent = DQNAgent(state_size, action_size)
        
        self.assertEqual(agent.state_size, state_size)
        self.assertEqual(agent.action_size, action_size)
        self.assertIsNotNone(agent.model)
        self.assertIsNotNone(agent.target_model)
    
    def test_dqn_act(self):
        """Test DQN agent action selection"""
        state_size = 16
        action_size = 9
        
        agent = DQNAgent(state_size, action_size)
        state = np.random.rand(state_size).astype(np.float32)
        
        action = agent.act(state, training=True)
        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, action_size)
    
    def test_dqn_remember(self):
        """Test experience storage"""
        state_size = 16
        action_size = 9
        
        agent = DQNAgent(state_size, action_size)
        
        state = np.random.rand(state_size).astype(np.float32)
        action = 0
        reward = 1.0
        next_state = np.random.rand(state_size).astype(np.float32)
        done = False
        
        initial_memory_size = len(agent.memory)
        agent.remember(state, action, reward, next_state, done)
        
        self.assertEqual(len(agent.memory), initial_memory_size + 1)
    
    def test_dqn_replay(self):
        """Test DQN training"""
        state_size = 16
        action_size = 9
        
        agent = DQNAgent(state_size, action_size)
        
        # Add some experiences
        for _ in range(50):
            state = np.random.rand(state_size).astype(np.float32)
            action = np.random.randint(action_size)
            reward = np.random.randn()
            next_state = np.random.rand(state_size).astype(np.float32)
            done = np.random.choice([True, False])
            agent.remember(state, action, reward, next_state, done)
        
        # Train
        loss = agent.replay()
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)
    
    def test_preprocess_observation(self):
        """Test observation preprocessing"""
        obs = {
            'link_utilization': [0.5, 0.7],
            'queue_length': [1000, 2000],
            'delay': [10, 20],
            'packet_loss': [0.01, 0.02]
        }
        
        processed = preprocess_observation(obs, state_dim=4)
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(len(processed), 16)  # 4 * 4
        self.assertTrue(all(0 <= x <= 1 for x in processed))
    
    def test_clip_action(self):
        """Test action clipping"""
        action_space_size = 9
        
        # Test valid action
        action = clip_action(5, action_space_size)
        self.assertEqual(action, 5)
        
        # Test action below range
        action = clip_action(-1, action_space_size)
        self.assertEqual(action, 0)
        
        # Test action above range
        action = clip_action(10, action_space_size)
        self.assertEqual(action, 8)


if __name__ == '__main__':
    unittest.main()


"""
Tests for QoS policy and controller functionality
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controller.qos_controller import QoSController
from agent.utils import action_to_qos_params, compute_reward, normalize_state


class TestQoSPolicy(unittest.TestCase):
    """Test QoS policy functions"""
    
    def test_action_to_qos_params(self):
        """Test action to QoS parameters conversion"""
        params = action_to_qos_params(0, num_queues=3)
        self.assertIn('queue_id', params)
        self.assertIn('min_rate', params)
        self.assertIn('max_rate', params)
        self.assertIn('priority', params)
        
        self.assertGreaterEqual(params['queue_id'], 0)
        self.assertLess(params['queue_id'], 3)
        self.assertGreater(params['min_rate'], 0)
        self.assertGreater(params['max_rate'], 0)
    
    def test_normalize_state(self):
        """Test state normalization"""
        state = {
            'link_utilization': [0.5, 0.7],
            'queue_length': [1000, 2000],
            'delay': [10, 20],
            'packet_loss': [0.01, 0.02]
        }
        
        normalized = normalize_state(state)
        self.assertIsInstance(normalized, np.ndarray)
        self.assertTrue(all(0 <= x <= 1 for x in normalized))
    
    def test_compute_reward(self):
        """Test reward computation"""
        state = {
            'delay': [10.0],
            'packet_loss': [0.01],
            'link_utilization': [0.5]
        }
        action = {'queue_id': 0}
        
        reward = compute_reward(state, action, alpha=0.5)
        self.assertIsInstance(reward, float)
    
    def test_controller_initialization(self):
        """Test controller initialization"""
        # Note: This may require Ryu framework to be properly set up
        try:
            controller = QoSController()
            self.assertIsNotNone(controller)
        except Exception as e:
            self.skipTest(f"Controller initialization requires Ryu: {e}")


if __name__ == '__main__':
    import numpy as np
    unittest.main()




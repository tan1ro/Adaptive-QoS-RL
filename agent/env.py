"""
OpenAI Gym-like environment for RL training
Simulates QoS feedback based on network state
"""

import gym
from gym import spaces
import numpy as np
import requests
import time
from typing import Dict, Tuple, Any
import logging

from agent.utils import normalize_state, compute_reward, action_to_qos_params

LOG = logging.getLogger(__name__)


class QoSEnvironment(gym.Env):
    """
    Custom environment for QoS management in SDN
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, controller_api_url: str = 'http://localhost:8080',
                 state_dim: int = 4, action_space_size: int = 9):
        """
        Initialize environment
        
        Args:
            controller_api_url: URL of Ryu controller REST API
            state_dim: Dimension of state vector per metric
            action_space_size: Size of action space
        """
        super(QoSEnvironment, self).__init__()
        
        self.controller_api_url = controller_api_url
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        
        # State space: [link_utilization, queue_length, delay, packet_loss]
        # Each metric can have multiple values (one per port/queue)
        # Total state size: state_dim * 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(state_dim * 4,),
            dtype=np.float32
        )
        
        # Action space: discrete actions for QoS adjustments
        self.action_space = spaces.Discrete(action_space_size)
        
        # Current state
        self.current_state = None
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        # Configuration
        self.reward_alpha = 0.5  # Weight for throughput in reward
        
        LOG.info(f"QoS Environment initialized: state_dim={state_dim}, action_space_size={action_space_size}")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state
        
        Returns:
            Initial observation
        """
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        # Fetch initial state from controller
        self.current_state = self._fetch_state()
        
        if self.current_state is None:
            # Return zero state if controller not available
            self.current_state = {
                'link_utilization': [0.0],
                'queue_length': [0.0],
                'delay': [0.0],
                'packet_loss': [0.0]
            }
        
        obs = self._state_to_observation(self.current_state)
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take
            
        Returns:
            observation: Next state observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Convert action to QoS parameters
        qos_params = action_to_qos_params(action)
        
        # Apply action to controller
        success = self._apply_action(qos_params)
        
        # Wait for state update
        time.sleep(0.5)
        
        # Fetch new state
        next_state = self._fetch_state()
        if next_state is None:
            next_state = self.current_state
        
        # Compute reward
        reward = compute_reward(next_state, qos_params, alpha=self.reward_alpha)
        
        # Update state
        self.current_state = next_state
        
        # Convert to observation
        obs = self._state_to_observation(next_state)
        
        # Determine if episode is done
        self.episode_steps += 1
        done = self.episode_steps >= 100  # Episode length limit
        
        # Track episode reward
        self.episode_reward += reward
        
        info = {
            'episode_reward': self.episode_reward,
            'steps': self.episode_steps,
            'qos_params': qos_params,
            'success': success
        }
        
        return obs, reward, done, info
    
    def _fetch_state(self) -> Dict:
        """
        Fetch current network state from controller
        
        Returns:
            State dictionary or None if request fails
        """
        try:
            response = requests.get(f'{self.controller_api_url}/api/v1/state', timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data.get('state', {})
            return None
        except Exception as e:
            LOG.warning(f"Failed to fetch state: {e}")
            return None
    
    def _apply_action(self, qos_params: Dict) -> bool:
        """
        Apply QoS action to controller
        
        Args:
            qos_params: QoS parameters dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {
                'dpid': 1,  # Default datapath ID
                'queue_id': qos_params['queue_id'],
                'min_rate': qos_params['min_rate'],
                'max_rate': qos_params['max_rate'],
                'priority': qos_params['priority']
            }
            
            response = requests.post(
                f'{self.controller_api_url}/api/v1/qos/apply',
                json=payload,
                timeout=2.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('success', False)
            return False
        except Exception as e:
            LOG.warning(f"Failed to apply action: {e}")
            return False
    
    def _state_to_observation(self, state: Dict) -> np.ndarray:
        """
        Convert state dictionary to observation vector
        
        Args:
            state: State dictionary
            
        Returns:
            Observation vector
        """
        from agent.utils import preprocess_observation
        return preprocess_observation(state, self.state_dim)
    
    def render(self, mode='human'):
        """
        Render environment state
        """
        if mode == 'human':
            if self.current_state:
                print(f"Step: {self.episode_steps}")
                print(f"State: {self.current_state}")
                print(f"Episode Reward: {self.episode_reward:.2f}")
    
    def close(self):
        """
        Clean up environment
        """
        pass




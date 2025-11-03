"""
Utility functions for preprocessing, reward computation, and data handling
"""

import numpy as np
from typing import Dict, List, Tuple


def normalize_state(state: Dict) -> np.ndarray:
    """
    Normalize network state for RL agent input
    
    Args:
        state: Dictionary with keys ['link_utilization', 'queue_length', 'delay', 'packet_loss']
        
    Returns:
        Normalized numpy array
    """
    features = []
    
    # Normalize link utilization (0-1 range)
    if 'link_utilization' in state and len(state['link_utilization']) > 0:
        util = np.array(state['link_utilization'])
        util = np.clip(util, 0.0, 1.0)
        features.extend(util)
    else:
        features.append(0.0)
    
    # Normalize queue length (assume max 10000)
    if 'queue_length' in state and len(state['queue_length']) > 0:
        queue = np.array(state['queue_length'])
        queue = np.clip(queue / 10000.0, 0.0, 1.0)
        features.extend(queue)
    else:
        features.append(0.0)
    
    # Normalize delay (assume max 500ms)
    if 'delay' in state and len(state['delay']) > 0:
        delay = np.array(state['delay'])
        delay = np.clip(delay / 500.0, 0.0, 1.0)
        features.extend(delay)
    else:
        features.append(0.0)
    
    # Normalize packet loss (0-1 range, assume max 10%)
    if 'packet_loss' in state and len(state['packet_loss']) > 0:
        loss = np.array(state['packet_loss'])
        loss = np.clip(loss / 0.1, 0.0, 1.0)
        features.extend(loss)
    else:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)


def compute_reward(state: Dict, action: Dict, alpha: float = 0.5) -> float:
    """
    Compute reward based on current state and action
    
    Reward = -(delay + packet_loss) + α * throughput
    
    Args:
        state: Current network state
        action: Action taken
        alpha: Weight for throughput component
        
    Returns:
        Reward value
    """
    # Extract metrics
    delay = np.mean(state.get('delay', [0.0])) if state.get('delay') else 0.0
    packet_loss = np.mean(state.get('packet_loss', [0.0])) if state.get('packet_loss') else 0.0
    
    # Compute throughput proxy (based on link utilization)
    utilization = np.mean(state.get('link_utilization', [0.0])) if state.get('link_utilization') else 0.0
    throughput = utilization * 100.0  # Normalize to 0-100 Mbps
    
    # Reward function: minimize delay and loss, maximize throughput
    reward = -(delay + packet_loss * 100) + alpha * throughput
    
    return float(reward)


def action_to_qos_params(action: int, num_queues: int = 3) -> Dict:
    """
    Convert discrete action to QoS parameters
    
    Args:
        action: Discrete action index
        num_queues: Number of queues
        
    Returns:
        Dictionary with QoS parameters
    """
    # Map action to queue configuration
    # Action space: [queue_id, min_rate, max_rate]
    
    queue_id = action % num_queues
    
    # Define rate tiers (in kbps)
    rate_tiers = [
        (1000, 10000),   # Low priority
        (5000, 50000),   # Medium priority
        (10000, 100000)  # High priority
    ]
    
    tier = (action // num_queues) % len(rate_tiers)
    min_rate, max_rate = rate_tiers[tier]
    
    return {
        'queue_id': queue_id,
        'min_rate': min_rate,
        'max_rate': max_rate,
        'priority': queue_id
    }


def qos_params_to_action(params: Dict, num_queues: int = 3) -> int:
    """
    Convert QoS parameters to discrete action
    
    Args:
        params: Dictionary with QoS parameters
        num_queues: Number of queues
        
    Returns:
        Discrete action index
    """
    queue_id = params.get('queue_id', 0)
    min_rate = params.get('min_rate', 1000)
    
    # Determine tier based on min_rate
    if min_rate >= 10000:
        tier = 2
    elif min_rate >= 5000:
        tier = 1
    else:
        tier = 0
    
    action = queue_id + tier * num_queues
    return action


def preprocess_observation(obs: Dict, state_dim: int = 4) -> np.ndarray:
    """
    Preprocess observation to fixed-size vector
    
    Args:
        obs: Observation dictionary
        state_dim: Target state dimension per metric
        
    Returns:
        Fixed-size numpy array
    """
    normalized = normalize_state(obs)
    
    # Pad or truncate to fixed size
    target_size = state_dim * 4  # 4 metrics
    
    if len(normalized) < target_size:
        # Pad with zeros
        padding = np.zeros(target_size - len(normalized))
        normalized = np.concatenate([normalized, padding])
    elif len(normalized) > target_size:
        # Truncate
        normalized = normalized[:target_size]
    
    return normalized.astype(np.float32)


def clip_action(action: np.ndarray, action_space_size: int) -> int:
    """
    Clip action to valid range
    
    Args:
        action: Raw action value
        action_space_size: Size of action space
        
    Returns:
        Clipped action index
    """
    if isinstance(action, np.ndarray):
        action = action.item() if action.size == 1 else action[0]
    
    action = int(np.clip(action, 0, action_space_size - 1))
    return action




"""
Main entry point for Adaptive QoS using Reinforcement Learning
Coordinates Ryu controller and RL agent
"""

import os
import sys
import subprocess
import threading
import time
import signal
import logging
import argparse
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)


class AdaptiveQoSSystem:
    """
    Main system orchestrator for controller and RL agent
    """
    
    def __init__(self, config_path: str = 'config/qos_config.yaml'):
        """
        Initialize system
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.controller_process = None
        self.agent_thread = None
        self.running = False
        
        LOG.info("Adaptive QoS System initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            LOG.error(f"Failed to load config: {e}")
            return {}
    
    def start_controller(self):
        """
        Start Ryu controller
        """
        LOG.info("Starting Ryu controller...")
        
        # Import controller components
        from ryu import cfg
        from ryu.base import app_manager
        from controller.qos_controller import QoSController
        from controller.rest_api import RESTAPI
        
        # Create controller instance
        # Note: RyuApp instances are automatically registered when imported
        # We create it here to ensure it's initialized
        controller = QoSController()
        
        # Start monitoring
        controller.start_monitoring()
        
        # Start REST API
        rest_api = RESTAPI(controller, host='0.0.0.0', port=8080)
        rest_api.start()
        
        # Keep controller thread alive
        # The controller will handle events through Ryu's event system
        # Since we're running in a separate thread, we just need to keep it alive
        try:
            while self.running:
                time.sleep(1)
        except Exception as e:
            LOG.error(f"Controller thread error: {e}")
    
    def start_agent_training(self, episodes: int = 1000, save_path: str = 'models/dqn_model'):
        """
        Start RL agent training
        
        Args:
            episodes: Number of training episodes
            save_path: Path to save trained model
        """
        LOG.info(f"Starting RL agent training for {episodes} episodes...")
        
        from agent.env import QoSEnvironment
        from agent.rl_agent import DQNAgent
        
        # Wait for controller to be ready
        import requests
        max_retries = 30
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.get('http://localhost:8080/api/v1/health', timeout=2.0)
                if response.status_code == 200:
                    LOG.info("Controller is ready")
                    break
            except:
                pass
            retry_count += 1
            time.sleep(1)
        
        if retry_count >= max_retries:
            LOG.error("Controller not available, exiting")
            return
        
        # Create environment
        env = QoSEnvironment(
            controller_api_url='http://localhost:8080',
            state_dim=4,
            action_space_size=9
        )
        
        # Create agent
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size)
        
        # Create model directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Training loop
        scores = []
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Agent chooses action
                action = agent.act(state, training=True)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                loss = agent.replay()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            scores.append(total_reward)
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            
            LOG.info(f"Episode {episode+1}/{episodes} - "
                    f"Score: {total_reward:.2f}, "
                    f"Avg (last 100): {avg_score:.2f}, "
                    f"Epsilon: {agent.epsilon:.3f}, "
                    f"Steps: {steps}")
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                agent.save(f"{save_path}_ep{episode+1}.h5")
                LOG.info(f"Model saved at episode {episode+1}")
        
        # Save final model
        agent.save(f"{save_path}_final.h5")
        LOG.info("Training completed")
    
    def start_agent_evaluation(self, model_path: str, episodes: int = 10):
        """
        Evaluate trained agent
        
        Args:
            model_path: Path to trained model
            episodes: Number of evaluation episodes
        """
        LOG.info(f"Starting RL agent evaluation for {episodes} episodes...")
        
        from agent.env import QoSEnvironment
        from agent.rl_agent import DQNAgent
        
        # Create environment
        env = QoSEnvironment(
            controller_api_url='http://localhost:8080',
            state_dim=4,
            action_space_size=9
        )
        
        # Create agent and load model
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size)
        agent.load(model_path)
        agent.epsilon = 0  # No exploration during evaluation
        
        # Evaluation loop
        scores = []
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            scores.append(total_reward)
            LOG.info(f"Evaluation Episode {episode+1}/{episodes} - "
                    f"Score: {total_reward:.2f}, Steps: {steps}")
        
        avg_score = np.mean(scores)
        LOG.info(f"Evaluation complete - Average Score: {avg_score:.2f}")
    
    def run(self, mode: str = 'training', episodes: int = 1000):
        """
        Run the complete system
        
        Args:
            mode: 'training' or 'evaluation'
            episodes: Number of episodes
        """
        self.running = True
        
        # Start controller in separate thread
        controller_thread = threading.Thread(target=self.start_controller, daemon=True)
        controller_thread.start()
        
        # Wait a bit for controller to start
        time.sleep(3)
        
        # Start agent based on mode
        if mode == 'training':
            self.start_agent_training(episodes=episodes)
        elif mode == 'evaluation':
            model_path = 'models/dqn_model_final.h5'
            self.start_agent_evaluation(model_path, episodes=episodes)
        else:
            LOG.error(f"Unknown mode: {mode}")
        
        self.running = False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Adaptive QoS using Reinforcement Learning')
    parser.add_argument('--mode', type=str, default='training',
                       choices=['training', 'evaluation'],
                       help='Operation mode: training or evaluation')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes')
    parser.add_argument('--config', type=str, default='config/qos_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize system
    system = AdaptiveQoSSystem(config_path=args.config)
    
    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        LOG.info("Shutting down...")
        system.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run system
    try:
        system.run(mode=args.mode, episodes=args.episodes)
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
    except Exception as e:
        LOG.error(f"Error: {e}", exc_info=True)
    finally:
        system.running = False


if __name__ == '__main__':
    import numpy as np  # Required for main.py
    main()




"""
Main entry point for Adaptive QoS using Reinforcement Learning
Coordinates Ryu controller and RL agent

This file orchestrates the entire system:
- Initializes and starts the Ryu SDN Controller
- Starts the REST API server for agent-controller communication
- Manages RL agent training/evaluation loops
- Provides real-time metrics updates to the web dashboard
"""

# Standard library imports for system operations
import os          # File and directory operations
import sys         # System-specific parameters and functions
import subprocess  # Spawning processes (not used but available)
import threading   # Thread management for concurrent operations
import time        # Time-related functions for delays and sleep
import signal      # Signal handling for graceful shutdown
import logging     # Logging system for debugging and monitoring
import argparse    # Command-line argument parsing
import yaml        # YAML file parsing for configuration
from pathlib import Path  # Object-oriented filesystem paths

# Configure logging system
# Sets up logging format: timestamp - logger name - log level - message
logging.basicConfig(
    level=logging.INFO,  # Log level: INFO shows informational messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)  # Create logger for this module


class AdaptiveQoSSystem:
    """
    Main system orchestrator for controller and RL agent
    
    This class coordinates:
    - Ryu SDN Controller initialization and management
    - REST API server for web dashboard and agent communication
    - RL agent training and evaluation loops
    - System lifecycle management
    """
    
    def __init__(self, config_path: str = 'config/qos_config.yaml'):
        """
        Initialize the Adaptive QoS system
        
        Args:
            config_path: Path to YAML configuration file containing:
                - Queue priorities and rate limits
                - Network state thresholds
                - RL parameters (learning rate, epsilon decay, etc.)
                - Topology settings
        """
        # Load configuration from YAML file
        self.config = self._load_config(config_path)
        
        # Store references for process/thread management
        self.controller_process = None  # Not used currently but available for future
        self.agent_thread = None        # Thread reference for agent execution
        self.running = False            # Flag to control system lifecycle
        
        # Log successful initialization
        LOG.info("Adaptive QoS System initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration parameters
            Returns empty dict if file doesn't exist or fails to load
        """
        try:
            # Open and read YAML file safely
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)  # Safe YAML parsing prevents code execution
            return config
        except Exception as e:
            # Log error but don't crash - use defaults
            LOG.error(f"Failed to load config: {e}")
            return {}  # Return empty dict as fallback
    
    def start_controller(self):
        """
        Start Ryu SDN Controller and REST API server
        
        This method:
        1. Imports and initializes Ryu application manager
        2. Loads and instantiates the QoS Controller application
        3. Starts statistics monitoring thread
        4. Starts REST API server for web dashboard and agent communication
        5. Keeps the controller thread alive to handle OpenFlow events
        """
        LOG.info("Starting Ryu controller...")
        
        # Import Ryu framework components
        from ryu import cfg               # Configuration system
        from ryu.base import app_manager # Application manager for Ryu apps
        from controller.qos_controller import QoSController  # Our custom QoS controller
        from controller.rest_api import RESTAPI              # REST API server
        
        # Import the controller module to register it with Ryu
        # This ensures the RyuApp decorator registers the class
        import controller.qos_controller
        
        # Initialize Ryu's application manager
        # This manages all Ryu applications and their lifecycle
        apps = ['controller.qos_controller']  # List of app module paths
        
        # Create AppManager instance - manages Ryu applications
        app_mgr = app_manager.AppManager()
        
        # Load application modules - discovers and imports the apps
        app_mgr.load_apps(apps)
        
        # Instantiate applications - creates instances of each app class
        # contexts=None: use default contexts
        # log_early=False: don't log before full initialization
        app_mgr.instantiate_apps(contexts=None, log_early=False)
        
        # Get the instantiated controller instance
        # 'QoSController' is the class name registered by RyuApp decorator
        controller = app_mgr.applications['QoSController']
        
        # Start statistics monitoring
        # This spawns a thread that periodically requests stats from switches
        controller.start_monitoring()
        
        # Initialize REST API server
        # host='0.0.0.0': listen on all network interfaces
        # port=8080: standard port for web dashboard
        rest_api = RESTAPI(controller, host='0.0.0.0', port=8080)
        
        # Start REST API in a separate daemon thread
        # Daemon threads automatically terminate when main program exits
        rest_api.start()
        
        # Keep controller thread alive
        # This loop ensures the thread stays running to handle:
        # - OpenFlow events from switches
        # - Statistics replies
        # - Packet-in events
        try:
            while self.running:  # Loop until running flag is False
                time.sleep(1)   # Sleep to avoid busy-waiting (CPU efficient)
        except Exception as e:
            LOG.error(f"Controller thread error: {e}")
    
    def start_agent_training(self, episodes: int = 1000, save_path: str = 'models/dqn_model'):
        """
        Start RL agent training loop
        
        This method:
        1. Waits for controller REST API to be ready
        2. Creates RL environment and DQN agent
        3. Runs training episodes:
           - Agent observes network state
           - Agent selects QoS action
           - Action is applied to network
           - Agent receives reward and learns
        4. Updates metrics for web dashboard
        5. Saves model periodically
        
        Args:
            episodes: Number of training episodes to run
            save_path: Base path for saving trained models
        """
        LOG.info(f"Starting RL agent training for {episodes} episodes...")
        
        # Import RL components
        from agent.env import QoSEnvironment  # RL environment wrapper
        from agent.rl_agent import DQNAgent  # Deep Q-Network agent
        
        # Wait for controller REST API to be ready
        # This ensures the controller and dashboard are initialized
        import requests  # HTTP library for API calls
        max_retries = 30  # Maximum number of retry attempts
        retry_count = 0
        
        # Poll the health endpoint until controller responds
        while retry_count < max_retries:
            try:
                # Try to connect to controller health endpoint
                response = requests.get('http://localhost:8080/api/v1/health', timeout=2.0)
                if response.status_code == 200:  # HTTP 200 = success
                    LOG.info("Controller is ready")
                    break  # Exit loop on success
            except:
                # If connection fails, wait and retry
                pass
            retry_count += 1
            time.sleep(1)  # Wait 1 second between retries
        
        # If controller never becomes ready, exit training
        if retry_count >= max_retries:
            LOG.error("Controller not available, exiting")
            return
        
        # Create RL environment
        # This wraps the network controller as a Gym-compatible environment
        env = QoSEnvironment(
            controller_api_url='http://localhost:8080',  # REST API endpoint
            state_dim=4,        # State space: [utilization, queue, delay, loss]
            action_space_size=9 # Action space: 9 different QoS configurations
        )
        
        # Create DQN agent
        # Get dimensions from environment
        state_size = env.observation_space.shape[0]  # Size of state vector
        action_size = env.action_space.n             # Number of possible actions
        agent = DQNAgent(state_size, action_size)     # Initialize agent
        
        # Create directory for saving models if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Initialize training tracking
        scores = []  # List to store reward for each episode
        
        # Notify web dashboard that training is starting
        try:
            requests.post('http://localhost:8080/api/v1/training/metrics', 
                         json={
                             'is_training': True,      # Training status flag
                             'total_episodes': episodes # Total episodes to run
                         })
        except:
            pass  # Don't crash if dashboard update fails
        
        # Main training loop - one iteration per episode
        for episode in range(episodes):
            # Reset environment to initial state
            state = env.reset()
            
            # Initialize episode tracking variables
            total_reward = 0      # Cumulative reward for this episode
            steps = 0             # Number of steps in this episode
            episode_loss = 0.0     # Cumulative training loss
            loss_count = 0         # Number of loss calculations
            
            # Episode loop - continues until environment signals done
            while True:
                # Agent selects action using epsilon-greedy policy
                # In training mode, agent explores with epsilon probability
                action = agent.act(state, training=True)
                
                # Execute action in environment
                # Returns: next state, reward, done flag, info dict
                next_state, reward, done, info = env.step(action)
                
                # Store experience in replay buffer
                # Format: (state, action, reward, next_state, done)
                # Used later for batch training
                agent.remember(state, action, reward, next_state, done)
                
                # Train agent on batch of experiences
                # Returns loss value if training occurred, None otherwise
                loss = agent.replay()
                
                # Accumulate loss for episode average
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
                
                # Update state for next iteration
                state = next_state
                total_reward += reward  # Accumulate reward
                steps += 1              # Increment step counter
                
                # Check if episode is complete
                if done:
                    break
            
            # Calculate episode statistics
            scores.append(total_reward)  # Store episode reward
            
            # Calculate average reward (moving average over last 100 episodes)
            # This shows learning progress over time
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            
            # Calculate average loss for this episode
            avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
            
            # Update web dashboard with latest metrics
            # This enables real-time visualization of training progress
            try:
                requests.post('http://localhost:8080/api/v1/training/metrics',
                            json={
                                'episode': episode + 1,                    # Current episode number
                                'current_reward': float(total_reward),     # Reward this episode
                                'average_reward': float(avg_score),        # Average reward
                                'epsilon': float(agent.epsilon),           # Exploration rate
                                'loss': float(avg_loss),                   # Training loss
                                'scores': [float(s) for s in scores[-100:]] # Last 100 scores for chart
                            })
            except:
                pass  # Don't crash if dashboard update fails
            
            # Log episode summary
            LOG.info(f"Episode {episode+1}/{episodes} - "
                    f"Score: {total_reward:.2f}, "          # Episode reward
                    f"Avg (last 100): {avg_score:.2f}, "    # Moving average
                    f"Epsilon: {agent.epsilon:.3f}, "       # Exploration rate
                    f"Steps: {steps}")                       # Episode length
            
            # Save model periodically (every 100 episodes)
            # This allows resuming training or evaluation later
            if (episode + 1) % 100 == 0:
                agent.save(f"{save_path}_ep{episode+1}.h5")
                LOG.info(f"Model saved at episode {episode+1}")
        
        # Mark training as complete in dashboard
        try:
            requests.post('http://localhost:8080/api/v1/training/metrics',
                         json={'is_training': False})
        except:
            pass
        
        # Save final model after all episodes complete
        agent.save(f"{save_path}_final.h5")
        LOG.info("Training completed")
    
    def start_agent_evaluation(self, model_path: str, episodes: int = 10):
        """
        Evaluate a trained RL agent
        
        This method:
        1. Loads a pre-trained model
        2. Runs episodes with exploration disabled (epsilon=0)
        3. Logs performance metrics
        
        Args:
            model_path: Path to saved model file (.h5 format)
            episodes: Number of evaluation episodes to run
        """
        LOG.info(f"Starting RL agent evaluation for {episodes} episodes...")
        
        # Import RL components
        from agent.env import QoSEnvironment
        from agent.rl_agent import DQNAgent
        
        # Create environment (same as training)
        env = QoSEnvironment(
            controller_api_url='http://localhost:8080',
            state_dim=4,
            action_space_size=9
        )
        
        # Create agent with same architecture
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size)
        
        # Load pre-trained weights from file
        agent.load(model_path)
        
        # Disable exploration - agent always selects best action
        agent.epsilon = 0
        
        # Evaluation loop
        scores = []
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            # Episode loop
            while True:
                # Select action without exploration
                action = agent.act(state, training=False)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Update state
                state = next_state
                total_reward += reward
                steps += 1
                
                # Check if episode complete
                if done:
                    break
            
            # Store and log results
            scores.append(total_reward)
            LOG.info(f"Evaluation Episode {episode+1}/{episodes} - "
                    f"Score: {total_reward:.2f}, Steps: {steps}")
        
        # Calculate and log average performance
        avg_score = np.mean(scores)
        LOG.info(f"Evaluation complete - Average Score: {avg_score:.2f}")
    
    def run(self, mode: str = 'training', episodes: int = 1000):
        """
        Run the complete system
        
        This is the main orchestration method that:
        1. Starts controller in background thread
        2. Waits for controller initialization
        3. Starts training or evaluation based on mode
        
        Args:
            mode: 'training' or 'evaluation'
            episodes: Number of episodes to run
        """
        # Set running flag to True
        self.running = True
        
        # Start controller in separate daemon thread
        # Daemon threads terminate when main program exits
        controller_thread = threading.Thread(target=self.start_controller, daemon=True)
        controller_thread.start()
        
        # Wait for controller to initialize
        # Gives time for REST API to start
        time.sleep(3)
        
        # Start agent based on selected mode
        if mode == 'training':
            self.start_agent_training(episodes=episodes)
        elif mode == 'evaluation':
            model_path = 'models/dqn_model_final.h5'
            self.start_agent_evaluation(model_path, episodes=episodes)
        else:
            LOG.error(f"Unknown mode: {mode}")
        
        # Set running flag to False (signals threads to stop)
        self.running = False


def main():
    """
    Main entry point for the application
    
    This function:
    1. Parses command-line arguments
    2. Initializes the system
    3. Sets up signal handlers for graceful shutdown
    4. Runs the system
    5. Handles exceptions and cleanup
    """
    # Create argument parser for command-line interface
    parser = argparse.ArgumentParser(
        description='Adaptive QoS using Reinforcement Learning'
    )
    
    # Add command-line arguments
    parser.add_argument(
        '--mode', 
        type=str, 
        default='training',
        choices=['training', 'evaluation'],
        help='Operation mode: training or evaluation'
    )
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=1000,
        help='Number of episodes to run'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/qos_config.yaml',
        help='Path to configuration file'
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Initialize the Adaptive QoS system
    system = AdaptiveQoSSystem(config_path=args.config)
    
    # Set up signal handlers for graceful shutdown
    # SIGINT: Ctrl+C
    # SIGTERM: Termination signal (kill command)
    def signal_handler(sig, frame):
        LOG.info("Shutting down...")
        system.running = False  # Signal threads to stop
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the system with error handling
    try:
        system.run(mode=args.mode, episodes=args.episodes)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        LOG.info("Interrupted by user")
    except Exception as e:
        # Log any unexpected errors with full traceback
        LOG.error(f"Error: {e}", exc_info=True)
    finally:
        # Ensure running flag is set to False
        system.running = False


# Entry point when script is run directly
if __name__ == '__main__':
    # Import numpy here (needed for calculations in training loop)
    import numpy as np
    
    # Run main function
    main()

"""
Main entry point for Adaptive QoS using Reinforcement Learning
Coordinates Ryu controller and RL agent

This file orchestrates the entire system:
- Initializes and starts the Ryu SDN Controller
- Starts the REST API server for agent-controller communication
- Manages RL agent training/evaluation loops
- Provides real-time metrics updates to the web dashboard
"""

# Set TensorFlow environment variables early to prevent CUDA compilation errors
# These settings disable XLA GPU compilation which requires CUDA toolkit
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
os.environ['XLA_FLAGS'] = '--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found'
# Alternative: Uncomment next line to force CPU-only mode if GPU issues persist
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Standard library imports for system operations
# Note: os is already imported above for environment variables
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
        self.controller_process = None  # Ryu controller subprocess (if using ryu-manager)
        self.controller_app_mgr = None   # AppManager instance for controller
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
    
    def _check_and_free_port(self, port: int):
        """
        Check if a port is in use and free it if necessary
        
        Args:
            port: Port number to check (e.g., 6653 for OpenFlow)
        """
        import socket
        import subprocess
        
        # First check if port is actually in use by trying to bind to it
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
            # Port is free, no action needed
            return
        except OSError:
            # Port is in use
            sock.close()
            LOG.warning(f"Port {port} is already in use. Attempting to free it...")
            
            # Try to find and kill the process using the port
            try:
                # Try using lsof first (more reliable on Linux)
                result = subprocess.run(
                    ['lsof', '-ti', f':{port}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid.strip():
                            try:
                                pid_int = int(pid.strip())
                                LOG.info(f"Killing process {pid_int} using port {port}...")
                                os.kill(pid_int, signal.SIGTERM)
                                time.sleep(1)  # Give process time to die
                                # Check if still running, force kill if needed
                                try:
                                    os.kill(pid_int, 0)  # Check if process exists
                                    LOG.warning(f"Process {pid_int} still running, sending SIGKILL...")
                                    os.kill(pid_int, signal.SIGKILL)
                                    time.sleep(0.5)
                                except ProcessLookupError:
                                    pass  # Process already dead
                            except (ValueError, ProcessLookupError, PermissionError) as e:
                                LOG.debug(f"Could not kill process {pid}: {e}")
                    
                    # Wait a bit for port to be freed
                    time.sleep(2)
                    
                    # Verify port is now free
                    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        sock2.bind(('0.0.0.0', port))
                        sock2.close()
                        LOG.info(f"Port {port} is now free")
                        return
                    except OSError:
                        sock2.close()
                        LOG.warning(f"Port {port} is still in use after cleanup attempt")
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                # lsof not available or failed, try alternative methods
                try:
                    # Try using fuser (alternative method)
                    result = subprocess.run(
                        ['fuser', '-k', f'{port}/tcp'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        LOG.info(f"Freed port {port} using fuser")
                        time.sleep(2)
                        return
                except (FileNotFoundError, subprocess.SubprocessError):
                    pass
                
                # Last resort: try to kill any Python processes that might be Ryu controllers
                LOG.warning("Could not use lsof/fuser. Trying to kill potential Ryu controller processes...")
                try:
                    result = subprocess.run(
                        ['pkill', '-f', 'ryu-manager|run_apps|qos_controller'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    time.sleep(2)
                    LOG.info("Attempted to kill potential Ryu controller processes")
                except subprocess.SubprocessError:
                    pass
    
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
        
        # Check and free port 6653 if it's already in use
        # This prevents "Address already in use" errors from previous runs
        self._check_and_free_port(6653)
        
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
        self.controller_app_mgr = app_mgr  # Store reference
        
        # CRITICAL: Start the OpenFlow controller server using run_apps()
        # run_apps() will:
        # 1. Load the apps
        # 2. Instantiate them  
        # 3. Start the OpenFlow server on port 6653
        # It blocks, so we run it in a background thread
        LOG.info("Starting OpenFlow controller server on port 6653...")
        
        # Controller instance will be available after run_apps() instantiates apps
        # Use threading.Event to signal when controller is ready
        controller_ref = {'instance': None}
        controller_ready_event = threading.Event()
        
        def run_controller_server():
            """
            Run controller server in background thread
            run_apps() loads, instantiates, AND starts the OpenFlow server
            This blocks, so it runs in a separate thread
            """
            # Poll for controller instance while run_apps() is starting up
            # run_apps() instantiates apps before blocking on the server loop
            def poll_for_controller():
                """Poll for controller instance and set it in controller_ref"""
                from controller.qos_controller import QoSController as QC
                import time as time_module
                
                for attempt in range(10):  # Check 10 times over ~2 seconds
                    try:
                        # Check SERVICE_BRICKS first (most reliable)
                        if hasattr(app_manager, 'SERVICE_BRICKS'):
                            if QC in app_manager.SERVICE_BRICKS:
                                controller_ref['instance'] = app_manager.SERVICE_BRICKS[QC]
                                LOG.info("Controller instance stored from SERVICE_BRICKS in thread")
                                controller_ready_event.set()
                                return True
                            # Search through all services
                            for svc_class, svc_inst in app_manager.SERVICE_BRICKS.items():
                                if isinstance(svc_inst, QC):
                                    controller_ref['instance'] = svc_inst
                                    LOG.info(f"Controller instance found in SERVICE_BRICKS: {svc_class.__name__}")
                                    controller_ready_event.set()
                                    return True
                        
                        # Fallback to applications dict
                        possible_keys = ['controller.qos_controller', 'QoSController', 'qos_controller']
                        for key in possible_keys:
                            if key in app_mgr.applications:
                                controller_ref['instance'] = app_mgr.applications[key]
                                LOG.info(f"Controller instance stored from applications dict in thread: {key}")
                                controller_ready_event.set()
                                return True
                        
                        # Search by type
                        for key, app_inst in app_mgr.applications.items():
                            if isinstance(app_inst, QC):
                                controller_ref['instance'] = app_inst
                                LOG.info(f"Controller instance stored by type in thread: {key}")
                                controller_ready_event.set()
                                return True
                    except Exception as e:
                        LOG.debug(f"Poll attempt {attempt+1} failed: {e}")
                    
                    time_module.sleep(0.2)  # Wait 200ms between attempts
                
                return False
            
            try:
                # run_apps() does everything: loads, instantiates, and starts OpenFlow server
                # The OpenFlow server will listen on port 6653 for switch connections
                LOG.info("Starting run_apps() - this will load apps and start OpenFlow server...")
                
                # Start polling in a separate thread within this thread context
                import threading
                poll_thread = threading.Thread(target=poll_for_controller, daemon=True)
                poll_thread.start()
                
                # Start run_apps (this blocks)
                app_mgr.run_apps(apps)
                
                LOG.info("run_apps() completed (this shouldn't normally happen unless server stops)")
            except Exception as e:
                LOG.error(f"Controller server error: {e}", exc_info=True)
                import traceback
                LOG.error(traceback.format_exc())
        
        # Start controller server in background daemon thread
        # Daemon thread will terminate when main program exits
        controller_server_thread = threading.Thread(target=run_controller_server, daemon=True)
        controller_server_thread.start()
        LOG.info("OpenFlow controller server thread started")
        
        # Give the thread a moment to start executing
        # This prevents race condition where we check before run_apps() begins
        time.sleep(2)  # Increased to 2 seconds to give run_apps() time to start
        
        # Wait for apps to be instantiated by run_apps() in the background thread
        # CRITICAL: When run_apps() runs in a thread, the apps may be stored in a 
        # thread-local context. We need to use Ryu's service registry or get the instance
        # through a different mechanism.
        LOG.info("Waiting for controller apps to initialize...")
        max_wait = 30
        controller_key = None
        
        # Import QoSController class for type checking and service lookup
        from controller.qos_controller import QoSController as QoSControllerClass
        
        # Wait for controller instance to be set by the background thread OR find it ourselves
        # The thread will try to set it, but we also poll as backup
        for i in range(max_wait):
            # First check if the thread found it and set it in controller_ref
            if controller_ref['instance'] is not None:
                LOG.info("Controller instance obtained from background thread")
                break
            
            # Method 1: Try Ryu's SERVICE_BRICKS registry (most reliable)
            if controller_ref['instance'] is None:
                try:
                    # Ryu stores app instances in SERVICE_BRICKS
                    if hasattr(app_manager, 'SERVICE_BRICKS'):
                        service_bricks = app_manager.SERVICE_BRICKS
                        # Try to get QoSController from service registry
                        if QoSControllerClass in service_bricks:
                            controller_ref['instance'] = service_bricks[QoSControllerClass]
                            LOG.info("Controller instance obtained via SERVICE_BRICKS registry")
                            break
                        # Or search through all services
                        for service_class, service_instance in service_bricks.items():
                            if isinstance(service_instance, QoSControllerClass):
                                controller_ref['instance'] = service_instance
                                LOG.info(f"Controller instance found in SERVICE_BRICKS: {service_class.__name__}")
                                break
                        if controller_ref['instance'] is not None:
                            break
                except Exception as e:
                    LOG.debug(f"SERVICE_BRICKS lookup failed: {e}")
            
            # Method 2: Check instance's applications dict
            if controller_ref['instance'] is None:
                possible_keys = ['controller.qos_controller', 'QoSController', 'qos_controller']
                for key in possible_keys:
                    if key in app_mgr.applications:
                        controller_ref['instance'] = app_mgr.applications[key]
                        controller_key = key
                        LOG.info(f"Controller instance obtained with key: {key}")
                        break
            
            # Method 3: Search by type if any apps are loaded
            if controller_ref['instance'] is None and len(app_mgr.applications) > 0:
                for key, app_instance in app_mgr.applications.items():
                    if isinstance(app_instance, QoSControllerClass):
                        controller_ref['instance'] = app_instance
                        controller_key = key
                        LOG.info(f"Controller instance found by type with key: {key}")
                        break
            
            if controller_ref['instance'] is not None:
                break
            
            # Debug: log what's available (every 5 seconds)
            if i > 0 and i % 5 == 0:
                instance_keys = list(app_mgr.applications.keys())
                try:
                    service_keys = [k.__name__ for k in app_manager.SERVICE_BRICKS.keys()] if hasattr(app_manager, 'SERVICE_BRICKS') else []
                except:
                    service_keys = []
                
                if instance_keys:
                    LOG.info(f"Instance dict keys: {instance_keys}")
                if service_keys:
                    LOG.info(f"SERVICE_BRICKS keys: {service_keys}")
                if not instance_keys and not service_keys:
                    LOG.debug(f"No apps loaded yet ({i}/{max_wait} seconds) - thread may still be starting run_apps()")
            
            time.sleep(1)
        
        if controller_ref['instance'] is None:
            # Log what's actually available for debugging
            LOG.error(f"Failed to get controller instance after {max_wait} seconds")
            LOG.error("Attempted methods:")
            LOG.error("  1. app_mgr.applications dict")
            LOG.error("  2. app_manager.applications (module-level)")
            LOG.error("  3. app_manager.get_instance()")
            LOG.error("  4. Type-based search")
            
            instance_keys = list(app_mgr.applications.keys())
            try:
                module_keys = list(app_manager.applications.keys())
            except:
                module_keys = []
            
            if instance_keys:
                LOG.error(f"Instance dict has: {instance_keys}")
            if module_keys:
                LOG.error(f"Module dict has: {module_keys}")
            if not instance_keys and not module_keys:
                LOG.error("Both dicts are empty - run_apps() may not have started or failed silently")
            
            # Don't raise exception yet - try one more thing: wait a bit longer
            # Sometimes apps take time to appear after instantiation logs
            LOG.warning("Trying one more time after brief wait...")
            time.sleep(3)
            
            # Final attempt
            for key in ['controller.qos_controller', 'QoSController', 'qos_controller']:
                if key in app_mgr.applications:
                    controller_ref['instance'] = app_mgr.applications[key]
                    LOG.info(f"Controller found on final attempt with key: {key}")
                    break
            
            if controller_ref['instance'] is None:
                # Last resort: raise exception
                raise Exception("Controller initialization failed - cannot proceed without controller instance")
        
        # Get the controller instance
        controller = controller_ref['instance']
        
        # Start statistics monitoring
        # This spawns a thread that periodically requests stats from switches
        controller.start_monitoring()
        
        # Check and free port 8888 if it's already in use
        # This prevents "Address already in use" errors from previous runs
        self._check_and_free_port(8888)
        
        # Initialize REST API server
        # host='0.0.0.0': listen on all network interfaces
        # port=8888: standard port for web dashboard
        rest_api = RESTAPI(controller, host='0.0.0.0', port=8888)
        
        # Start REST API in a separate daemon thread
        # Daemon threads automatically terminate when main program exits
        rest_api.start()
        
        # Give REST API time to bind to port
        LOG.info("Waiting for REST API server to bind to port 8888...")
        time.sleep(3)  # Give Flask time to start
        
        # Verify REST API is accessible
        import requests
        for attempt in range(10):
            try:
                response = requests.get('http://localhost:8888/api/v1/health', timeout=1.0)
                if response.status_code == 200:
                    LOG.info("REST API is ready and responding")
                    break
            except:
                if attempt == 9:
                    LOG.warning("REST API may not be ready yet, but continuing...")
                time.sleep(0.5)
        
        # Give OpenFlow server time to bind to port 6653
        LOG.info("Waiting for OpenFlow server to bind to port 6653...")
        time.sleep(3)
        
        # Keep controller thread alive to handle events
        # The REST API and controller service run in background
        LOG.info("Controller initialized. Waiting for switches to connect...")
        try:
            # Wait and periodically check for switch connections
            connection_timeout = 60  # Wait up to 60 seconds for first connection
            elapsed = 0
            while self.running and elapsed < connection_timeout:
                if len(controller.datapaths) > 0:
                    LOG.info(f"Switches connected: {list(controller.datapaths.keys())}")
                    break
                time.sleep(1)
                elapsed += 1
            
            if elapsed >= connection_timeout:
                LOG.warning("No switches connected within timeout. Continuing anyway...")
                LOG.info("Mininet switches may connect later, or check Mininet logs")
            
            # Keep thread alive to handle ongoing connections and events
            while self.running:
                time.sleep(1)
        except Exception as e:
            LOG.error(f"Controller thread error: {e}")
            # If error occurs, keep thread alive to maintain REST API
            while self.running:
                time.sleep(1)
    
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
        max_retries = 60  # Increased to 60 seconds to match wait_for_controller
        retry_count = 0
        
        LOG.info("Waiting for controller REST API to be ready...")
        
        # Poll the health endpoint until controller responds
        while retry_count < max_retries:
            try:
                # Try to connect to controller health endpoint
                response = requests.get('http://localhost:8888/api/v1/health', timeout=2.0)
                if response.status_code == 200:  # HTTP 200 = success
                    LOG.info("Controller REST API is ready and responding")
                    break  # Exit loop on success
            except requests.exceptions.ConnectionError:
                # Connection refused - API not started yet
                if retry_count % 10 == 0 and retry_count > 0:
                    LOG.info(f"Still waiting for REST API... ({retry_count}/{max_retries})")
            except Exception as e:
                # Other errors
                if retry_count % 10 == 0:
                    LOG.debug(f"REST API check error: {e}")
            
            retry_count += 1
            time.sleep(1)  # Wait 1 second between retries
        
        # If controller never becomes ready, exit training with error
        if retry_count >= max_retries:
            LOG.error("Controller REST API not available after 60 seconds")
            LOG.error("Please check:")
            LOG.error("  1. Is the controller thread running?")
            LOG.error("  2. Is port 8888 in use by another process?")
            LOG.error("  3. Check /tmp/adaptive_qos.log for errors")
            raise RuntimeError("Controller REST API not available - cannot start training")
        
        # Create RL environment
        # This wraps the network controller as a Gym-compatible environment
        env = QoSEnvironment(
            controller_api_url='http://localhost:8888',  # REST API endpoint
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
            requests.post('http://localhost:8888/api/v1/training/metrics', 
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
                requests.post('http://localhost:8888/api/v1/training/metrics',
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
            requests.post('http://localhost:8888/api/v1/training/metrics',
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
            controller_api_url='http://localhost:8888',
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
        
        # Wait for controller REST API to be ready before proceeding
        # This is critical - training will fail if REST API isn't ready
        LOG.info("Waiting for controller REST API to be ready...")
        import requests
        max_wait = 60
        api_ready = False
        
        for attempt in range(max_wait):
            try:
                response = requests.get('http://localhost:8888/api/v1/health', timeout=2.0)
                if response.status_code == 200:
                    LOG.info("Controller REST API is ready!")
                    api_ready = True
                    break
            except:
                if attempt % 10 == 0 and attempt > 0:
                    LOG.info(f"Still waiting for REST API... ({attempt}/{max_wait})")
            time.sleep(1)
        
        if not api_ready:
            LOG.warning("REST API not ready after waiting, but continuing...")
            LOG.warning("Training may fail if REST API doesn't start soon")
        
        # Wait a bit more for switches to connect (if Mininet is running)
        LOG.info("Waiting for switches to connect (if Mininet is running)...")
        time.sleep(5)
        
        # Verify switches are connected by checking controller state
        for attempt in range(10):
            try:
                response = requests.get('http://localhost:8888/api/v1/state', timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        dpids = data.get('available_dpids', [])
                        if dpids:
                            LOG.info(f"Switches connected: {dpids}")
                            break
                        elif attempt == 9:
                            LOG.warning("No switches connected yet - Mininet may still be starting")
            except:
                pass
            time.sleep(1)
        
        # Start agent based on selected mode
        # These methods run synchronously (block until complete)
        # The controller and REST API continue running in background threads
        try:
            if mode == 'training':
                LOG.info(f"Starting training mode with {episodes} episodes...")
                self.start_agent_training(episodes=episodes)
                LOG.info("Training completed successfully")
            elif mode == 'evaluation':
                model_path = 'models/dqn_model_final.h5'
                LOG.info(f"Starting evaluation mode with {episodes} episodes...")
                self.start_agent_evaluation(model_path, episodes=episodes)
                LOG.info("Evaluation completed successfully")
            else:
                LOG.error(f"Unknown mode: {mode}")
                raise ValueError(f"Invalid mode: {mode}")
        except KeyboardInterrupt:
            LOG.info("Training/evaluation interrupted by user")
            raise
        except Exception as e:
            LOG.error(f"Error during training/evaluation: {e}", exc_info=True)
            raise
        finally:
            # Set running flag to False (signals threads to stop)
            # This allows controller and REST API threads to exit gracefully
            LOG.info("Shutting down system...")
            self.running = False
            
            # Give threads a moment to clean up
            time.sleep(2)


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

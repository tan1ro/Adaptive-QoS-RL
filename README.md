# Adaptive QoS using Reinforcement Learning

A research-grade Software Defined Network (SDN) project that uses Reinforcement Learning (RL) to dynamically adjust QoS parameters based on real-time traffic conditions.

## Overview

This project implements an adaptive Quality of Service (QoS) management system for Software Defined Networks using:
- **Ryu SDN Controller**: Manages network flows and applies QoS rules via OpenFlow (port 6653)
- **TensorFlow-based RL Agent**: Uses Deep Q-Network (DQN) to learn optimal QoS policies
- **Flask REST API**: Provides communication interface and web dashboard (port 8888)
- **Mininet**: For network emulation and testing
- **Real-time Web Dashboard**: Interactive UI for monitoring training progress and network metrics

The system continuously monitors network state (link utilization, queue length, delay, packet loss) and adjusts QoS parameters (bandwidth allocation, queue priorities) to optimize network performance.

## Features

- **Real-time Network Monitoring**: Collects flow statistics and network metrics from OpenFlow switches
- **Adaptive QoS Management**: Dynamically adjusts queue weights and bandwidth allocation
- **RL-based Decision Making**: Uses DQN to learn optimal QoS policies from experience
- **REST API Integration**: Communication between Ryu controller and RL agent
- **Interactive Web Dashboard**: Real-time visualization of training progress and network state
- **Automated Setup**: Single-command demo script that handles everything
- **Comprehensive Testing**: Unit tests for controller and agent components

## Project Structure

```
adaptive-qos-rl/
├── controller/
│   ├── qos_controller.py      # Ryu app: collects stats, applies QoS rules
│   ├── rest_api.py            # REST API server and web dashboard
│   └── __init__.py
├── agent/
│   ├── rl_agent.py            # TensorFlow-based RL agent (DQN)
│   ├── env.py                 # OpenAI Gym-like environment
│   ├── utils.py               # Preprocessing, reward computation
│   └── __init__.py
├── frontend/
│   ├── index.html             # Web dashboard HTML
│   ├── css/
│   │   └── style.css          # Dashboard styling
│   └── js/
│       └── dashboard.js       # Real-time updates and charts
├── config/
│   └── qos_config.yaml       # Queue priorities, thresholds, parameters
├── tests/
│   ├── test_qos_policy.py     # QoS policy tests
│   ├── test_rl_agent.py       # RL agent tests
│   └── __init__.py
├── models/                    # Saved trained models (created automatically)
├── main.py                    # Entry point: orchestrates controller and agent
├── start_demo.sh              # Single-command demo script (recommended)
├── setup.sh                   # Automated environment setup
├── fix_ryu_eventlet.sh       # Fix for Ryu compatibility issues
├── requirements.txt          # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- **Python 3.10+** (3.12 recommended, tested with 3.12)
- **Linux** (recommended) or Windows with WSL2
- **Mininet** (for network emulation)
- **Root/sudo access** (required for Mininet)

### Quick Setup

**The easiest way to get started:**

1. **Clone or download the project:**
   ```bash
   cd Adaptive-QoS-RL
   ```

2. **Run the automated setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   This script will:
   - Create a Python virtual environment
   - Install all dependencies (including TensorFlow and Ryu)
   - Apply compatibility patches for Ryu
   - Verify the installation

3. **You're ready!** The system is now set up and ready to run.

### Manual Setup (Alternative)

If you prefer manual setup:

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Mininet:**
   ```bash
   # On Ubuntu/Debian:
   sudo apt-get update
   sudo apt-get install mininet
   
   # Or build from source:
   git clone https://github.com/mininet/mininet.git
   cd mininet
   sudo ./util/install.sh -a
   ```

5. **Apply Ryu compatibility fix (if needed):**
   ```bash
   ./fix_ryu_eventlet.sh
   ```

## Usage

### Quick Start (Recommended)

**Single command to run everything:**

```bash
sudo ./start_demo.sh [episodes]
```

This script automatically:
- ✅ Checks all dependencies
- ✅ Activates the virtual environment
- ✅ Cleans up any existing processes
- ✅ Starts the Ryu SDN Controller (port 6653)
- ✅ Starts the REST API server (port 8888)
- ✅ Waits for services to be ready
- ✅ Starts Mininet network emulation
- ✅ Waits for switches to connect
- ✅ Starts RL agent training
- ✅ Cleans up on exit (Ctrl+C)

**Examples:**
```bash
# Run with 100 episodes (default)
sudo ./start_demo.sh

# Run with 500 episodes
sudo ./start_demo.sh 500

# Run with 1000 episodes for longer training
sudo ./start_demo.sh 1000
```

**Then open your browser to:** **http://localhost:8888**

The dashboard will show real-time training progress and network metrics!

### Manual Start (For Advanced Users)

If you want to start components manually:

**Step 1: Activate virtual environment**
```bash
source venv/bin/activate
```

**Step 2: Start Mininet (in separate terminal)**
```bash
sudo mn --controller=remote,ip=127.0.0.1,port=6653 --topo=linear,3
```

**Step 3: Start the system (in original terminal)**
```bash
python3 main.py --mode training --episodes 1000
```

The system will:
- Start Ryu controller on port 6653
- Start REST API on port 8888
- Wait for Mininet switches to connect
- Begin RL agent training

### Training Mode

Train the RL agent to learn optimal QoS policies:

```bash
# Via demo script (recommended)
sudo ./start_demo.sh 1000

# Or manually
python3 main.py --mode training --episodes 1000
```

The agent will:
- Observe network state
- Apply QoS actions
- Learn from rewards
- Save model checkpoints in `models/` directory

### Evaluation Mode

Evaluate a trained model:

```bash
python3 main.py --mode evaluation --episodes 10
```

This loads the trained model from `models/dqn_model_final.h5` and runs evaluation episodes without exploration.

## Web Dashboard

The system includes a comprehensive web dashboard accessible at:

**http://localhost:8888**

### Dashboard Features

- **Training Progress Panel**
  - Current episode number
  - Total episodes
  - Current reward
  - Average reward (last 100 episodes)
  - Epsilon value (exploration rate)
  - Training loss

- **Network State Panel**
  - Link utilization metrics
  - Queue lengths
  - Delay measurements
  - Packet loss rates
  - Available switch DPIDs

- **Flow Statistics Panel**
  - Real-time flow statistics from switches
  - Port-level metrics
  - Switch connection status

- **Interactive Charts**
  - Real-time reward graph (Chart.js)
  - Network metrics visualization
  - Auto-updating every second

- **Automatic Updates**
  - Polls API every second for fresh data
  - No page refresh needed
  - Responsive design

The dashboard is served directly from the REST API server and requires no additional setup.

## Architecture

### System Components

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Mininet   │──────│  Ryu Controller │──────│  REST API   │
│  (Switches) │      │  (OpenFlow)    │      │  (Flask)    │
└─────────────┘      └──────────────┘      └─────────────┘
                              │                     │
                              │                     │
                              ▼                     ▼
                         ┌──────────────────────────┐
                         │   RL Agent (DQN)         │
                         │   TensorFlow/Keras       │
                         └──────────────────────────┘
```

### Component Details

**1. QoS Controller (`controller/qos_controller.py`)**
- Ryu SDN application
- Collects flow statistics from OpenFlow switches
- Monitors link utilization, queue lengths, delays, packet loss
- Applies QoS rules (queue configurations, rate limits) based on RL agent actions
- Exposes network state via REST API

**2. REST API (`controller/rest_api.py`)**
- Flask-based web server
- Endpoints:
  - `GET /api/v1/health` - Health check
  - `GET /api/v1/state` - Get current network state
  - `GET /api/v1/stats` - Get flow statistics
  - `POST /api/v1/qos/apply` - Apply QoS action from RL agent
  - `GET /api/v1/training/metrics` - Get training metrics
  - `POST /api/v1/training/metrics` - Update training metrics
- Serves web dashboard from `frontend/` directory

**3. RL Environment (`agent/env.py`)**
- OpenAI Gym-compatible environment
- State space: `[link_utilization, queue_length, delay, packet_loss]`
- Action space: Discrete actions for QoS adjustments (9 configurations)
- Reward function: `reward = -(delay + packet_loss) + α * throughput`

**4. RL Agent (`agent/rl_agent.py`)**
- **DQN (Deep Q-Network)** implementation
- Experience replay buffer for stable learning
- Target network for stable Q-value estimation
- Epsilon-greedy exploration strategy
- Model saving/loading support

**5. Main Orchestrator (`main.py`)**
- Coordinates all components
- Manages lifecycle of controller and agent
- Handles graceful shutdown
- Comprehensive logging

## Configuration

Edit `config/qos_config.yaml` to customize:

- **Queue Priorities**: Priority levels for different traffic classes
- **Rate Limits**: Minimum and maximum bandwidth allocations
- **Network Thresholds**: Utilization, delay, and loss thresholds
- **RL Parameters**: Update intervals, reward weights
- **Topology Settings**: Switch and link configurations

## Testing

Run unit tests:

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run all tests
python3 -m pytest tests/

# Run specific test file
python3 -m pytest tests/test_rl_agent.py -v

# Run with coverage
python3 -m pytest tests/ --cov=controller --cov=agent
```

## Expected Behavior

### Startup Sequence

When running `./start_demo.sh`, you should see:

1. **Dependency checks** ✓
2. **Virtual environment activation** ✓
3. **Cleanup of existing processes** ✓
4. **Controller initialization**
   - OpenFlow server starting on port 6653
   - REST API server starting on port 8888
5. **Wait for services** (with progress indicators)
6. **Mininet startup**
7. **Switch connection** (switches connect to controller)
8. **Training begins**
   - Episode progress
   - Reward tracking
   - Model updates

### Training Output Example

```
2025-11-03 10:00:00 - INFO - Starting RL agent training for 1000 episodes...
2025-11-03 10:00:05 - INFO - Controller REST API is ready and responding
2025-11-03 10:00:10 - INFO - Episode 1/1000 - Score: -5.23, Avg (last 100): -5.23, Epsilon: 1.000, Steps: 100
2025-11-03 10:00:15 - INFO - Episode 2/1000 - Score: -4.87, Avg (last 100): -5.05, Epsilon: 0.995, Steps: 100
2025-11-03 10:00:20 - INFO - Applied QoS rule: dpid=1, queue=0, min_rate=10000, max_rate=100000
...
```

### Dashboard Metrics

The dashboard displays:
- **Throughput**: Link utilization and bandwidth usage
- **Latency**: End-to-end delay measurements
- **Packet Loss**: Dropped packet rates
- **QoS Effectiveness**: Reward improvements over time
- **Training Progress**: Episode count and learning metrics

## Troubleshooting

### Common Issues

**1. "externally-managed-environment" error**
- **Solution**: Use the virtual environment as shown in the setup instructions
- The `setup.sh` script handles this automatically

**2. "No matching distribution found for tensorflow==2.15.0"**
- **Cause**: Python 3.12 compatibility
- **Solution**: `requirements.txt` has been updated to use `tensorflow>=2.16.0` which supports Python 3.12

**3. "ImportError: cannot import name 'ALREADY_HANDLED' from 'eventlet.wsgi'"**
- **Cause**: Ryu 4.34 compatibility with newer eventlet versions
- **Solution**: Run `./fix_ryu_eventlet.sh` or the `setup.sh` script handles this automatically

**4. Controller REST API not starting**
- **Check**: Port 8888 is not in use: `netstat -tln | grep 8888`
- **Check**: Controller logs in `/tmp/adaptive_qos.log`
- **Check**: Virtual environment is activated
- **Solution**: The `start_demo.sh` script waits up to 60 seconds for the API to start

**5. OpenFlow server not listening on port 6653**
- **Check**: Port 6653 is not in use: `netstat -tln | grep 6653`
- **Check**: Ryu controller started successfully (check logs)
- **Solution**: The `start_demo.sh` script waits up to 30 seconds for the OpenFlow server

**6. Switches not connecting**
- **Check**: Mininet started successfully
- **Check**: Controller is listening on port 6653 before Mininet starts
- **Solution**: The script ensures controller is ready before starting Mininet

**7. Training completes immediately**
- **Cause**: REST API not ready when training starts
- **Solution**: The system now waits for REST API before starting training

**8. "Switch X not found" warnings**
- **Note**: This is normal if switches haven't connected yet
- The controller dynamically discovers switches as they connect
- The system will use any available switch

### Debug Commands

```bash
# Check if services are running
netstat -tln | grep -E "(6653|8888)"

# View logs
tail -f /tmp/adaptive_qos.log

# Check Python processes
ps aux | grep python | grep main.py

# Check Mininet
sudo mn -c  # Cleanup Mininet
sudo mn --test pingall  # Test Mininet

# Test REST API
curl http://localhost:8888/api/v1/health

# Test controller state
curl http://localhost:8888/api/v1/state
```

## Performance Considerations

- **State Collection Frequency**: Balance between responsiveness and overhead
- **Action Update Frequency**: Too frequent updates may cause instability
- **Replay Buffer Size**: Larger buffers improve stability but use more memory
- **Network Architecture**: Deeper networks may overfit, simpler networks may underfit
- **Training Episodes**: More episodes = better learning but longer time

## Advanced Usage

### Custom Environment

Modify `agent/env.py` to customize:
- State representation
- Action space
- Reward function
- Episode termination conditions

### Custom RL Algorithm

Extend `agent/rl_agent.py` to implement:
- Different neural network architectures
- Alternative RL algorithms (PPO, A3C, SAC, etc.)
- Multi-agent scenarios

### Integration with Real Networks

To use with physical switches:
1. Update controller to support your switch's OpenFlow version
2. Configure switch IP addresses in configuration
3. Adjust state collection mechanisms for hardware limits
4. Test with one switch first before scaling

### Custom Topology

Modify Mininet topology in `start_demo.sh`:
```bash
# Change this line in start_demo.sh
MININET_TOPO="tree,depth=2,fanout=3"  # Tree topology
MININET_TOPO="linear,5"                # Linear with 5 switches
MININET_TOPO="single,3"                # Single switch with 3 hosts
```

## Future Enhancements

- [x] Visualization dashboard for real-time metrics ✓
- [x] Automated setup and demo script ✓
- [x] Dynamic switch discovery ✓
- [ ] Multi-switch topology optimization
- [ ] PPO implementation completion
- [ ] Distributed training for multiple network domains
- [ ] Transfer learning for different network topologies
- [ ] Integration with other SDN controllers (ONOS, OpenDaylight)
- [ ] GPU acceleration for training
- [ ] Advanced visualization (network topology graphs)

## Technical Details

### Ports Used
- **6653**: OpenFlow controller (Ryu)
- **8888**: REST API and web dashboard (Flask)

### File Locations
- **Logs**: `/tmp/adaptive_qos.log`
- **Models**: `models/dqn_model_*.h5`
- **Config**: `config/qos_config.yaml`

### Environment Variables
The system sets TensorFlow environment variables to prevent CUDA compilation issues:
- `TF_XLA_FLAGS=--tf_xla_cpu_global_jit`
- `XLA_FLAGS=--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found`

These are set automatically by `start_demo.sh`.

## References

- [Ryu SDN Framework](https://ryu.readthedocs.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenFlow Specification](https://opennetworking.org/wp-content/uploads/2014/10/openflow-spec-v1.3.0.pdf)
- [Mininet Documentation](http://mininet.org/)
- [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602)

## License

This project is intended for research and educational purposes.

## Authors

Research-grade SDN project for adaptive QoS management using Reinforcement Learning.

## Acknowledgments

Built using:
- Ryu SDN Framework
- TensorFlow/Keras
- Mininet
- Flask
- Chart.js
- OpenAI Gym

---

**Ready to get started?** Run `sudo ./start_demo.sh` and open **http://localhost:8888** in your browser!

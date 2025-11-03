# Adaptive QoS using Reinforcement Learning

A research-grade Software Defined Network (SDN) project that uses Reinforcement Learning (RL) to dynamically adjust QoS parameters based on real-time traffic conditions.

## Overview

This project implements an adaptive Quality of Service (QoS) management system for Software Defined Networks using:
- **Ryu SDN Controller**: Manages network flows and applies QoS rules via OpenFlow
- **TensorFlow-based RL Agent**: Uses Deep Q-Network (DQN) to learn optimal QoS policies
- **Mininet**: For network emulation and testing

The system continuously monitors network state (link utilization, queue length, delay, packet loss) and adjusts QoS parameters (bandwidth allocation, queue priorities) to optimize network performance.

## Features

- **Real-time Network Monitoring**: Collects flow statistics and network metrics
- **Adaptive QoS Management**: Dynamically adjusts queue weights and bandwidth allocation
- **RL-based Decision Making**: Uses DQN to learn optimal QoS policies
- **REST API Integration**: Communication between Ryu controller and RL agent
- **Comprehensive Testing**: Unit tests for controller and agent components

## Project Structure

```
adaptive-qos-rl/
├── controller/
│   ├── qos_controller.py      # Ryu app: collects stats, applies QoS rules
│   ├── rest_api.py            # REST API for agent interaction
│   └── __init__.py
├── agent/
│   ├── rl_agent.py            # TensorFlow-based RL agent (DQN/PPO)
│   ├── env.py                 # OpenAI Gym-like environment
│   ├── utils.py               # Preprocessing, reward computation
│   └── __init__.py
├── tests/
│   ├── test_qos_policy.py     # QoS policy tests
│   ├── test_rl_agent.py       # RL agent tests
│   └── __init__.py
├── config/
│   └── qos_config.yaml        # Queue priorities, thresholds, parameters
├── main.py                    # Entry point: orchestrates controller and agent
├── requirements.txt           # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- Linux (recommended) or Windows with WSL
- Mininet (for network emulation)
- OpenFlow-compatible switches

### Setup

1. **Clone or create the project directory:**
   ```bash
   cd adaptive-qos-rl
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ryu SDN Controller:**
   ```bash
   pip install ryu
   # Or from source: git clone https://github.com/faucetsdn/ryu.git
   ```

4. **Install Mininet (for network emulation):**
   ```bash
   # On Ubuntu/Debian:
   sudo apt-get install mininet
   
   # Or build from source:
   git clone https://github.com/mininet/mininet.git
   cd mininet
   sudo ./util/install.sh -a
   ```

5. **Install TensorFlow:**
   ```bash
   pip install tensorflow==2.15.0
   ```

## Configuration

Edit `config/qos_config.yaml` to configure:
- Queue priorities and rate limits
- Network state thresholds
- RL parameters (update intervals, reward weights)
- Topology settings

## Usage

### 1. Start Mininet Network (in one terminal)

```bash
sudo mn --controller=remote,ip=127.0.0.1,port=6653 --topo=linear,3
```

This creates a simple linear topology with 3 switches and connects to the Ryu controller.

### 2. Run the Complete System

#### Training Mode

Train the RL agent to learn optimal QoS policies:

```bash
python main.py --mode training --episodes 1000
```

#### Evaluation Mode

Evaluate a trained model:

```bash
python main.py --mode evaluation --episodes 10
```

### 3. Monitor Logs

The system provides real-time logs showing:
- Network state updates
- QoS actions applied
- Training progress (episode rewards, epsilon decay)
- Controller statistics

### 4. Generate Traffic (optional)

In Mininet, generate traffic to test the adaptive QoS:

```bash
mininet> h1 ping h2
mininet> h1 iperf -s -u &
mininet> h2 iperf -c h1 -u -b 10M -t 60
```

## Architecture

### Controller Component (`controller/qos_controller.py`)

- Collects flow statistics from OpenFlow switches
- Monitors link utilization, queue lengths, delays, and packet loss
- Applies QoS rules (queue configurations, rate limits) based on RL agent actions
- Exposes network state via REST API

### REST API (`controller/rest_api.py`)

Endpoints:
- `GET /api/v1/state` - Get current network state
- `GET /api/v1/stats` - Get flow statistics
- `POST /api/v1/qos/apply` - Apply QoS action from RL agent
- `GET /api/v1/health` - Health check

### RL Environment (`agent/env.py`)

- State space: `[link_utilization, queue_length, delay, packet_loss]`
- Action space: Discrete actions for QoS adjustments (queue ID, rate limits)
- Reward function: `reward = -(delay + packet_loss) + α * throughput`

### RL Agent (`agent/rl_agent.py`)

- **DQN (Deep Q-Network)**: Primary implementation
  - Experience replay buffer
  - Target network for stable training
  - Epsilon-greedy exploration
- **PPO (Proximal Policy Optimization)**: Alternative implementation (optional)

## Testing

Run unit tests:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_rl_agent.py -v

# Run with coverage
python -m pytest tests/ --cov=controller --cov=agent
```

## Expected Outputs

### Training Output

```
2024-01-01 10:00:00 - INFO - Starting RL agent training for 1000 episodes...
2024-01-01 10:00:05 - INFO - Controller is ready
2024-01-01 10:00:10 - INFO - Episode 1/1000 - Score: -5.23, Avg (last 100): -5.23, Epsilon: 0.995, Steps: 100
2024-01-01 10:00:15 - INFO - Episode 2/1000 - Score: -4.87, Avg (last 100): -5.05, Epsilon: 0.990, Steps: 100
...
```

### Real-time QoS Adjustment Logs

```
2024-01-01 10:00:20 - INFO - Applied QoS rule: dpid=1, queue=0, min_rate=10000, max_rate=100000
2024-01-01 10:00:25 - INFO - Link utilization: 0.65, Queue length: 1200, Delay: 25ms
```

### Metrics

The system tracks:
- **Throughput**: Link utilization and bandwidth usage
- **Latency**: End-to-end delay measurements
- **Packet Loss**: Dropped packet rates
- **QoS Effectiveness**: Reward improvements over time

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
- Alternative RL algorithms (A3C, SAC, etc.)
- Multi-agent scenarios

### Integration with Real Networks

To use with physical switches:
1. Update controller to support your switch's OpenFlow version
2. Configure switch IP addresses in configuration
3. Adjust state collection mechanisms for hardware limits

## Troubleshooting

### Controller Not Starting

- Ensure Ryu is properly installed: `ryu-manager --version`
- Check for port conflicts (6653 for OpenFlow, 8080 for REST API)
- Verify Python dependencies are installed

### Agent Not Connecting to Controller

- Check REST API is running: `curl http://localhost:8080/api/v1/health`
- Verify firewall settings allow local connections
- Check controller logs for errors

### Training Not Converging

- Adjust learning rate and epsilon decay in `rl_agent.py`
- Modify reward function weights in `utils.py`
- Increase replay buffer size
- Adjust network architecture (add/remove layers)

## Performance Considerations

- **State Collection Frequency**: Balance between responsiveness and overhead
- **Action Update Frequency**: Too frequent updates may cause instability
- **Replay Buffer Size**: Larger buffers improve stability but use more memory
- **Network Architecture**: Deeper networks may overfit, simpler networks may underfit

## Future Enhancements

- [ ] Multi-switch topology support
- [ ] PPO implementation completion
- [ ] Visualization dashboard for real-time metrics
- [ ] Distributed training for multiple network domains
- [ ] Transfer learning for different network topologies
- [ ] Integration with SDN controllers (ONOS, OpenDaylight)

## References

- [Ryu SDN Framework](https://ryu.readthedocs.io/)
- [TensorFlow RL Guide](https://www.tensorflow.org/agents)
- [OpenFlow Specification](https://opennetworking.org/wp-content/uploads/2014/10/openflow-spec-v1.3.0.pdf)
- [Mininet Documentation](http://mininet.org/)

## License

This project is intended for research and educational purposes.

## Authors

Research-grade SDN project for adaptive QoS management using Reinforcement Learning.

## Acknowledgments

Built using:
- Ryu SDN Framework
- TensorFlow
- Mininet
- OpenAI Gym




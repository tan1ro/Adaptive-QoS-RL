#!/bin/bash
# Example script to run the Adaptive QoS RL system

echo "Starting Adaptive QoS using Reinforcement Learning..."
echo ""

# Check if running as root (needed for Mininet)
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo) for Mininet"
    exit 1
fi

# Start Mininet in background
echo "Starting Mininet topology..."
sudo mn --controller=remote,ip=127.0.0.1,port=6653 --topo=linear,3 &
MININET_PID=$!
sleep 5

# Run the main system
echo "Starting Ryu controller and RL agent..."
python main.py --mode training --episodes 100

# Cleanup
echo "Cleaning up..."
sudo kill $MININET_PID
sudo mn -c

echo "Done!"




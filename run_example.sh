#!/bin/bash
###############################################################################
# Adaptive QoS RL - Simple Example Script
# This is a simpler alternative to start_demo.sh for quick testing
# Usage: sudo ./run_example.sh [episodes]
# 
# NOTE: For the full-featured demo with web dashboard, use: sudo ./start_demo.sh
###############################################################################

# Configuration
EPISODES=${1:-100}  # Default to 100 episodes if not specified

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "Starting Adaptive QoS using Reinforcement Learning..."
echo ""

# Check if running as root (needed for Mininet)
if [ "$EUID" -ne 0 ]; then 
    print_error "This script must be run as root (sudo) for Mininet"
    echo ""
    print_info "Please run one of the following:"
    echo "  1. Full demo (recommended): sudo ./start_demo.sh $EPISODES"
    echo "  2. Simple example: sudo ./run_example.sh $EPISODES"
    echo ""
    print_info "Note: Mininet requires root privileges to create virtual network interfaces"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    print_info "Activating virtual environment..."
    source venv/bin/activate
    if [ $? -eq 0 ]; then
        print_success "Virtual environment activated"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi
else
    print_error "Virtual environment not found!"
    print_info "Please run ./setup.sh first to create the virtual environment"
    exit 1
fi

# Set TensorFlow environment variables to prevent CUDA issues
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export XLA_FLAGS=--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found

# Cleanup function
cleanup() {
    print_info "Cleaning up..."
    if [ ! -z "$MININET_PID" ]; then
        kill $MININET_PID 2>/dev/null || true
    fi
    sudo mn -c 2>/dev/null || true
    pkill -f "python.*main.py" 2>/dev/null || true
    print_success "Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Clean up any existing Mininet instances
print_info "Cleaning up any existing Mininet instances..."
sudo mn -c > /dev/null 2>&1 || true

# Start Mininet in background
print_info "Starting Mininet topology (linear,3)..."
print_info "Controller should be at 127.0.0.1:6653"
sudo mn --controller=remote,ip=127.0.0.1,port=6653 --topo=linear,3 > /tmp/mininet.log 2>&1 &
MININET_PID=$!

# Wait for Mininet to start
sleep 3

if ps -p $MININET_PID > /dev/null; then
    print_success "Mininet started (PID: $MININET_PID)"
else
    print_error "Failed to start Mininet"
    print_info "Check /tmp/mininet.log for errors"
    exit 1
fi

# Wait a bit for switches to initialize
print_info "Waiting for switches to connect to controller..."
sleep 5

# Run the main system
print_info "Starting Ryu controller and RL agent..."
print_info "This will start:"
echo "  - Ryu SDN Controller (port 6653)"
echo "  - REST API Server (port 8888)"
echo "  - RL Agent Training ($EPISODES episodes)"
echo ""

print_info "Training will begin once controller is ready..."
print_warning "Press Ctrl+C to stop the system"
echo ""

# Run main.py (this will block until training completes or interrupted)
python3 main.py --mode training --episodes $EPISODES

# Note: Cleanup happens automatically via trap
print_success "Done!"

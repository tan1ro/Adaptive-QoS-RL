#!/bin/bash
###############################################################################
# Adaptive QoS RL - Complete Demo Startup Script
# This script starts the entire system including Mininet, Controller, and Training
# Usage: ./start_demo.sh [episodes]
###############################################################################

set -e  # Exit on error

# Configuration
EPISODES=${1:-100}  # Default to 100 episodes if not specified
MININET_TOPO="linear,3"  # Mininet topology
CONTROLLER_PORT=6653
REST_API_PORT=8080

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

# Check if running as root (needed for Mininet)
check_root() {
    if [ "$EUID" -ne 0 ]; then 
        print_error "This script must be run as root (sudo) for Mininet"
        print_info "Please run: sudo ./start_demo.sh $EPISODES"
        exit 1
    fi
}

# Activate virtual environment
activate_venv() {
    if [ -d "venv" ]; then
        print_info "Activating virtual environment..."
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found!"
        print_info "Please run ./setup.sh first to create the virtual environment"
        exit 1
    fi
}

# Check if dependencies are installed
check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! python3 -c "import ryu" 2>/dev/null; then
        print_error "Ryu not installed"
        print_info "Run: ./fix_ryu_eventlet.sh (if needed) and verify installation"
        exit 1
    fi
    
    if ! python3 -c "import tensorflow" 2>/dev/null; then
        print_error "TensorFlow not installed"
        print_info "Please run: pip install -r requirements.txt"
        exit 1
    fi
    
    print_success "All dependencies checked"
}

# Cleanup function
cleanup() {
    print_info "Cleaning up..."
    
    # Kill Mininet if running
    if [ ! -z "$MININET_PID" ]; then
        print_info "Stopping Mininet (PID: $MININET_PID)..."
        kill $MININET_PID 2>/dev/null || true
        sleep 2
    fi
    
    # Clean Mininet
    print_info "Cleaning Mininet..."
    mn -c 2>/dev/null || true
    
    # Kill Python processes
    print_info "Stopping Python processes..."
    pkill -f "python.*main.py" 2>/dev/null || true
    
    print_success "Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Start Mininet
start_mininet() {
    print_info "Starting Mininet topology ($MININET_TOPO)..."
    print_info "Controller will connect on port $CONTROLLER_PORT"
    
    # Start Mininet in background
    mn --controller=remote,ip=127.0.0.1,port=$CONTROLLER_PORT --topo=$MININET_TOPO > /tmp/mininet.log 2>&1 &
    MININET_PID=$!
    
    print_success "Mininet started (PID: $MININET_PID)"
    print_info "Waiting 5 seconds for Mininet to initialize..."
    sleep 5
    
    # Verify Mininet is running
    if ! kill -0 $MININET_PID 2>/dev/null; then
        print_error "Mininet failed to start. Check /tmp/mininet.log for details"
        exit 1
    fi
}

# Wait for controller to be ready
wait_for_controller() {
    print_info "Waiting for controller REST API to be ready..."
    local max_wait=30
    local waited=0
    
    while [ $waited -lt $max_wait ]; do
        if curl -s http://localhost:$REST_API_PORT/api/v1/health > /dev/null 2>&1; then
            print_success "Controller REST API is ready!"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
        echo -n "."
    done
    
    echo ""
    print_warning "Controller may not be ready, but continuing..."
}

# Main execution
main() {
    clear
    echo "=================================================================="
    echo "  Adaptive QoS RL - Complete Demonstration System"
    echo "=================================================================="
    echo ""
    print_info "Starting complete demo with $EPISODES episodes"
    echo ""
    
    # Pre-flight checks
    check_root
    activate_venv
    check_dependencies
    
    # Cleanup any existing instances
    cleanup
    
    # Start Mininet
    start_mininet
    
    # Start the main system
    print_info "Starting Adaptive QoS RL system..."
    print_info "This will start:"
    echo "  - Ryu SDN Controller"
    echo "  - REST API Server (port $REST_API_PORT)"
    echo "  - RL Agent Training ($EPISODES episodes)"
    echo ""
    print_info "Access the dashboard at: http://localhost:$REST_API_PORT"
    echo ""
    print_warning "Press Ctrl+C to stop the system"
    echo ""
    
    # Wait a moment for everything to initialize
    sleep 2
    
    # Run the main system
    python3 main.py --mode training --episodes $EPISODES
    
    print_success "Demo completed!"
}

# Run main function
main


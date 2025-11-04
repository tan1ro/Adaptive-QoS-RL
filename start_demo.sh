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
REST_API_PORT=8888

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
    
    # Kill main Python process if running
    if [ ! -z "$MAIN_PID" ]; then
        print_info "Stopping main system (PID: $MAIN_PID)..."
        kill $MAIN_PID 2>/dev/null || true
        sleep 2
    fi
    
    # Kill Mininet if running
    if [ ! -z "$MININET_PID" ]; then
        print_info "Stopping Mininet (PID: $MININET_PID)..."
        kill $MININET_PID 2>/dev/null || true
        sleep 2
    fi
    
    # Clean Mininet
    print_info "Cleaning Mininet..."
    mn -c 2>/dev/null || true
    
    # Kill any remaining Python processes
    print_info "Stopping any remaining Python processes..."
    pkill -f "python.*main.py" 2>/dev/null || true
    
    print_success "Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Start Mininet
start_mininet() {
    print_info "Starting Mininet topology ($MININET_TOPO)..."
    print_info "Controller should be listening on port $CONTROLLER_PORT"
    
    # Check if controller is listening on port 6653
    if ! netstat -tln 2>/dev/null | grep -q ":$CONTROLLER_PORT "; then
        print_warning "Controller may not be listening on port $CONTROLLER_PORT yet"
        print_info "Mininet will retry connection automatically"
    else
        print_success "Controller is listening on port $CONTROLLER_PORT"
    fi
    
    # Start Mininet in background
    # Mininet will automatically retry connecting to controller
    mn --controller=remote,ip=127.0.0.1,port=$CONTROLLER_PORT --topo=$MININET_TOPO > /tmp/mininet.log 2>&1 &
    MININET_PID=$!
    
    print_success "Mininet started (PID: $MININET_PID)"
    print_info "Waiting 5 seconds for Mininet switches to initialize and connect..."
    sleep 5
    
    # Verify Mininet is running
    if ! kill -0 $MININET_PID 2>/dev/null; then
        print_error "Mininet failed to start. Check /tmp/mininet.log for details"
        exit 1
    fi
    
    # Check log for connection status
    if grep -q "Connecting to controller" /tmp/mininet.log 2>/dev/null; then
        print_info "Mininet is attempting to connect switches..."
    fi
}

# Wait for controller to be ready
wait_for_controller() {
    print_info "Waiting for controller REST API to be ready..."
    local max_wait=60  # Increased timeout to 60 seconds
    local waited=0
    
    # First check if the process is running
    local process_running=false
    for i in {1..10}; do
        if kill -0 $MAIN_PID 2>/dev/null; then
            process_running=true
            break
        fi
        sleep 1
    done
    
    if [ "$process_running" = false ]; then
        print_error "Main process (PID: $MAIN_PID) is not running!"
        print_info "Check /tmp/adaptive_qos.log for errors:"
        tail -20 /tmp/adaptive_qos.log 2>/dev/null || echo "Log file not found"
        return 1
    fi
    
    # Now wait for REST API to respond
    while [ $waited -lt $max_wait ]; do
        # Try health endpoint
        if curl -s --connect-timeout 2 http://localhost:$REST_API_PORT/api/v1/health > /dev/null 2>&1; then
            echo ""
            print_success "Controller REST API is ready!"
            return 0
        fi
        # Also check if port is open (indicates server is starting)
        if netstat -tln 2>/dev/null | grep -q ":$REST_API_PORT " || \
           ss -tln 2>/dev/null | grep -q ":$REST_API_PORT "; then
            # Port is open, give it a moment more
            sleep 2
            if curl -s --connect-timeout 2 http://localhost:$REST_API_PORT/api/v1/health > /dev/null 2>&1; then
                echo ""
                print_success "Controller REST API is ready!"
                return 0
            fi
        fi
        sleep 1
        waited=$((waited + 1))
        if [ $((waited % 5)) -eq 0 ]; then
            echo -n "$waited"
        else
            echo -n "."
        fi
    done
    
    echo ""
    print_warning "Controller REST API did not become ready within $max_wait seconds"
    print_info "Checking logs..."
    tail -30 /tmp/adaptive_qos.log 2>/dev/null | tail -10 || echo "Log file not accessible"
    print_warning "Continuing anyway - controller may start later..."
    return 0  # Don't fail, just warn
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
    
    # Set TensorFlow environment variables early (before starting anything)
    # These disable XLA GPU compilation which requires ptxas (CUDA toolkit)
    # Using CPU fallback instead for better compatibility
    export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
    export XLA_FLAGS=--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found
    
    # Alternative: Force CPU if GPU issues persist (uncomment if needed)
    # export CUDA_VISIBLE_DEVICES=""
    
    # Start the main system FIRST (so controller is listening before Mininet connects)
    print_info "Starting Adaptive QoS RL system..."
    print_info "This will start:"
    echo "  - Ryu SDN Controller (port $CONTROLLER_PORT)"
    echo "  - REST API Server (port $REST_API_PORT)"
    echo "  - RL Agent Training ($EPISODES episodes)"
    echo ""
    
    # Start system in background so we can monitor and start Mininet after controller is ready
    python3 main.py --mode training --episodes $EPISODES > /tmp/adaptive_qos.log 2>&1 &
    MAIN_PID=$!
    
    # Wait for controller REST API to be ready (indicates controller started)
    print_info "Waiting for controller to initialize..."
    wait_for_controller
    
    # Wait for OpenFlow server to bind to port 6653
    # This is critical - Mininet needs the controller to be listening
    print_info "Waiting for OpenFlow server to bind to port $CONTROLLER_PORT..."
    max_wait=30
    waited=0
    while [ $waited -lt $max_wait ]; do
        if netstat -tln 2>/dev/null | grep -q ":$CONTROLLER_PORT " || \
           ss -tln 2>/dev/null | grep -q ":$CONTROLLER_PORT "; then
            echo ""
            print_success "OpenFlow controller is listening on port $CONTROLLER_PORT"
            break
        fi
        sleep 1
        waited=$((waited + 1))
        if [ $((waited % 5)) -eq 0 ]; then
            echo -n "$waited"
        else
            echo -n "."
        fi
    done
    
    if [ $waited -ge $max_wait ]; then
        echo ""
        print_warning "OpenFlow controller may not be listening yet (will retry)"
        print_info "Checking if controller process is running..."
        ps aux | grep -E "python.*main.py" | grep -v grep || print_warning "Controller process not found"
    fi
    
    # NOW start Mininet (controller should be listening)
    start_mininet
    
    # Wait for switches to connect to controller
    print_info "Waiting for switches to connect to controller..."
    sleep 8  # Give switches time to connect
    
    print_info "Access the dashboard at: http://localhost:$REST_API_PORT"
    echo ""
    print_warning "Press Ctrl+C to stop the system"
    echo ""
    print_info "Main system PID: $MAIN_PID"
    print_info "Mininet PID: $MININET_PID"
    echo ""
    print_info "System is running. Check logs at /tmp/adaptive_qos.log"
    echo ""
    
    # Wait for main process to complete
    print_info "Waiting for training to complete..."
    wait $MAIN_PID
    MAIN_EXIT_CODE=$?
    
    if [ $MAIN_EXIT_CODE -eq 0 ]; then
        print_success "Demo completed successfully!"
    else
        print_error "Demo ended with exit code $MAIN_EXIT_CODE"
        print_info "Check /tmp/adaptive_qos.log for details"
    fi
}

# Run main function
main


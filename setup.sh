#!/bin/bash
# Setup script for Adaptive QoS RL project
# Creates a virtual environment and installs dependencies

set -e  # Exit on error

echo "Setting up Adaptive QoS RL environment..."
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Found Python version: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created!"
else
    echo "Virtual environment already exists, skipping creation..."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install Ryu with compatibility patch (for Python 3.12)
echo "Installing Ryu (with compatibility patch for Python 3.12)..."
if ! pip show ryu &> /dev/null; then
    echo "Downloading and patching Ryu source..."
    cd /tmp
    rm -rf ryu-source
    git clone --depth 1 --branch v4.34 https://github.com/faucetsdn/ryu.git ryu-source 2>/dev/null || \
    git clone --depth 1 https://github.com/faucetsdn/ryu.git ryu-source
    
    # Apply compatibility patches
    if [ -f "ryu-source/ryu/hooks.py" ]; then
        echo "Applying compatibility patch for setuptools (hooks.py)..."
        sed -i '36s/.*/    # Handle compatibility with newer setuptools where get_script_args may not exist\n    if hasattr(easy_install, '\''get_script_args'\''):\n        _main_module()._orig_get_script_args = easy_install.get_script_args\n    else:\n        # Fallback for newer setuptools versions\n        def _dummy_get_script_args(*args, **kwargs):\n            return []\n        _main_module()._orig_get_script_args = _dummy_get_script_args/' ryu-source/ryu/hooks.py
        sed -i '67s/.*/    if hasattr(easy_install, '\''get_script_args'\''):\n        easy_install.get_script_args = my_get_script_args/' ryu-source/ryu/hooks.py
    fi
    
    # Apply eventlet compatibility patch
    if [ -f "ryu-source/ryu/app/wsgi.py" ]; then
        echo "Applying compatibility patch for eventlet (wsgi.py)..."
        python3 << 'EOF'
import re

wsgi_path = "ryu-source/ryu/app/wsgi.py"
with open(wsgi_path, 'r') as f:
    content = f.read()

# Replace the ALREADY_HANDLED import with a try/except block
old_pattern = r'class _AlreadyHandledResponse\(Response\):\n    # XXX: Eventlet API should not be used directly\.\n    from eventlet\.wsgi import ALREADY_HANDLED\n    _ALREADY_HANDLED = ALREADY_HANDLED\n\n    def __call__\(self, environ, start_response\):\n        return self\._ALREADY_HANDLED'

new_code = '''class _AlreadyHandledResponse(Response):
    # XXX: Eventlet API should not be used directly.
    # Compatibility fix: ALREADY_HANDLED was removed in newer eventlet versions
    try:
        from eventlet.wsgi import ALREADY_HANDLED
        _ALREADY_HANDLED = ALREADY_HANDLED
    except ImportError:
        # Fallback for newer eventlet versions (0.40+)
        # Return empty byte string as response has already been handled
        _ALREADY_HANDLED = b''

    def __call__(self, environ, start_response):
        return self._ALREADY_HANDLED'''

content = re.sub(old_pattern, new_code, content, flags=re.MULTILINE)

with open(wsgi_path, 'w') as f:
    f.write(content)
EOF
    fi
    
    cd - > /dev/null
    pip install /tmp/ryu-source
    echo "Ryu installed successfully!"
else
    echo "Ryu already installed, skipping..."
fi

# Install remaining dependencies
echo ""
echo "Installing remaining dependencies from requirements.txt..."
echo "Note: TensorFlow is a large package (~600MB), this may take a few minutes..."
cd /home/taniro/Adaptive-QoS-RL
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Or use the run script which will activate it automatically:"
echo "  ./run_example.sh"
echo ""

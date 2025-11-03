#!/bin/bash
# Fix Ryu eventlet compatibility issue
# This script patches the installed Ryu to work with newer eventlet versions

set -e

echo "Fixing Ryu eventlet compatibility..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Find Ryu wsgi.py file
RYU_WSGI=$(python3 -c "import ryu.app.wsgi; import os; print(os.path.dirname(ryu.app.wsgi.__file__))")/wsgi.py

if [ ! -f "$RYU_WSGI" ]; then
    echo "Error: Could not find Ryu wsgi.py file"
    exit 1
fi

echo "Found Ryu wsgi.py at: $RYU_WSGI"

# Check if already patched
if grep -q "_ALREADY_HANDLED = b'" "$RYU_WSGI"; then
    echo "Already patched, skipping..."
    exit 0
fi

# Apply patch
python3 << EOF
import re

wsgi_path = "$RYU_WSGI"
with open(wsgi_path, 'r') as f:
    content = f.read()

# Replace old-style import with try/except
old_pattern = r'    from eventlet\.wsgi import ALREADY_HANDLED\n    _ALREADY_HANDLED = ALREADY_HANDLED'

new_code = '''    # Compatibility fix: ALREADY_HANDLED was removed in newer eventlet versions
    try:
        from eventlet.wsgi import ALREADY_HANDLED
        _ALREADY_HANDLED = ALREADY_HANDLED
    except ImportError:
        # Fallback for newer eventlet versions (0.40+)
        # Return empty byte string as response has already been handled
        _ALREADY_HANDLED = b\'\''''

if re.search(old_pattern, content):
    content = re.sub(old_pattern, new_code, content)
    with open(wsgi_path, 'w') as f:
        f.write(content)
    print("Patch applied successfully!")
else:
    print("Pattern not found. Ryu may already be patched or has a different structure.")
EOF

echo "Done!"


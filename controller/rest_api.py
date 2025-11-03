"""
REST API for communication between Ryu controller and RL agent
"""

from flask import Flask, jsonify, request
import threading
import logging

LOG = logging.getLogger(__name__)

# Try to import CORS, fallback if not available
try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False
    LOG.warning("flask-cors not available, CORS disabled")


class RESTAPI:
    """
    REST API server for controller-agent communication
    """
    
    def __init__(self, controller, host='0.0.0.0', port=8080):
        self.controller = controller
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        if HAS_CORS:
            CORS(self.app)  # Enable CORS for agent requests
        else:
            # Add manual CORS headers
            @self.app.after_request
            def after_request(response):
                response.headers.add('Access-Control-Allow-Origin', '*')
                response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
                response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
                return response
        self.setup_routes()
        self.server_thread = None

    def setup_routes(self):
        """
        Setup REST API endpoints
        """
        
        @self.app.route('/api/v1/state', methods=['GET'])
        def get_state():
            """Get current network state"""
            try:
                state = self.controller.get_state()
                return jsonify({
                    'success': True,
                    'state': state
                }), 200
            except Exception as e:
                LOG.error(f"Error getting state: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/api/v1/stats', methods=['GET'])
        def get_stats():
            """Get flow statistics"""
            try:
                stats = self.controller.flow_stats
                return jsonify({
                    'success': True,
                    'stats': stats
                }), 200
            except Exception as e:
                LOG.error(f"Error getting stats: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/api/v1/qos/apply', methods=['POST'])
        def apply_qos():
            """Apply QoS action from RL agent"""
            try:
                data = request.get_json()
                dpid = data.get('dpid', 1)
                queue_id = data.get('queue_id', 0)
                min_rate = data.get('min_rate', 10000)
                max_rate = data.get('max_rate', 100000)
                priority = data.get('priority', 0)

                success = self.controller.apply_qos_action(
                    dpid, queue_id, min_rate, max_rate, priority
                )

                if success:
                    return jsonify({
                        'success': True,
                        'message': 'QoS action applied successfully'
                    }), 200
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Failed to apply QoS action'
                    }), 400
            except Exception as e:
                LOG.error(f"Error applying QoS: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/api/v1/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'controller': 'running'
            }), 200

    def start(self):
        """
        Start REST API server in a separate thread
        """
        def run_server():
            LOG.info(f"Starting REST API server on {self.host}:{self.port}")
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        LOG.info("REST API server started")

    def stop(self):
        """
        Stop REST API server
        """
        # Flask doesn't have a clean shutdown, but this will stop the thread
        LOG.info("REST API server stopped")


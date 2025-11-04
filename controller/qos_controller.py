"""
Ryu SDN Controller Application for Adaptive QoS Management
Collects flow statistics and applies QoS rules via OpenFlow
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp
from ryu.topology import event
from ryu.topology.api import get_switch, get_link
import logging
import threading
import time
import json

LOG = logging.getLogger(__name__)


class QoSController(app_manager.RyuApp):
    """
    Main Ryu application for QoS management
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(QoSController, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.flow_stats = {}  # {dpid: {port: stats}}
        self.qos_rules = {}  # {dpid: {queue_id: rule}}
        self.monitor_thread = None
        self.stats_interval = 2.0  # seconds
        self.state_data = {
            'link_utilization': {},
            'queue_length': {},
            'delay': {},
            'packet_loss': {}
        }
        
        LOG.info("QoS Controller initialized")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        Configure switch when it connects
        """
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install default flow entry (send to controller)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

        LOG.info(f"Switch {datapath.id} connected")

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        """
        Handle switch connection/disconnection
        """
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
                self.flow_stats[datapath.id] = {}
                LOG.info(f"Switch {datapath.id} registered")
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]
                if datapath.id in self.flow_stats:
                    del self.flow_stats[datapath.id]
                LOG.info(f"Switch {datapath.id} disconnected")

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """
        Install flow entry to switch
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                   priority=priority, match=match,
                                   instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                   match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """
        Handle packets sent to controller
        """
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Install flow rule for this packet
        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        
        self.add_flow(datapath, 1, match, actions, msg.buffer_id)

    def request_stats(self):
        """
        Request flow statistics from all switches
        """
        for dpid, datapath in self.datapaths.items():
            parser = datapath.ofproto_parser
            req = parser.OFPFlowStatsRequest(datapath)
            datapath.send_msg(req)

            # Request port stats
            req = parser.OFPPortStatsRequest(datapath, 0, ofproto_v1_3.OFPP_ANY)
            datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """
        Handle flow statistics reply
        """
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        stats = {}
        for stat in body:
            port = stat.match.get('in_port', 0)
            stats[port] = {
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count,
                'duration_sec': stat.duration_sec,
                'duration_nsec': stat.duration_nsec
            }

        self.flow_stats[dpid] = stats

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """
        Handle port statistics reply
        """
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        for stat in body:
            port_no = stat.port_no
            if port_no not in self.state_data['link_utilization']:
                self.state_data['link_utilization'][port_no] = []
            
            # Calculate link utilization (simplified)
            # In real scenario, compare with port capacity
            tx_bytes = stat.tx_bytes
            rx_bytes = stat.rx_bytes
            
            # Update state data
            utilization = min(1.0, (tx_bytes + rx_bytes) / (1024 * 1024 * 100))  # Normalized
            self.state_data['link_utilization'][port_no] = utilization

    def start_monitoring(self):
        """
        Start periodic statistics collection
        """
        def monitor_loop():
            while True:
                if self.datapaths:
                    self.request_stats()
                time.sleep(self.stats_interval)

        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        LOG.info("Statistics monitoring started")

    def apply_qos_action(self, dpid, queue_id, min_rate, max_rate, priority):
        """
        Apply QoS action (queue configuration) to switch
        
        Args:
            dpid: DataPath ID (switch identifier)
            queue_id: Queue identifier (0-7 typically)
            min_rate: Minimum rate in bits per second
            max_rate: Maximum rate in bits per second
            priority: Queue priority
        """
        # Check if switch is connected and registered
        # If not found, try to use any available switch as fallback
        if dpid not in self.datapaths:
            # If no switches available at all, return False
            if not self.datapaths:
                LOG.warning(f"Switch {dpid} not found - no switches connected yet")
                return False
            
            # Use first available switch as fallback
            # In demo mode, we'll use any connected switch
            available_dpid = list(self.datapaths.keys())[0]
            LOG.info(f"Switch {dpid} not found, using available switch {available_dpid}")
            dpid = available_dpid

        datapath = self.datapaths[dpid]
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        try:
            # Create queue configuration
            queue_prop = [parser.OFPQueuePropMinRate(min_rate)]
            queue = parser.OFPQueue(queue_id, queue_prop)

            # Configure queue
            req = parser.OFPQueueMod(datapath, queue_id, ofproto.OFPQCF_MIN_RATE, queue)
            datapath.send_msg(req)

            # Update QoS rules
            if dpid not in self.qos_rules:
                self.qos_rules[dpid] = {}
            self.qos_rules[dpid][queue_id] = {
                'min_rate': min_rate,
                'max_rate': max_rate,
                'priority': priority
            }

            LOG.info(f"Applied QoS rule: dpid={dpid}, queue={queue_id}, min_rate={min_rate}, max_rate={max_rate}")
            return True
        except Exception as e:
            LOG.error(f"Failed to apply QoS action: {e}")
            return False

    def get_state(self):
        """
        Get current network state for RL agent
        """
        # Aggregate state data from all switches
        state = {
            'link_utilization': [],
            'queue_length': [],
            'delay': [],
            'packet_loss': []
        }

        # Collect utilization data
        for port, util in self.state_data['link_utilization'].items():
            if isinstance(util, float):
                state['link_utilization'].append(util)
            elif isinstance(util, list) and len(util) > 0:
                state['link_utilization'].append(util[-1])

        # Default values for missing metrics (in real scenario, compute from stats)
        num_ports = len(state['link_utilization'])
        if num_ports == 0:
            num_ports = 1

        state['queue_length'] = [0.0] * num_ports  # Placeholder
        state['delay'] = [0.0] * num_ports  # Placeholder
        state['packet_loss'] = [0.0] * num_ports  # Placeholder

        return state




#!/usr/bin/env python3
"""
LawinTracker RC Controller — pymavlink UDP Version
===================================================
Reads RC channels from Pixhawk via MAVProxy UDP bridge.

Architecture:
  Pixhawk TELEM2 → MAVProxy → UDP:14550 → MAVROS
                             → UDP:14551 → this node

Run MAVProxy bridge first:
  mavproxy.py --master=/dev/ttyAMA0 --baudrate 921600 \
    --out udp:127.0.0.1:14550 \
    --out udp:127.0.0.1:14551 \
    --no-state

Channel Mapping:
  CH5 → ARM switch
    2000 = ARM
    999  = DISARM

  CH6 → Flight mode switch
    2000 = ALT_HOLD
    1503 = GUIDED (mid)
    999  = RTL

  CH8 → Track switch
    2000 = TRACK_START
    999  = TRACK_CANCEL

Run:
  ros2 run lawin_mavros rc_controller
"""

import rclpy
import threading
import time
from rclpy.node import Node
from std_msgs.msg import String

try:
    from pymavlink import mavutil
    PYMAVLINK_OK = True
except ImportError:
    PYMAVLINK_OK = False


# ── MAVLink UDP connection ─────────────────
# MAVProxy forwards to this port
MAVLINK_URL = 'udpin:127.0.0.1:14551'

# ── Channel indices (1-based from MAVLink) ─
CH_ARM    = 5   # chan5_raw  → ARM/DISARM
CH_FLIGHT = 6   # chan6_raw  → ALT_HOLD/GUIDED/RTL
CH_TRACK  = 8   # chan8_raw  → TRACK_START/CANCEL

# ── PWM thresholds ────────────────────────
HIGH_THRESHOLD = 1700   # > 1700 = HIGH (2000)
LOW_THRESHOLD  = 1200   # < 1200 = LOW  (999)
# MID = between 1200-1700 (1503 = GUIDED)


class RCController(Node):

    def __init__(self):
        super().__init__('rc_controller')

        # ── Dedup state ───────────────────
        self.last_flight_cmd   = None
        self.last_arm_cmd      = None
        self.last_tracking_cmd = None

        # ── Startup lock ──────────────────
        # System is LOCKED on boot. Pilot must explicitly
        # flip CH6 to ALT_HOLD after power-up to unlock.
        # Even if CH6 is already in ALT_HOLD at startup,
        # it does not count — requires a deliberate flip.
        self.stabilize_activated = False

        # ── Publisher ─────────────────────
        self.cmd_pub = self.create_publisher(
            String, '/lawin/rc_command', 10)

        # ── MAVLink connection ────────────
        self.mav = None
        if not PYMAVLINK_OK:
            self.get_logger().error(
                "pymavlink not installed! "
                "Run: pip3 install pymavlink")
            return

        self.get_logger().info(
            "RC Controller (pymavlink UDP) ready!")
        self.get_logger().info(
            f"Connecting to MAVProxy on {MAVLINK_URL}...")
        self.get_logger().info(
            "Make sure MAVProxy is running:")
        self.get_logger().info(
            "  mavproxy.py --master=/dev/ttyAMA0 "
            "--baudrate 921600 "
            "--out udp:127.0.0.1:14550 "
            "--out udp:127.0.0.1:14551 "
            "--no-state")

        # ── Start RC read thread ──────────
        self.rc_thread = threading.Thread(
            target=self.rc_read_loop, daemon=True)
        self.rc_thread.start()

    # ─────────────────────────────────────
    # RC Read Loop (background thread)
    # ─────────────────────────────────────
    def rc_read_loop(self):
        """Connect to MAVProxy UDP and read RC channels."""
        while True:
            try:
                self.get_logger().info(
                    f"Connecting to {MAVLINK_URL}...")
                self.mav = mavutil.mavlink_connection(
                    MAVLINK_URL)
                self.mav.wait_heartbeat(timeout=30)
                self.get_logger().info(
                    f"MAVLink connected! "
                    f"System: {self.mav.target_system}")
                self.get_logger().info(
                    "CH5=ARM | CH6=FLIGHT | CH8=TRACK")

                # ── Read initial switch state ──────────────
                # On startup, dedup would suppress publishing
                # if switches haven't moved. Force-read the
                # current RC state so test.py knows the mode
                # immediately without needing a switch flip.
                self.get_logger().info(
                    "Reading initial RC switch state...")
                self._read_initial_state()

                # Main RC read loop
                while True:
                    msg = self.mav.recv_match(
                        type='RC_CHANNELS',
                        blocking=True,
                        timeout=2.0)

                    if msg is None:
                        self.get_logger().warn(
                            "No RC data for 2s...",
                            throttle_duration_sec=5.0)
                        continue

                    self._process_rc(msg)

            except Exception as e:
                self.get_logger().error(
                    f"MAVLink error: {e}")
                self.get_logger().info(
                    "Reconnecting in 3s...")
                time.sleep(3)

    # ─────────────────────────────────────
    # Read Initial RC State on Connect
    # ─────────────────────────────────────
    def _read_initial_state(self):
        """
        Read current RC positions silently on startup.
        Sets dedup state so we detect CHANGES from here.
        Does NOT publish ALT_HOLD or activate the system —
        the pilot must explicitly flip CH6 to ALT_HOLD
        after power-up to unlock the system.
        """
        try:
            msg = self.mav.recv_match(
                type='RC_CHANNELS',
                blocking=True,
                timeout=3.0)

            if msg is None:
                self.get_logger().warn(
                    "No RC_CHANNELS in 3s — "
                    "waiting for pilot input...")
                return

            arm_val    = getattr(msg, f'chan{CH_ARM}_raw',    0)
            flight_val = getattr(msg, f'chan{CH_FLIGHT}_raw', 0)
            track_val  = getattr(msg, f'chan{CH_TRACK}_raw',  0)

            # Silently record current positions as dedup baseline
            # so we only publish on CHANGES from here forward.
            # We do NOT publish anything here — system stays locked.
            if arm_val > HIGH_THRESHOLD:
                self.last_arm_cmd = 'ARM'
            elif arm_val < LOW_THRESHOLD:
                self.last_arm_cmd = 'DISARM'

            if flight_val > HIGH_THRESHOLD:
                self.last_flight_cmd = 'ALT_HOLD'
            elif flight_val < LOW_THRESHOLD:
                self.last_flight_cmd = 'RTL'
            else:
                self.last_flight_cmd = 'GUIDED'

            if track_val > HIGH_THRESHOLD:
                self.last_tracking_cmd = 'TRACK_START'
            elif track_val < LOW_THRESHOLD:
                self.last_tracking_cmd = 'TRACK_CANCEL'

            self.get_logger().info(
                f"Startup RC state read — "
                f"ARM:{arm_val} FLIGHT:{flight_val} "
                f"TRACK:{track_val}")
            self.get_logger().info(
                "SYSTEM LOCKED — flip CH6 to ALT_HOLD "
                "to unlock and enable arming")

            # Publish locked status so test.py shows it
            locked_msg = String()
            locked_msg.data = 'SYSTEM_LOCKED'
            self.cmd_pub.publish(locked_msg)

        except Exception as e:
            self.get_logger().warn(
                f"Could not read initial RC state: {e}")

    # ─────────────────────────────────────
    # Process RC Message
    # ─────────────────────────────────────
    def _process_rc(self, msg):
        """Process RC_CHANNELS message and publish commands."""

        arm_val    = getattr(msg, f'chan{CH_ARM}_raw',    0)
        flight_val = getattr(msg, f'chan{CH_FLIGHT}_raw', 0)
        track_val  = getattr(msg, f'chan{CH_TRACK}_raw',  0)

        # ── Flight mode (CH6) — always process first ───────
        # Detect explicit ALT_HOLD flip to unlock system
        if flight_val > HIGH_THRESHOLD:
            if self.last_flight_cmd != 'ALT_HOLD':
                # This is a real flip TO ALT_HOLD
                self.stabilize_activated = True
                self.get_logger().info(
                    "ALT_HOLD flipped — system UNLOCKED!")
            self.publish_command(
                'ALT_HOLD', 'last_flight_cmd',
                f"ALT_HOLD! (CH6={flight_val})")
        elif flight_val < LOW_THRESHOLD:
            self.publish_command(
                'RTL', 'last_flight_cmd',
                f"RTL! (CH6={flight_val})")
        else:
            self.publish_command(
                'GUIDED', 'last_flight_cmd',
                f"GUIDED! (CH6={flight_val})")

        # ── ARM switch (CH5) — blocked until ALT_HOLD flipped
        if arm_val > HIGH_THRESHOLD:
            if not self.stabilize_activated:
                # System still locked — warn pilot
                if self.last_arm_cmd != 'ARM_LOCKED':
                    self.last_arm_cmd = 'ARM_LOCKED'
                    self.get_logger().warn(
                        "ARM ignored — flip CH6 to "
                        "ALT_HOLD first to unlock system!")
                    locked_msg = String()
                    locked_msg.data = 'SYSTEM_LOCKED'
                    self.cmd_pub.publish(locked_msg)
            else:
                self.publish_command(
                    'ARM', 'last_arm_cmd',
                    f"ARM! (CH5={arm_val})")
        elif arm_val < LOW_THRESHOLD:
            # DISARM always works regardless of lock state
            if self.last_arm_cmd != 'DISARM':
                self.last_arm_cmd = 'DISARM'
                msg_out = String()
                msg_out.data = 'DISARM'
                self.cmd_pub.publish(msg_out)
                self.get_logger().info(
                    f"DISARM! (CH5={arm_val})")

        # ── Track switch (CH8) — blocked until unlocked ────
        if track_val > HIGH_THRESHOLD:
            if self.stabilize_activated:
                self.publish_command(
                    'TRACK_START', 'last_tracking_cmd',
                    f"TRACK START! (CH8={track_val})")
        elif track_val < LOW_THRESHOLD:
            if self.stabilize_activated:
                self.publish_command(
                    'TRACK_CANCEL', 'last_tracking_cmd',
                    f"TRACK CANCEL! (CH8={track_val})")

    # ─────────────────────────────────────
    # Publish command (with dedup)
    # ─────────────────────────────────────
    def publish_command(self, cmd: str,
                        last_attr: str,
                        log_msg: str,
                        force: bool = False):
        """Publish if command changed, or always if force=True."""
        if force or getattr(self, last_attr) != cmd:
            setattr(self, last_attr, cmd)
            msg = String()
            msg.data = cmd
            self.cmd_pub.publish(msg)
            self.get_logger().info(log_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RCController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            "RC Controller shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
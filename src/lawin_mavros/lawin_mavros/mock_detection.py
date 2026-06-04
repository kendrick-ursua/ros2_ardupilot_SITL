#!/usr/bin/env python3
"""
LawinTracker Mock Detection Node  (mock_detection.py)
======================================================
Publishes fake /lawin/target_offset data to test drone
tracking movement without a camera or Hailo inference.

Also sends TRACK_START on /lawin/rc_command so the flight
controller enters FOLLOW mode automatically.

Test cases:
  1 → person RIGHT  of centre  (drone strafes/yaws RIGHT)
  2 → person LEFT   of centre  (drone strafes/yaws LEFT)
  3 → person ABOVE  centre     (drone moves FORWARD)
  4 → person BELOW  centre     (drone moves BACKWARD)
  0 → STOP — publish zero offset (drone holds)
  q → quit

Point message convention (matches oak1_tracker.py):
  x  = horizontal pixel error  (+ = right, - = left)
  y  = vertical   pixel error  (+ = down,  - = up  )
  z  = 1.0 if target present, 0.0 if lost

Offset magnitudes are set well outside STOP_ZONE (15px) and
START_ZONE (35px) so the controller actually moves.

Run:
  ros2 run lawin_mavros mock_detection
"""

import rclpy
import threading
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String


# ── How far off-centre to pretend the person is (pixels) ─────────
OFFSET_PX = 120   # must be > START_ZONE (35) in flight controller

# ── Test case definitions ─────────────────────────────────────────
TEST_CASES = {
    '1': ('RIGHT',    OFFSET_PX,   0),
    '2': ('LEFT',    -OFFSET_PX,   0),
    '3': ('FORWARD',  0,          -OFFSET_PX),   # person above → drone fwd
    '4': ('BACKWARD', 0,           OFFSET_PX),   # person below → drone back
    '0': ('STOP',     0,           0),
}


class MockDetectionNode(Node):

    def __init__(self):
        super().__init__('mock_detection')

        self.offset_pub = self.create_publisher(
            Point, '/lawin/target_offset', 10)
        self.rc_pub = self.create_publisher(
            String, '/lawin/rc_command', 10)

        # Publish at 10 Hz so the flight controller sees a steady stream
        self._active_x  = 0.0
        self._active_y  = 0.0
        self._active_z  = 0.0   # 0 = no target
        self._lock = threading.Lock()

        self.create_timer(0.1, self._publish_offset)

        self.get_logger().info("=" * 52)
        self.get_logger().info("  LawinTracker MOCK DETECTION NODE")
        self.get_logger().info("=" * 52)
        self.get_logger().info(
            f"  Offset magnitude : ±{OFFSET_PX}px")
        self.get_logger().info(
            "  Publishing to    : /lawin/target_offset")
        self.get_logger().info(
            "  Publishing to    : /lawin/rc_command")
        self.get_logger().info("-" * 52)
        self.get_logger().info(
            "  1 → person RIGHT   → drone goes RIGHT")
        self.get_logger().info(
            "  2 → person LEFT    → drone goes LEFT")
        self.get_logger().info(
            "  3 → person FORWARD → drone goes FORWARD")
        self.get_logger().info(
            "  4 → person BACK    → drone goes BACKWARD")
        self.get_logger().info(
            "  0 → STOP (zero offset, target present)")
        self.get_logger().info(
            "  q → QUIT node")
        self.get_logger().info("-" * 52)
        self.get_logger().info(
            "  Sending TRACK_START now...")
        self.get_logger().info("=" * 52)

        # Send TRACK_START immediately so flight controller enters FOLLOW
        self._send_track_start()

        # Start keyboard input on a background thread so ROS spin
        # is never blocked
        t = threading.Thread(
            target=self._input_loop,
            daemon=True,
            name="mock_input")
        t.start()

    # ──────────────────────────────────────────────────────
    # TRACK_START
    # ──────────────────────────────────────────────────────

    def _send_track_start(self):
        """Send TRACK_START a few times to make sure it lands."""
        def _fire():
            msg = String()
            msg.data = 'TRACK_START'
            for _ in range(5):
                self.rc_pub.publish(msg)
        self.create_timer(0.5, _fire)

    # ──────────────────────────────────────────────────────
    # Keyboard input loop  (background thread)
    # ──────────────────────────────────────────────────────

    def _input_loop(self):
        while rclpy.ok():
            try:
                key = input(
                    "\nTest [1=RIGHT 2=LEFT 3=FWD 4=BACK "
                    "0=STOP q=QUIT]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if key == 'q':
                self.get_logger().info("Quitting mock node.")
                rclpy.shutdown()
                break

            if key not in TEST_CASES:
                self.get_logger().warn(
                    f"Unknown key '{key}' — "
                    "use 1/2/3/4/0/q")
                continue

            name, ox, oy = TEST_CASES[key]

            with self._lock:
                self._active_x = float(ox)
                self._active_y = float(oy)
                self._active_z = 0.0 if (ox == 0 and oy == 0) else 1.0

            if key == '0':
                self.get_logger().info(
                    "STOP — publishing zero offset "
                    "(target still present, drone holds)")
            else:
                h = 'RIGHT' if ox > 0 else ('LEFT' if ox < 0 else '-')
                v = ('DOWN/BACK' if oy > 0
                     else ('UP/FWD' if oy < 0 else '-'))
                self.get_logger().info(
                    f"Test {key}: {name} | "
                    f"offset=({ox:+d},{oy:+d})px | "
                    f"H:{h} V:{v}")

    # ──────────────────────────────────────────────────────
    # Timer — publishes at 10 Hz
    # ──────────────────────────────────────────────────────

    def _publish_offset(self):
        with self._lock:
            x = self._active_x
            y = self._active_y
            z = self._active_z

        msg = Point()
        msg.x = x
        msg.y = y
        msg.z = z
        self.offset_pub.publish(msg)


# ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MockDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down mock detection node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
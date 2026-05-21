#!/usr/bin/env python3
"""Simple example: arm the drone and take off to a target altitude via MAVROS.

Usage (inside the container):
    ros2 run mavros_scripts arm_and_takeoff  # after installing the package, or
    python3 mavros/scripts/arm_and_takeoff.py
"""

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped


TARGET_ALTITUDE = 3.0  # metres


class ArmAndTakeoff(Node):
    def __init__(self):
        super().__init__('arm_and_takeoff')
        self.current_state = State()

        # Subscribers
        self.state_sub = self.create_subscription(
            State, '/mavros/state', self._state_cb, 10)

        # Publishers
        self.local_pos_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10)

        # Service clients
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # Wait for FCU connection
        self.get_logger().info('Waiting for FCU connection…')
        while rclpy.ok() and not self.current_state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('FCU connected!')

        self._run()

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _state_cb(self, msg: State):
        self.current_state = msg

    # ── main logic ────────────────────────────────────────────────────────────
    def _run(self):
        # Stream setpoints before switching mode (OFFBOARD requirement)
        pose = PoseStamped()
        pose.pose.position.z = TARGET_ALTITUDE
        for _ in range(100):
            self.local_pos_pub.publish(pose)
            rclpy.spin_once(self, timeout_sec=0.05)

        # Switch to GUIDED (ArduPilot) or OFFBOARD (PX4)
        self._set_mode('GUIDED')
        self._arm(True)

        self.get_logger().info(f'Climbing to {TARGET_ALTITUDE} m …')
        while rclpy.ok():
            self.local_pos_pub.publish(pose)
            rclpy.spin_once(self, timeout_sec=0.1)

    def _set_mode(self, mode: str):
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.set_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().mode_sent:
            self.get_logger().info(f'Mode set to {mode}')
        else:
            self.get_logger().error(f'Failed to set mode {mode}')

    def _arm(self, arm: bool):
        req = CommandBool.Request()
        req.value = arm
        future = self.arming_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().success:
            self.get_logger().info('Armed!' if arm else 'Disarmed!')
        else:
            self.get_logger().error('Arming failed!')


def main():
    rclpy.init()
    node = ArmAndTakeoff()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

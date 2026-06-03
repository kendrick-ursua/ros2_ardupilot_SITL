#!/usr/bin/env python3
"""
Zigzag Mission Node — MAVROS / ROS 2 Humble
=============================================
Converted from ArduPilot DDS → MAVROS topics & services.

Reference:
  https://ardupilot.org/dev/docs/ros-install.html#installing-mavros
  https://github.com/mavlink/mavros/tree/master/mavros

MAVROS Topics / Services Used
------------------------------
  SUB  /mavros/global_position/global      sensor_msgs/NavSatFix
         (replaces /ap/navsat)

  PUB  /mavros/setpoint_raw/global         mavros_msgs/GlobalPositionTarget
         (replaces /ap/cmd_gps_pose + ardupilot_msgs/GlobalPosition)

  SRV  /mavros/cmd/arming                  mavros_msgs/CommandBool
         (replaces /ap/arm_motors)

  SRV  /mavros/set_mode                    mavros_msgs/SetMode
         (replaces /ap/mode_switch)

  SRV  /mavros/cmd/takeoff                 mavros_msgs/CommandTOL
         (replaces /ap/experimental/takeoff)

GlobalPositionTarget type_mask bits
-------------------------------------
  Ignore velocity, acceleration, yaw rate — position only:
    type_mask = 0b111111111000  = 0xFF8  = 4088
  That leaves bits 0-2 (Lat/Lon/Alt) ACTIVE.

coordinate_frame values (MAV_FRAME_*)
--------------------------------------
  6  = GLOBAL_RELATIVE_ALT  — altitude relative to home/takeoff ← used here

Run
----
  ros2 launch mavros apm.launch fcu_url:=udp://localhost:14551@14555
  ros2 run lawin_tracker zigzag_mission_mavros
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from std_msgs.msg import Header
from sensor_msgs.msg import NavSatFix, NavSatStatus

# ── MAVROS message / service types ────────────────────────────────────────────
from mavros_msgs.msg import GlobalPositionTarget   # /mavros/setpoint_raw/global
from mavros_msgs.srv import CommandBool            # /mavros/cmd/arming
from mavros_msgs.srv import SetMode                # /mavros/set_mode
from mavros_msgs.srv import CommandTOL             # /mavros/cmd/takeoff
# ──────────────────────────────────────────────────────────────────────────────


############################################################
# MISSION PARAMETERS
############################################################

ALTITUDE     = 3.0   # metres (relative to home/takeoff)
STEP_FORWARD = 2.0   # metres North per leg
STEP_SIDE    = 2.0   # metres East/West per leg
THRESHOLD    = 0.30  # metres — "close enough" radius
TIMEOUT      = 30    # seconds before skipping a waypoint

# MAV_FRAME_GLOBAL_RELATIVE_ALT — altitude relative to home position
MAV_FRAME_GLOBAL_RELATIVE_ALT = 6

# GlobalPositionTarget type_mask:
#   Ignore velocity, acceleration, yaw rate → position control only.
#   Bits 0-2 (lat/lon/alt) = 0 → ACTIVE
#   Bits 3-11              = 1 → IGNORED
TYPE_MASK_POS_ONLY = (
    GlobalPositionTarget.IGNORE_VX
    | GlobalPositionTarget.IGNORE_VY
    | GlobalPositionTarget.IGNORE_VZ
    | GlobalPositionTarget.IGNORE_AFX
    | GlobalPositionTarget.IGNORE_AFY
    | GlobalPositionTarget.IGNORE_AFZ
    | GlobalPositionTarget.IGNORE_YAW
    | GlobalPositionTarget.IGNORE_YAW_RATE
)

# ArduCopter flight mode strings for /mavros/set_mode
MODE_GUIDED = "GUIDED"
MODE_RTL    = "RTL"


############################################################
# NODE
############################################################

class ZigzagMissionNode(Node):

    def __init__(self):
        super().__init__('zigzag_mission')

        # ── QoS — BEST_EFFORT matches MAVROS sensor publishers ──────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── State ─────────────────────────────────────────────────────────────
        self._position: tuple | None = None   # (lat_deg, lon_deg, alt_amsl_m)

        # ── Subscriber — GPS position ─────────────────────────────────────────
        # /mavros/global_position/global  (sensor_msgs/NavSatFix)
        #   replaces /ap/navsat
        #
        # NavSatFix fields:
        #   status.status : int8   (STATUS_NO_FIX=-1, STATUS_FIX=0, STATUS_SBAS_FIX=1, …)
        #   latitude      : float64
        #   longitude     : float64
        #   altitude      : float64  (AMSL, metres)
        self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self._navsat_cb,
            sensor_qos,
        )

        # ── Publisher — GPS setpoint ──────────────────────────────────────────
        # /mavros/setpoint_raw/global  (mavros_msgs/GlobalPositionTarget)
        #   replaces /ap/cmd_gps_pose (ardupilot_msgs/GlobalPosition)
        #
        # GlobalPositionTarget fields:
        #   header           : std_msgs/Header
        #   coordinate_frame : uint8   (MAV_FRAME_*)
        #   type_mask        : uint16  (bitmask — 0=active, 1=ignore)
        #   latitude         : float64
        #   longitude        : float64
        #   altitude         : float32
        #   velocity         : geometry_msgs/Vector3  (ignored)
        #   acceleration_or_force : geometry_msgs/Vector3 (ignored)
        #   yaw              : float32  (ignored)
        #   yaw_rate         : float32  (ignored)
        self._cmd_pub = self.create_publisher(
            GlobalPositionTarget,
            '/mavros/setpoint_raw/global',
            sensor_qos,
        )

        # ── Service clients ───────────────────────────────────────────────────

        # /mavros/cmd/arming  (mavros_msgs/CommandBool)
        #   replaces /ap/arm_motors (ardupilot_msgs/ArmMotors)
        #   Request:  value (bool)  — True=arm, False=disarm
        #   Response: success (bool), result (uint8)
        self._arm_cli = self.create_client(
            CommandBool, '/mavros/cmd/arming')

        # /mavros/set_mode  (mavros_msgs/SetMode)
        #   replaces /ap/mode_switch (ardupilot_msgs/ModeSwitch)
        #   Request:  base_mode (uint8)=0, custom_mode (string) e.g. "GUIDED"
        #   Response: mode_sent (bool)
        self._mode_cli = self.create_client(
            SetMode, '/mavros/set_mode')

        # /mavros/cmd/takeoff  (mavros_msgs/CommandTOL)
        #   replaces /ap/experimental/takeoff (ardupilot_msgs/Takeoff)
        #   Request:  min_pitch, yaw, latitude, longitude, altitude (float32)
        #   Response: success (bool), result (uint8)
        self._takeoff_cli = self.create_client(
            CommandTOL, '/mavros/cmd/takeoff')

        self.get_logger().info(
            'ZigzagMissionNode (MAVROS) initialised — waiting for topics…')

    # ──────────────────────────────────────────────────────────────────────────
    # Subscriber Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _navsat_cb(self, msg: NavSatFix) -> None:
        """
        /mavros/global_position/global callback.
        Only store position when we have a valid fix.
        """
        if msg.status.status == NavSatStatus.STATUS_NO_FIX:
            return
        self._position = (msg.latitude, msg.longitude, msg.altitude)

    # ──────────────────────────────────────────────────────────────────────────
    # Spin Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _spin_for(self, seconds: float) -> None:
        """Non-blocking spin for *seconds*."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _wait_for_position(self, timeout: float = 30.0) -> bool:
        self.get_logger().info('Waiting for GPS fix on /mavros/global_position/global…')
        deadline = time.time() + timeout
        while self._position is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._position is None:
            self.get_logger().error('GPS fix timed out!')
            return False
        self.get_logger().info(
            f'GPS fix: lat={self._position[0]:.6f}  '
            f'lon={self._position[1]:.6f}  '
            f'alt={self._position[2]:.1f} m AMSL'
        )
        return True

    def _get_position(self) -> tuple | None:
        rclpy.spin_once(self, timeout_sec=0.2)
        return self._position

    # ──────────────────────────────────────────────────────────────────────────
    # Geometry Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2) -> float:
        """Flat-Earth approximation (accurate for short distances <1 km)."""
        dlat = (lat2 - lat1) * 111_111.0
        dlon = (lon2 - lon1) * 111_111.0 * math.cos(math.radians(lat1))
        return math.hypot(dlat, dlon)

    # ──────────────────────────────────────────────────────────────────────────
    # Command Builder
    # ──────────────────────────────────────────────────────────────────────────

    def _make_gps_cmd(
        self, lat: float, lon: float, rel_alt: float
    ) -> GlobalPositionTarget:
        """
        Build a mavros_msgs/GlobalPositionTarget setpoint.

        coordinate_frame = MAV_FRAME_GLOBAL_RELATIVE_ALT (6)
            → altitude is relative to home/takeoff position.
        type_mask = TYPE_MASK_POS_ONLY
            → only lat/lon/alt are active; velocity and yaw are ignored.
        """
        msg                  = GlobalPositionTarget()
        msg.header           = Header()
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.header.frame_id  = 'map'
        msg.coordinate_frame = MAV_FRAME_GLOBAL_RELATIVE_ALT
        msg.type_mask        = TYPE_MASK_POS_ONLY
        msg.latitude         = lat
        msg.longitude        = lon
        msg.altitude         = float(rel_alt)
        # velocity / acceleration / yaw are zero-initialised — ignored via mask
        return msg

    # ──────────────────────────────────────────────────────────────────────────
    # MAVROS Primitives
    # ──────────────────────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> bool:
        """
        Set ArduPilot flight mode via /mavros/set_mode.
        mavros_msgs/SetMode:
          Request:  base_mode=0, custom_mode="GUIDED" / "RTL" / etc.
          Response: mode_sent (bool)
        """
        self.get_logger().info(f'Setting mode → {mode}')
        while not self._mode_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/mavros/set_mode not ready, retrying…')

        req             = SetMode.Request()
        req.base_mode   = 0       # 0 = use custom_mode string
        req.custom_mode = mode    # e.g. "GUIDED", "RTL"

        future = self._mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() and future.result().mode_sent:
            self.get_logger().info(f'Mode set: {mode} ✓')
            return True

        self.get_logger().error(f'Mode switch FAILED for {mode}')
        return False

    def arm(self) -> bool:
        """
        Arm motors via /mavros/cmd/arming.
        mavros_msgs/CommandBool:
          Request:  value=True (arm) / False (disarm)
          Response: success (bool), result (uint8)
        """
        self.get_logger().info('Arming motors via /mavros/cmd/arming…')
        while not self._arm_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/mavros/cmd/arming not ready, retrying…')

        req       = CommandBool.Request()
        req.value = True   # True = ARM

        future = self._arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() and future.result().success:
            self.get_logger().info('Motors ARMED ✓')
            return True

        self.get_logger().error(
            f'Arming FAILED '
            f'(result={future.result().result if future.result() else "timeout"}) '
            '— check pre-arm conditions in ArduPilot')
        return False

    def takeoff(self, alt: float) -> bool:
        """
        Takeoff via /mavros/cmd/takeoff.
        mavros_msgs/CommandTOL:
          Request:
            min_pitch : float32  (0.0 for copter)
            yaw       : float32  (desired heading, 0.0 = current)
            latitude  : float32  (0.0 = current position)
            longitude : float32  (0.0 = current position)
            altitude  : float32  (RELATIVE altitude in metres)
          Response: success (bool), result (uint8)
        """
        self.get_logger().info(f'Taking off to {alt} m (relative) via /mavros/cmd/takeoff…')
        while not self._takeoff_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/mavros/cmd/takeoff not ready, retrying…')

        req           = CommandTOL.Request()
        req.min_pitch = 0.0    # copter ignores this
        req.yaw       = 0.0    # hold current heading
        req.latitude  = 0.0    # 0.0 = takeoff at current position
        req.longitude = 0.0    # 0.0 = takeoff at current position
        req.altitude  = float(alt)   # relative altitude in metres

        future = self._takeoff_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() and future.result().success:
            self.get_logger().info(f'Takeoff accepted — climbing to {alt} m…')
            self._spin_for(8.0)   # wait for copter to reach altitude
            return True

        self.get_logger().error(
            f'Takeoff FAILED '
            f'(result={future.result().result if future.result() else "timeout"})')
        return False

    def goto(self, lat: float, lon: float, label: str = 'Waypoint') -> None:
        """
        Fly to GPS coordinate by continuously publishing to
        /mavros/setpoint_raw/global until within THRESHOLD metres.
        """
        self.get_logger().info(
            f'\nFlying to {label}  →  {lat:.6f}, {lon:.6f}')
        cmd        = self._make_gps_cmd(lat, lon, ALTITUDE)
        start_time = time.time()

        while True:
            cmd.header.stamp = self.get_clock().now().to_msg()
            self._cmd_pub.publish(cmd)
            self._spin_for(0.5)

            pos = self._get_position()
            if pos:
                dist = self._haversine_m(pos[0], pos[1], lat, lon)
                self.get_logger().info(
                    f'  ↳ Distance to {label}: {dist:.2f} m')
                if dist < THRESHOLD:
                    self.get_logger().info(f'✓ Reached {label}')
                    return

            if time.time() - start_time > TIMEOUT:
                self.get_logger().warn(
                    f'Timeout reaching {label} — skipping')
                return

    # ──────────────────────────────────────────────────────────────────────────
    # Waypoint Factory
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def create_zigzag(lat: float, lon: float) -> list[tuple]:
        m_lat = 1.0 / 111_111.0
        m_lon = 1.0 / (111_111.0 * math.cos(math.radians(lat)))
        return [
            ('A', lat + 1 * STEP_FORWARD * m_lat, lon + STEP_SIDE * m_lon),
            ('B', lat + 2 * STEP_FORWARD * m_lat, lon - STEP_SIDE * m_lon),
            ('C', lat + 3 * STEP_FORWARD * m_lat, lon + STEP_SIDE * m_lon),
            ('D', lat + 4 * STEP_FORWARD * m_lat, lon - STEP_SIDE * m_lon),
            ('E', lat + 5 * STEP_FORWARD * m_lat, lon + STEP_SIDE * m_lon),
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Mission Sequence
    # ──────────────────────────────────────────────────────────────────────────

    def run_mission(self) -> None:
        log = self.get_logger()

        log.info('==========================================')
        log.info('      ZIGZAG MISSION  —  MAVROS           ')
        log.info('==========================================')
        log.info(f'  Altitude : {ALTITUDE} m (relative to home)')
        log.info(f'  Forward  : {STEP_FORWARD} m / leg')
        log.info(f'  Sideways : {STEP_SIDE} m (±East)')
        log.info(f'  Threshold: {THRESHOLD} m')
        log.info('')
        log.info('    A         C         E   ← RTL')
        log.info('    |  \\   /  |  \\   /  |')
        log.info('  Home ──→ B         D      ')
        log.info('==========================================')

        # 1 — GPS fix
        if not self._wait_for_position():
            return

        # 2 — Set GUIDED mode via /mavros/set_mode
        if not self.set_mode(MODE_GUIDED):
            return
        self._spin_for(1.0)

        # 3 — Arm motors via /mavros/cmd/arming
        if not self.arm():
            return
        self._spin_for(1.0)

        # 4 — Takeoff via /mavros/cmd/takeoff
        if not self.takeoff(ALTITUDE):
            return

        # 5 — Snapshot home position AFTER climb
        home = self._get_position()
        if home is None:
            log.error('Lost GPS after takeoff — aborting')
            return
        log.info(f'Home: lat={home[0]:.6f}  lon={home[1]:.6f}')

        # 6 — Build zigzag waypoints
        waypoints = self.create_zigzag(home[0], home[1])
        log.info('--- Waypoint Coordinates ---')
        for name, wlat, wlon in waypoints:
            log.info(f'  {name}: lat={wlat:.6f}  lon={wlon:.6f}')

        # 7 — Fly zigzag via /mavros/setpoint_raw/global
        log.info('--- Starting Zigzag Flight ---')
        for name, wlat, wlon in waypoints:
            self.goto(wlat, wlon, f'Waypoint {name}')
            self._spin_for(1.0)

        # 8 — RTL via /mavros/set_mode
        log.info('==========================================')
        log.info('          RTL FROM WAYPOINT E             ')
        log.info('==========================================')
        self.set_mode(MODE_RTL)
        log.info('==========================================')
        log.info('            MISSION COMPLETE              ')
        log.info('==========================================')


############################################################
# ENTRY POINT
############################################################

def main(args=None):
    rclpy.init(args=args)
    node = ZigzagMissionNode()
    try:
        node.run_mission()
    except KeyboardInterrupt:
        node.get_logger().info('Mission interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
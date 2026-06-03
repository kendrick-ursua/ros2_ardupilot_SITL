#!/usr/bin/env python3
"""
LawinTracker Original Hover Test Node (orig_hover_test.py)
=============================================
Standalone hover + tracking test node.
DO NOT run alongside flight_controller.py or test.py.

Purpose:
  Test person tracking without a waypoint mission.
  Drone takes off, hovers in POSHOLD, and tracks on CH8.

Workflow:
  ALT_HOLD mode (CH6 HIGH):
    → Pixhawk set to ALT_HOLD
    → Pilot arms with CH5 UP (manual flight)
    → CH5 DOWN = disarm

  GUIDED mode (CH6 MID):
    → 14 pre-arm checks
    → If cleared: GUIDED → arm → takeoff to HOVER_ALT (5m)
    → STAGE_POSHOLD — drone holds position indefinitely
    → CH8 UP   → STAGE_FOLLOW (track person centroid)
    → CH8 DOWN → STAGE_POSHOLD (back to position hold)
    → CH6 LOW  → RTL emergency

  RTL (CH6 LOW):
    → Emergency RTL from any stage

Startup lock:
  → System boots LOCKED
  → Pilot must explicitly flip CH6 to ALT_HOLD to unlock

RC Mapping:
  CH5 UP   = ARM    (ALT_HOLD only, after unlock)
  CH5 DOWN = DISARM (always)
  CH6 HIGH = ALT_HOLD
  CH6 MID  = GUIDED → hover at HOVER_ALT
  CH6 LOW  = RTL emergency
  CH8 UP   = TRACK_START (GUIDED only, in air)
  CH8 DOWN = TRACK_CANCEL → back to POSHOLD

Run:
  ros2 run lawin_mavros orig_hover_test
"""

import math
import rclpy
import time
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy,
    DurabilityPolicy, HistoryPolicy
)
from sensor_msgs.msg import NavSatFix, NavSatStatus, BatteryState
from std_msgs.msg import Float64, String, Header
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from mavros_msgs.msg import (
    State, ExtendedState, HomePosition,
    GPSRAW, GlobalPositionTarget
)
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL


# ── Hover config ────────────────────────────────────────────────
HOVER_ALT    = 5.0    # m — target hover altitude

# ── Battery ─────────────────────────────────────────────────────
BATTERY_MIN_VOLTAGE = 10.5   # 3S=10.5V | 4S=14.0V

# ── GPS quality ─────────────────────────────────────────────────
GPS_MIN_SATELLITES = 10
GPS_MAX_HDOP       = 1.5
EKF_POSE_MAX_AGE   = 2.0

# ── Landed state names ───────────────────────────────────────────
LANDED_NAMES = {
    0: "UNDEFINED", 1: "ON_GROUND",
    2: "IN_AIR",    3: "TAKEOFF", 4: "LANDING"
}

MAV_STATE_NAMES = {
    0: "UNINIT", 1: "BOOT", 2: "CALIBRATING",
    3: "STANDBY", 4: "ACTIVE", 5: "CRITICAL",
    6: "EMERGENCY", 7: "POWEROFF"
}

# ── Stages ──────────────────────────────────────────────────────
STAGE_LOCKED     = 0
STAGE_IDLE       = 1
STAGE_ALT_HOLD   = 2
STAGE_PREARM     = 3
STAGE_SET_GUIDED = 4
STAGE_ARM        = 5
STAGE_TAKEOFF    = 6
STAGE_CLIMB      = 7
STAGE_POSHOLD    = 8   # hovering, position hold
STAGE_FOLLOW     = 9   # tracking person centroid
STAGE_RTL        = 10

STAGE_NAMES = {
    0: "LOCKED",     1: "IDLE",       2: "ALT_HOLD",
    3: "PREARM",     4: "SET_GUIDED", 5: "ARM",
    6: "TAKEOFF",    7: "CLIMB",      8: "POSHOLD",
    9: "FOLLOW",     10: "RTL"
}

# ── Tracking controller config ───────────────────────────────────
KP_YAW       = 0.006   # doubled from 0.003 — faster yaw
KP_FWD       = 0.020   # doubled from 0.010 — faster forward
MAX_YAW      = 1.2     # increased from 0.8
MAX_VEL_X    = 3.0     # capped lower for safety
MIN_VEL      = 0.01
SMOOTH       = 0.4     # reduced from 0.7 — faster response
STOP_ZONE    = 20      # px dead zone
START_ZONE   = 40      # px exit dead zone
LOST_TIMEOUT = 2.0

PASS    = "PASS"
FAIL    = "FAIL"
WARN    = "WARN"
UNKNOWN = "UNKN"


class OrigHoverTestNode(Node):

    def __init__(self):
        super().__init__('orig_hover_test')

        # ── Stage ──────────────────────────────────────────
        self.stage       = STAGE_LOCKED
        self.arm_blocked = False

        # ── Tracking state ─────────────────────────────────
        self.rc_tracking_enabled = False
        self.offset_x   = 0.0
        self.offset_y   = 0.0
        self.has_target = False
        self.last_seen  = 0.0
        self.is_holding = False
        self.smooth_vx  = 0.0
        self.smooth_yaw = 0.0

        # ── MAVROS state ───────────────────────────────────
        self.mavros_connected = False
        self.mavros_armed     = False
        self.mavros_mode      = ''
        self.system_status    = 0
        self.last_state_time  = 0.0

        # ── ExtendedState ──────────────────────────────────
        self.landed_state       = 0
        self.ext_state_received = False

        # ── GPS ────────────────────────────────────────────
        self.gps_fix_type  = NavSatStatus.STATUS_NO_FIX
        self.gps_lat       = None
        self.gps_lon       = None
        self.gps_received  = False
        self.last_gps_time = 0.0

        # ── GPS quality ────────────────────────────────────
        self.gps_satellites   = 0
        self.gps_hdop         = 99.9
        self.gps_raw_received = False

        # ── EKF ────────────────────────────────────────────
        self.ekf_healthy    = False
        self.last_pose_time = 0.0
        self._local_pose    = None

        # ── Altitude ───────────────────────────────────────
        self.current_alt   = 0.0
        self.alt_received  = False
        self.last_alt_time = 0.0

        # ── Battery ────────────────────────────────────────
        self.battery_voltage  = None
        self.battery_received = False
        self.last_bat_time    = 0.0

        # ── Home position ──────────────────────────────────
        self.home_set = False
        self.home_lat = None
        self.home_lon = None

        # ── RC / startup lock ──────────────────────────────
        self.rc_flight_mode = None
        self.system_locked  = True
        self.last_arm_time  = 0.0
        self.ARM_COOLDOWN   = 3.0

        # ── POSHOLD setpoint ───────────────────────────────
        # GPS for reference, local pose for stable hold
        self.poshold_lat   = None
        self.poshold_lon   = None
        self.poshold_alt   = HOVER_ALT
        self.poshold_local = None   # PoseStamped snapshot

        # ── QoS ────────────────────────────────────────────
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10)
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10)
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        # ── Subscribers ────────────────────────────────────
        self.create_subscription(
            State, '/mavros/state',
            self._on_state, qos_reliable)
        self.create_subscription(
            ExtendedState, '/mavros/extended_state',
            self._on_extended_state, qos_reliable)
        self.create_subscription(
            NavSatFix, '/mavros/global_position/global',
            self._on_gps, qos_sensor)
        self.create_subscription(
            Float64, '/mavros/global_position/rel_alt',
            self._on_alt, qos_sensor)
        self.create_subscription(
            BatteryState, '/mavros/battery',
            self._on_battery, qos_sensor)
        self.create_subscription(
            HomePosition, '/mavros/home_position/home',
            self._on_home, qos_latched)
        self.create_subscription(
            GPSRAW, '/mavros/gpsstatus/gps1/raw',
            self._on_gps_raw, qos_sensor)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose',
            self._on_local_pose, qos_sensor)
        self.create_subscription(
            String, '/lawin/rc_command',
            self._on_rc_command, 10)
        self.create_subscription(
            Point, '/lawin/target_offset',
            self._on_target_offset, 10)

        # ── Publishers ─────────────────────────────────────
        self.status_pub = self.create_publisher(
            String, '/lawin/prearm_status', 10)
        self.gps_cmd_pub = self.create_publisher(
            GlobalPositionTarget,
            '/mavros/setpoint_raw/global',
            qos_sensor)
        self.pos_pub = self.create_publisher(
            PoseStamped,
            '/mavros/setpoint_position/local',
            qos_sensor)
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',
            qos_sensor)

        # ── Service clients ────────────────────────────────
        self.arming_client  = self.create_client(
            CommandBool, '/mavros/cmd/arming')
        self.mode_client    = self.create_client(
            SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(
            CommandTOL, '/mavros/cmd/takeoff')

        # ── Timers ─────────────────────────────────────────
        self.create_timer(0.05, self._main_loop)   # 20Hz
        self.create_timer(10.0, self._health_report)

        self.get_logger().info("=" * 52)
        self.get_logger().info("  LawinTracker HOVER TEST NODE")
        self.get_logger().info(
            f"  GUIDED → takeoff {HOVER_ALT}m → POSHOLD (EKF local)")
        self.get_logger().info(
            "  CH8 UP=TRACK | CH8 DOWN=POSHOLD")
        self.get_logger().info(
            "  SYSTEM LOCKED — flip CH6 to ALT_HOLD")
        self.get_logger().info("=" * 52)

    # ──────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────

    def _delay(self, seconds: float, callback):
        timer = None
        def _fire():
            timer.cancel()
            callback()
        timer = self.create_timer(seconds, _fire)

    def _is_in_air(self) -> bool:
        return self.current_alt > 2.5

    # ──────────────────────────────────────────────────────
    # MAVROS Callbacks
    # ──────────────────────────────────────────────────────

    def _on_state(self, msg: State):
        self.mavros_connected = msg.connected
        self.mavros_armed     = msg.armed
        self.mavros_mode      = msg.mode
        self.system_status    = msg.system_status
        self.last_state_time  = time.time()

    def _on_extended_state(self, msg: ExtendedState):
        self.landed_state       = msg.landed_state
        self.ext_state_received = True

    def _on_gps(self, msg: NavSatFix):
        self.gps_fix_type  = msg.status.status
        self.gps_lat       = msg.latitude
        self.gps_lon       = msg.longitude
        self.gps_received  = True
        self.last_gps_time = time.time()

    def _on_alt(self, msg: Float64):
        self.current_alt  = msg.data
        self.alt_received = True
        self.last_alt_time = time.time()

    def _on_battery(self, msg: BatteryState):
        self.battery_voltage  = msg.voltage
        self.battery_received = True
        self.last_bat_time    = time.time()

    def _on_home(self, msg: HomePosition):
        self.home_set = True
        self.home_lat = msg.geo.latitude
        self.home_lon = msg.geo.longitude

    def _on_gps_raw(self, msg: GPSRAW):
        self.gps_satellites   = msg.satellites_visible
        self.gps_hdop         = msg.eph / 100.0
        self.gps_raw_received = True

    def _on_local_pose(self, msg: PoseStamped):
        self._local_pose    = msg
        self.last_pose_time = time.time()
        if self._local_pose:
            self.ekf_healthy = True

    def _on_target_offset(self, msg: Point):
        now = self.get_clock().now().nanoseconds / 1e9
        if msg.z > 0:
            self.offset_x   = msg.x
            self.offset_y   = msg.y
            self.has_target = True
            self.last_seen  = now
        else:
            self.has_target = False

    # ──────────────────────────────────────────────────────
    # RC Command Handler
    # ──────────────────────────────────────────────────────

    def _on_rc_command(self, msg: String):
        cmd = msg.data

        if cmd == 'SYSTEM_LOCKED':
            self.system_locked = True
            return

        # ── CH6 flight mode ────────────────────────────────
        if cmd == 'ALT_HOLD':
            prev = self.rc_flight_mode
            self.rc_flight_mode = cmd
            if prev != 'ALT_HOLD':
                if self.system_locked:
                    self.system_locked = False
                    self.get_logger().info(
                        "ALT_HOLD flipped — system UNLOCKED!")
            if not self.system_locked:
                if self.stage in (STAGE_LOCKED, STAGE_IDLE):
                    self.stage = STAGE_ALT_HOLD
                    self._set_mode('ALT_HOLD')
                    self.get_logger().warn(
                        "ALT_HOLD — manual control! "
                        "CH5 UP=ARM | CH6 MID=hover mission")
                elif self.stage == STAGE_ALT_HOLD:
                    pass
                elif self.stage == STAGE_RTL:
                    self.get_logger().warn(
                        "RTL CANCELLED — manual ALT_HOLD!")
                    self.rc_tracking_enabled = False
                    self.arm_blocked = False
                    self.stage = STAGE_ALT_HOLD
                    self._set_mode('ALT_HOLD')
                else:
                    self.get_logger().warn(
                        f"ALT_HOLD noted — "
                        f"in {STAGE_NAMES[self.stage]}")

        elif cmd == 'GUIDED':
            self.rc_flight_mode = cmd
            if self.system_locked:
                self.get_logger().warn(
                    "GUIDED ignored — system locked!")
                return
            if self._is_in_air():
                self.get_logger().warn(
                    "GUIDED ignored — already in air!")
                return
            if self.stage in (STAGE_IDLE, STAGE_ALT_HOLD):
                if self.mavros_armed:
                    self.get_logger().warn(
                        "GUIDED — disarming first...")
                    self.stage = STAGE_PREARM
                    self._do_arm(False)
                    self._delay(
                        2.0, self._run_checks_then_fly)
                else:
                    self.get_logger().info(
                        "GUIDED — running pre-arm checks...")
                    self.stage = STAGE_PREARM
                    self._run_checks_then_fly()
            elif self.stage == STAGE_LOCKED:
                self.get_logger().warn(
                    "GUIDED ignored — unlock first!")
            else:
                self.get_logger().warn(
                    f"GUIDED ignored — "
                    f"in {STAGE_NAMES[self.stage]}")

        elif cmd == 'RTL':
            self.rc_flight_mode = cmd
            if self.system_locked:
                return
            self.get_logger().warn("EMERGENCY RTL!")
            self.rc_tracking_enabled = False
            self.stage = STAGE_RTL
            self.arm_blocked = True
            self._set_mode('RTL')

        # ── CH5 ARM switch ─────────────────────────────────
        elif cmd == 'ARM':
            if self.system_locked:
                self.get_logger().warn(
                    "ARM ignored — system locked!")
                return
            if self.arm_blocked:
                self.get_logger().warn(
                    "ARM blocked — flip CH5 DOWN then UP!")
                return
            if self._is_in_air():
                self.get_logger().warn(
                    "ARM ignored — already in air!")
                return
            if self.rc_flight_mode != 'ALT_HOLD':
                self.get_logger().warn(
                    f"ARM ignored — CH6 must be ALT_HOLD, "
                    f"currently '{self.rc_flight_mode}'")
                return
            if self.stage == STAGE_ALT_HOLD:
                now = time.time()
                if now - self.last_arm_time < self.ARM_COOLDOWN:
                    return
                self.last_arm_time = now
                self.get_logger().info("Manual ARM!")
                self._do_arm(True)

        elif cmd == 'DISARM':
            self.get_logger().warn("DISARM!")
            self.arm_blocked         = False
            self.rc_tracking_enabled = False
            self._do_arm(False)
            if not self.system_locked and \
                    self.rc_flight_mode == 'ALT_HOLD':
                self.stage = STAGE_ALT_HOLD
            else:
                self.stage = STAGE_IDLE

        # ── CH8 TRACK switch ───────────────────────────────
        elif cmd == 'TRACK_START':
            # Only works in GUIDED (not ALT_HOLD)
            if self.stage == STAGE_ALT_HOLD:
                self.get_logger().warn(
                    "TRACK ignored — ALT_HOLD mode. "
                    "Switch to GUIDED (CH6 MID) first!")
                return
            if not self._is_in_air():
                self.get_logger().warn(
                    "TRACK ignored — not in air!")
                return
            if self.stage == STAGE_RTL:
                self.get_logger().warn(
                    "TRACK ignored — RTL in progress!")
                return
            if self.stage in (STAGE_POSHOLD, STAGE_FOLLOW,
                              STAGE_CLIMB) or self._is_in_air():
                # Already in GUIDED — just switch to velocity mode
                # No mode change needed, GUIDED accepts both
                self.get_logger().info(
                    "TRACK_START — velocity tracking!")
                self.rc_tracking_enabled = True
                self.is_holding          = False
                self.smooth_vx           = 0.0
                self.smooth_yaw          = 0.0
                self.stage               = STAGE_FOLLOW

        elif cmd == 'TRACK_CANCEL':
            self.rc_tracking_enabled = False
            self.smooth_vx           = 0.0
            self.smooth_yaw          = 0.0
            self._send_velocity(0.0, 0.0, 0.0)
            if self.stage == STAGE_FOLLOW:
                self.get_logger().info(
                    "TRACK_CANCEL — returning to POSHOLD.")
                self._save_poshold()
                self.stage = STAGE_POSHOLD
            else:
                self.get_logger().info("Tracking CANCELLED.")

    # ──────────────────────────────────────────────────────
    # Main Loop (10Hz)
    # ──────────────────────────────────────────────────────

    def _main_loop(self):

        if self.stage == STAGE_LOCKED:
            self.get_logger().info(
                "LOCKED — flip CH6 to ALT_HOLD to unlock",
                throttle_duration_sec=10.0)

        elif self.stage == STAGE_IDLE:
            self.get_logger().info(
                "IDLE — CH6 HIGH=ALT_HOLD | CH6 MID=hover mission",
                throttle_duration_sec=5.0)

        elif self.stage == STAGE_ALT_HOLD:
            self.get_logger().info(
                "ALT_HOLD — manual control! "
                "CH5 UP=ARM | CH6 MID=hover mission",
                throttle_duration_sec=3.0)

        elif self.stage == STAGE_CLIMB:
            self.get_logger().info(
                f"Climbing... {self.current_alt:.1f}m "
                f"/ {HOVER_ALT:.1f}m",
                throttle_duration_sec=1.0)
            if self.current_alt >= HOVER_ALT - 0.4:
                self.get_logger().info(
                    f"Reached {self.current_alt:.1f}m! "
                    f"Entering POSHOLD...")
                self._save_poshold()
                self.stage = STAGE_POSHOLD

        elif self.stage == STAGE_POSHOLD:
            # Publish LOCAL position setpoint (EKF-based)
            # Much more stable than GPS setpoints
            if self.poshold_local is not None:
                self.poshold_local.header.stamp = (
                    self.get_clock().now().to_msg())
                self.pos_pub.publish(self.poshold_local)
            self.get_logger().info(
                f"POSHOLD — holding at {self.current_alt:.1f}m | "
                f"CH8 UP=TRACK | CH6 LOW=RTL",
                throttle_duration_sec=3.0)

        elif self.stage == STAGE_FOLLOW:
            if self.rc_tracking_enabled:
                self._follow_target()
            else:
                self._send_velocity(0.0, 0.0, 0.0)
                self._save_poshold()
                self.stage = STAGE_POSHOLD

        elif self.stage == STAGE_RTL:
            self.get_logger().info(
                "RTL in progress... "
                "flip CH5 DOWN then UP to re-arm",
                throttle_duration_sec=2.0)

    # ──────────────────────────────────────────────────────
    # POSHOLD helpers
    # ──────────────────────────────────────────────────────

    def _save_poshold(self):
        """Snapshot current LOCAL pose for stable position hold.
        Uses EKF local frame — much more stable than GPS.
        """
        if self._local_pose is not None:
            # Snapshot current local pose
            self.poshold_local = PoseStamped()
            self.poshold_local.header.frame_id = 'map'
            self.poshold_local.pose = \
                self._local_pose.pose
            self.get_logger().info(
                f"POSHOLD saved (local EKF): "
                f"x={self.poshold_local.pose.position.x:.2f} "
                f"y={self.poshold_local.pose.position.y:.2f} "
                f"z={self.poshold_local.pose.position.z:.2f}")
        elif self.gps_lat is not None:
            # Fallback to GPS if local pose unavailable
            self.poshold_lat = self.gps_lat
            self.poshold_lon = self.gps_lon
            self.get_logger().warn(
                "POSHOLD: using GPS fallback (no local pose)")

    def _make_gps_cmd(self, lat, lon, alt):
        from mavros_msgs.msg import GlobalPositionTarget
        TYPE_MASK = (
            GlobalPositionTarget.IGNORE_VX
            | GlobalPositionTarget.IGNORE_VY
            | GlobalPositionTarget.IGNORE_VZ
            | GlobalPositionTarget.IGNORE_AFX
            | GlobalPositionTarget.IGNORE_AFY
            | GlobalPositionTarget.IGNORE_AFZ
            | GlobalPositionTarget.IGNORE_YAW
            | GlobalPositionTarget.IGNORE_YAW_RATE
        )
        cmd = GlobalPositionTarget()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'map'
        cmd.coordinate_frame = 6  # MAV_FRAME_GLOBAL_RELATIVE_ALT
        cmd.type_mask       = TYPE_MASK
        cmd.latitude        = float(lat)
        cmd.longitude       = float(lon)
        cmd.altitude        = float(alt)
        return cmd

    # ──────────────────────────────────────────────────────
    # Person Tracking
    # ──────────────────────────────────────────────────────

    def _follow_target(self):
        """Full tracking — yaw + forward/back toward centroid."""
        if not self.has_target:
            return

        err_x    = self.offset_x
        err_y    = self.offset_y
        distance = math.sqrt(err_x**2 + err_y**2)

        if self.is_holding:
            if distance > START_ZONE:
                self.is_holding = False
            else:
                self.smooth_vx  = self._smooth(self.smooth_vx,  0.0)
                self.smooth_yaw = self._smooth(self.smooth_yaw, 0.0)
                self._send_velocity(self.smooth_vx, 0.0, 0.0,
                                    yaw=self.smooth_yaw)
                self.get_logger().info(
                    f"HOLD (dist={distance:.0f}px)",
                    throttle_duration_sec=1.0)
                return
        else:
            if distance < STOP_ZONE:
                self.is_holding = True
                self.smooth_vx  = self._smooth(self.smooth_vx,  0.0)
                self.smooth_yaw = self._smooth(self.smooth_yaw, 0.0)
                self._send_velocity(self.smooth_vx, 0.0, 0.0,
                                    yaw=self.smooth_yaw)
                self.get_logger().info(
                    f"HOLD — centered! (dist={distance:.0f}px)")
                return

        # Proportional yaw + forward controller
        raw_yaw = -KP_YAW * err_x
        raw_vx  = -KP_FWD * err_y
        raw_yaw = max(-MAX_YAW, min(MAX_YAW, raw_yaw))
        raw_vx  = max(-MAX_VEL_X, min(MAX_VEL_X, raw_vx))

        self.smooth_yaw = self._smooth(self.smooth_yaw, raw_yaw)
        self.smooth_vx  = self._smooth(self.smooth_vx,  raw_vx)
        self._send_velocity(self.smooth_vx, 0.0, 0.0,
                            yaw=self.smooth_yaw)

        h = "RIGHT" if err_x > 0 else "LEFT"
        v = "BACK"  if err_y > 0 else "FWD"
        self.get_logger().info(
            f"TRACK {h}/{v} | "
            f"err=({err_x:+.0f},{err_y:+.0f})px | "
            f"yaw={self.smooth_yaw:+.3f} | "
            f"vx={self.smooth_vx:+.2f}m/s | "
            f"dist={distance:.0f}px",
            throttle_duration_sec=0.3)

    def _smooth(self, current: float, target: float) -> float:
        result = SMOOTH * current + (1.0 - SMOOTH) * target
        return 0.0 if abs(result) < MIN_VEL else result

    def _send_velocity(self, vx: float, vy: float, vz: float,
                       yaw: float = 0.0):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x  = float(vx)
        msg.twist.linear.y  = float(vy)
        msg.twist.linear.z  = float(vz)
        msg.twist.angular.z = float(yaw)
        self.vel_pub.publish(msg)

    # ──────────────────────────────────────────────────────
    # Pre-Arm Checks (same 14 checks as test.py)
    # ──────────────────────────────────────────────────────

    def _run_checks_then_fly(self):
        now      = time.time()
        items    = []
        failures = []
        warnings = []

        age = now - self.last_state_time
        if not self.mavros_connected:
            s = FAIL; d = "FCU not connected"
            failures.append("MAVROS not connected")
        elif age > 3.0:
            s = WARN; d = f"State stale ({age:.1f}s)"
            warnings.append("MAVROS state stale")
        else:
            s = PASS; d = f"Connected (age {age:.1f}s)"
        items.append(("[1]  MAVROS Connection ", s, d))

        if self.mavros_armed:
            s = FAIL; d = "Already ARMED"
            failures.append("Already armed")
        else:
            s = PASS; d = "Disarmed — safe to arm"
        items.append(("[2]  Not Already Armed ", s, d))

        if self.system_locked:
            s = FAIL; d = "Locked — flip CH6 to ALT_HOLD"
            failures.append("System still locked")
        else:
            s = PASS; d = "Unlocked by ALT_HOLD flip"
        items.append(("[3]  System Unlocked   ", s, d))

        if not self.gps_received:
            s = UNKNOWN; d = "No GPS data yet"
            warnings.append("GPS not received")
        elif self.gps_fix_type == NavSatStatus.STATUS_NO_FIX:
            s = FAIL; d = "No fix — go outside"
            failures.append("No GPS fix")
        else:
            s = PASS; d = f"Fix type {self.gps_fix_type}"
        items.append(("[4]  GPS Fix           ", s, d))

        gps_age = (now - self.last_gps_time
                   if self.gps_received else 999)
        if not self.gps_received or self.gps_lat is None:
            s = UNKNOWN; d = "No position data"
            warnings.append("GPS position unknown")
        elif abs(self.gps_lat) < 0.1 and abs(self.gps_lon) < 0.1:
            s = FAIL; d = "Position (0,0) — invalid"
            failures.append("GPS at (0,0)")
        elif gps_age > 3.0:
            s = WARN; d = f"Stale ({gps_age:.1f}s)"
            warnings.append("GPS data stale")
        else:
            s = PASS
            d = f"lat={self.gps_lat:.6f} lon={self.gps_lon:.6f}"
        items.append(("[5]  GPS Position      ", s, d))

        if not self.gps_raw_received:
            s = UNKNOWN; d = "No GPS raw data"
            warnings.append("GPS raw not received")
        elif self.gps_satellites < GPS_MIN_SATELLITES:
            s = FAIL
            d = (f"{self.gps_satellites} sats — "
                 f"need {GPS_MIN_SATELLITES}+")
            failures.append(
                f"Only {self.gps_satellites} satellites")
        else:
            s = PASS; d = f"{self.gps_satellites} satellites"
        items.append(("[6]  GPS Satellites    ", s, d))

        if not self.gps_raw_received:
            s = UNKNOWN; d = "No GPS raw data"
            warnings.append("HDOP unknown")
        elif self.gps_hdop > GPS_MAX_HDOP:
            s = FAIL
            d = f"HDOP {self.gps_hdop:.2f} > {GPS_MAX_HDOP}"
            failures.append(f"HDOP {self.gps_hdop:.2f}")
        else:
            s = PASS; d = f"HDOP {self.gps_hdop:.2f} OK"
        items.append(("[7]  GPS HDOP          ", s, d))

        pose_age = time.time() - self.last_pose_time
        if self.last_pose_time == 0.0:
            s = FAIL; d = "EKF not publishing"
            failures.append("EKF not running")
        elif pose_age > EKF_POSE_MAX_AGE:
            s = FAIL; d = f"EKF stale ({pose_age:.1f}s)"
            failures.append("EKF stale")
        elif not self.ekf_healthy:
            s = WARN; d = "EKF initialising"
            warnings.append("EKF initialising")
        else:
            s = PASS; d = f"EKF healthy ({pose_age:.2f}s)"
        items.append(("[8]  EKF Health        ", s, d))

        if not self.alt_received:
            s = UNKNOWN; d = "No altitude data"
            warnings.append("Altitude unknown")
        elif self._is_in_air():
            s = FAIL; d = f"Alt {self.current_alt:.1f}m — in air!"
            failures.append(f"In air: {self.current_alt:.1f}m")
        else:
            s = PASS; d = f"Alt {self.current_alt:.2f}m on ground"
        items.append(("[9]  On Ground         ", s, d))

        if not self.ext_state_received:
            s = UNKNOWN
            d = "No ExtendedState (Pixhawk1 may not publish)"
            warnings.append("Landed state unknown")
        elif self.landed_state == 1:
            s = PASS; d = "ON_GROUND confirmed"
        elif self.landed_state == 2:
            s = FAIL; d = "IN_AIR — cannot arm"
            failures.append("ExtendedState: IN_AIR")
        else:
            name = LANDED_NAMES.get(self.landed_state, "?")
            s = WARN; d = f"State: {name}"
            warnings.append(f"Landed: {name}")
        items.append(("[10] Landed State      ", s, d))

        unsafe = {'AUTO', 'RTL', 'LAND', 'LOITER', 'GUIDED'}
        mode = self.mavros_mode
        if mode in unsafe:
            s = FAIL; d = f"'{mode}' unsafe for arming"
            failures.append(f"Unsafe mode: {mode}")
        else:
            s = PASS; d = f"Mode '{mode}' OK"
        items.append(("[11] Flight Mode       ", s, d))

        bat_age = (now - self.last_bat_time
                   if self.battery_received else 999)
        if not self.battery_received or \
                self.battery_voltage is None:
            s = FAIL; d = "No data — check BATT_MONITOR"
            failures.append("No battery data")
        elif self.battery_voltage < 0.1:
            s = WARN; d = "0.00V — BATT_MONITOR=0"
            warnings.append("Battery monitor not configured")
        elif self.battery_voltage < BATTERY_MIN_VOLTAGE:
            s = FAIL
            d = (f"{self.battery_voltage:.2f}V < "
                 f"{BATTERY_MIN_VOLTAGE:.1f}V — CHARGE!")
            failures.append(
                f"Low battery: {self.battery_voltage:.2f}V")
        else:
            s = PASS
            d = f"{self.battery_voltage:.2f}V (age {bat_age:.1f}s)"
        items.append(("[12] Battery Voltage   ", s, d))

        if not self.home_set:
            s = WARN; d = "Not set (auto on GPS lock)"
            warnings.append("Home not set")
        else:
            s = PASS
            d = f"lat={self.home_lat:.6f} lon={self.home_lon:.6f}"
        items.append(("[13] Home Position     ", s, d))

        sys_name = MAV_STATE_NAMES.get(self.system_status, "?")
        if not self.mavros_connected:
            s = UNKNOWN; d = "No connection"
        elif self.system_status in [3, 4]:
            s = PASS; d = f"{sys_name}"
        elif self.system_status in [0, 1, 2]:
            s = WARN; d = f"{sys_name} — initialising"
            warnings.append(f"FCU: {sys_name}")
        else:
            s = FAIL; d = f"{sys_name} — CRITICAL!"
            failures.append(f"FCU: {sys_name}")
        items.append(("[14] System Status     ", s, d))

        n_fail  = len(failures)
        n_warn  = len(warnings)
        n_pass  = 14 - n_fail - n_warn
        cleared = (n_fail == 0)
        verdict = "PRE-ARM CLEARED" if cleared else "PRE-ARM BLOCKED"
        ICON = {PASS:"OK", FAIL:"XX", WARN:"!!", UNKNOWN:"??"}
        div = "-" * 56

        self.get_logger().info(div)
        self.get_logger().info(" HoverTest PRE-ARM CHECK REPORT")
        self.get_logger().info(div)
        for label, status, detail in items:
            self.get_logger().info(
                f"  [{ICON.get(status,'??')}] "
                f"{label}: {status}")
            self.get_logger().info(f"           {detail}")
        self.get_logger().info(div)
        self.get_logger().info(f"  Result : {verdict}")
        self.get_logger().info(
            f"  Score  : {n_pass}/14 PASS | "
            f"{n_warn} WARN | {n_fail} FAIL")
        if failures:
            self.get_logger().info("  Blocking:")
            for f in failures:
                self.get_logger().info(f"    X {f}")
        if warnings:
            self.get_logger().info("  Warnings:")
            for w in warnings:
                self.get_logger().info(f"    ! {w}")
        self.get_logger().info(div)

        pub = String()
        pub.data = (f"{verdict} | {n_pass}/14 PASS "
                    f"| {n_warn} WARN | {n_fail} FAIL")
        if failures:
            pub.data += " | " + ", ".join(failures)
        self.status_pub.publish(pub)

        if cleared:
            self.get_logger().info(
                "All checks passed — taking off!")
            self._start_guided_sequence()
        else:
            self.get_logger().warn(
                f"BLOCKED — fix {n_fail} issue(s)!")
            self.stage = STAGE_ALT_HOLD

    # ──────────────────────────────────────────────────────
    # GUIDED → ARM → TAKEOFF Sequence
    # ──────────────────────────────────────────────────────

    def _start_guided_sequence(self):
        self.get_logger().info(
            "Step 1/3: Setting GUIDED mode...")
        self.stage = STAGE_SET_GUIDED
        self._set_mode('GUIDED', callback=self._on_guided_set)

    def _on_guided_set(self, future):
        try:
            result = future.result()
            if result and result.mode_sent:
                self.get_logger().info(
                    "GUIDED set! Step 2/3: Arming...")
                self.stage = STAGE_ARM
                self._delay(1.0, lambda: self._do_arm(True))
                self._delay(4.0, self._check_armed_then_takeoff)
            else:
                self.get_logger().error(
                    "GUIDED failed! Retrying...")
                self._delay(2.0, self._start_guided_sequence)
        except Exception as e:
            self.get_logger().error(f"GUIDED error: {e}")
            self._delay(2.0, self._start_guided_sequence)

    def _check_armed_then_takeoff(self):
        if self.stage not in (STAGE_ARM, STAGE_TAKEOFF):
            return
        if self.mavros_armed:
            self.get_logger().info(
                f"Armed! Step 3/3: Takeoff to {HOVER_ALT}m...")
            self.stage = STAGE_TAKEOFF
            self._do_takeoff(HOVER_ALT)
        else:
            self.get_logger().warn("Waiting for arm...")
            self._delay(1.0, self._check_armed_then_takeoff)

    def _on_takeoff_result(self, future):
        try:
            result = future.result()
            if result and result.success:
                self.get_logger().info(
                    f"Takeoff accepted! "
                    f"Climbing to {HOVER_ALT}m...")
                self.stage = STAGE_CLIMB
            else:
                self.get_logger().error(
                    "Takeoff rejected! Retrying...")
                self._delay(
                    2.0, lambda: self._do_takeoff(HOVER_ALT))
        except Exception as e:
            self.get_logger().error(f"Takeoff error: {e}")
            self._delay(2.0, lambda: self._do_takeoff(HOVER_ALT))

    # ──────────────────────────────────────────────────────
    # MAVROS Service Wrappers
    # ──────────────────────────────────────────────────────

    def _set_mode(self, mode: str, callback=None):
        if not self.mode_client.service_is_ready():
            self._delay(1.0, lambda: self._set_mode(
                mode, callback))
            return
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.mode_client.call_async(req)
        if callback:
            future.add_done_callback(lambda f: callback(f))
        else:
            future.add_done_callback(
                lambda f: self.get_logger().info(
                    f"Mode set: {mode}"))

    def _do_arm(self, arm: bool):
        action = "ARM" if arm else "DISARM"
        if not self.arming_client.service_is_ready():
            self.get_logger().error(
                f"{action}: arming service not ready!")
            return
        req = CommandBool.Request()
        req.value = arm
        future = self.arming_client.call_async(req)
        future.add_done_callback(
            lambda f: self._on_arm_result(f, arm))

    def _on_arm_result(self, future, arm: bool):
        action = "ARM" if arm else "DISARM"
        try:
            result = future.result()
            if result and result.success:
                self.get_logger().info(
                    f"Pixhawk accepted {action}!")
                if not arm:
                    self.stage = STAGE_IDLE
            else:
                self.get_logger().error(
                    f"Pixhawk rejected {action}!")
                if self.stage == STAGE_ARM:
                    self.stage = STAGE_ALT_HOLD
        except Exception as e:
            self.get_logger().error(f"{action} failed: {e}")

    def _do_takeoff(self, alt: float):
        if not self.takeoff_client.service_is_ready():
            self._delay(1.0, lambda: self._do_takeoff(alt))
            return
        req = CommandTOL.Request()
        req.altitude  = float(alt)
        req.latitude  = float(self.gps_lat or 0)
        req.longitude = float(self.gps_lon or 0)
        req.min_pitch = 0.0
        req.yaw       = 0.0
        future = self.takeoff_client.call_async(req)
        future.add_done_callback(self._on_takeoff_result)

    # ──────────────────────────────────────────────────────
    # Health Report (every 10s)
    # ──────────────────────────────────────────────────────

    def _health_report(self):
        if not self.mavros_connected:
            return
        gps_ok = (self.gps_received and
                  self.gps_fix_type != NavSatStatus.STATUS_NO_FIX)
        landed = LANDED_NAMES.get(self.landed_state, "?")
        bat_str = (f"{self.battery_voltage:.2f}V"
                   if self.battery_voltage else "?")
        self.get_logger().info(
            f"Health | "
            f"{'LOCKED' if self.system_locked else 'UNLOCKED'} "
            f"Stage:{STAGE_NAMES.get(self.stage,'?')} "
            f"Armed:{self.mavros_armed} "
            f"Tracking:{self.rc_tracking_enabled} "
            f"Target:{self.has_target} "
            f"RCMode:{self.rc_flight_mode or '?'} "
            f"GPS:{'OK' if gps_ok else 'NO'} "
            f"Sats:{self.gps_satellites} "
            f"HDOP:{self.gps_hdop:.2f} "
            f"EKF:{'OK' if self.ekf_healthy else 'NO'} "
            f"Alt:{self.current_alt:.1f}m "
            f"Gnd:{landed} "
            f"Bat:{bat_str} "
            f"FCUMode:{self.mavros_mode}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = OrigHoverTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

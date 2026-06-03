#!/usr/bin/env python3
"""
LawinTracker Test Node (test.py)
=================================
Standalone test — run INSTEAD of flight_controller.py.
DO NOT run alongside flight_controller.py.

Mirrors flight_controller.py behaviour:

  ALT_HOLD mode (CH6 HIGH):
    → Pixhawk set to ALT_HOLD
    → Drone holds altitude automatically
    → Pilot arms with CH5 UP
    → Fly manually with sticks (altitude locked)
    → CH8 UP = start tracking while in air
    → CH5 DOWN = disarm

  GUIDED mode (CH6 MID):
    → Pre-arm checks (14 checks)
    → If cleared: set GUIDED → arm → takeoff 3m
      → zigzag mission (A→B→C→D→E) → RTL

  RTL mode (CH6 LOW):
    → Emergency RTL immediately
    → arm_blocked set — must flip CH5 DOWN then UP to re-arm

  TRACK_START (CH8 UP):
    → Only works in GUIDED mode (not ALT_HOLD)
    → During zigzag mission: pauses ONLY if person detected
      → tracks person, resumes mission on TRACK_CANCEL
    → During climb/hover: tracks person directly
    → ALT_HOLD → ignored (tracking requires GUIDED)

  TRACK_CANCEL (CH8 DOWN):
    → If mission was paused → resume from saved waypoint
    → Otherwise → stop tracking, hold in GUIDED

Startup lock:
    → System boots LOCKED
    → Pilot must explicitly flip CH6 to ALT_HOLD to unlock
    → Even if CH6 is already in ALT_HOLD at boot, doesn't count

RC Mapping:
  CH5 UP   = ARM    (ALT_HOLD only, after unlock)
  CH5 DOWN = DISARM (always works)
  CH6 HIGH = ALT_HOLD
  CH6 MID  = GUIDED → zigzag mission
  CH6 LOW  = RTL    → emergency

Zigzag Pattern:
  A    C    E  (RTL after E)
  Home -> B -> D

Tracking modes:
  Default    = FBRL strafe (linear.x / linear.y, no yaw)
  Yaw mode   = ros2 run lawin_mavros test --ros-args -p yaw_tracking:=true

Run:
  ros2 run lawin_mavros test
  ros2 run lawin_mavros test --ros-args -p yaw_tracking:=true
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


# ── Mission config ──────────────────────────────────────────────
TAKEOFF_ALT  = 5.0    # m — takeoff altitude
STEP_FORWARD = 2.0    # m North per leg
STEP_SIDE    = 2.0    # m East/West per leg
WP_THRESHOLD = 0.30   # m — "close enough" to waypoint
WP_TIMEOUT   = 30     # s — skip waypoint if not reached

# MAV_FRAME_GLOBAL_RELATIVE_ALT
MAV_FRAME_GLOBAL_RELATIVE_ALT = 6

# GlobalPositionTarget type_mask — position only
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
STAGE_MISSION    = 8
STAGE_FOLLOW     = 9
STAGE_RTL        = 10
STAGE_TRACK_WAIT = 11   # mission paused, person detected, tracking

STAGE_NAMES = {
    0: "LOCKED",     1: "IDLE",       2: "ALT_HOLD",
    3: "PREARM",     4: "SET_GUIDED", 5: "ARM",
    6: "TAKEOFF",    7: "CLIMB",      8: "MISSION",
    9: "FOLLOW",     10: "RTL",       11: "TRACK_WAIT"
}

TRACK_WAIT_TIMEOUT = 5.0   # s — resume mission if person disappears

# ── Tracking controller config ───────────────────────────────────
KP_YAW       = 0.003
KP_FWD       = 0.010
MAX_YAW      = 0.8
MAX_VEL_X    = 5.0
MIN_VEL      = 0.01
SMOOTH       = 0.7
STOP_ZONE    = 15
START_ZONE   = 35
LOST_TIMEOUT = 2.0

PASS    = "PASS"
FAIL    = "FAIL"
WARN    = "WARN"
UNKNOWN = "UNKN"


class TestNode(Node):

    def __init__(self):
        super().__init__('lawin_test')

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
        self.smooth_vy  = 0.0
        self.smooth_yaw = 0.0

        # FBRL strafe by default; pass --ros-args -p yaw_tracking:=true to enable yaw
        self.declare_parameter('yaw_tracking', False)
        self.fbrl_only = not self.get_parameter('yaw_tracking').value

        # ── Mission pause state ────────────────────────────
        self.mission_paused   = False
        self.resume_wp_idx    = 0
        self.track_wait_start = 0.0

        # ── Mission state ──────────────────────────────────
        self.home_gps       = None   # (lat, lon) snapshot at climb
        self.waypoints      = []     # list of (name, lat, lon)
        self.current_wp_idx = 0
        self.wp_start_time  = 0.0

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

        # GPS setpoint publisher for zigzag navigation
        self.gps_cmd_pub = self.create_publisher(
            GlobalPositionTarget,
            '/mavros/setpoint_raw/global',
            qos_sensor)

        # Velocity publisher for person tracking
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
        self.create_timer(0.1,  self._main_loop)
        self.create_timer(10.0, self._health_report)

        mode_str = "YAW+FORWARD" if not self.fbrl_only else "FBRL STRAFE"
        self.get_logger().info("=" * 52)
        self.get_logger().info("  LawinTracker TEST NODE ready")
        self.get_logger().info(f"  Tracking mode : {mode_str}")
        self.get_logger().info("  SYSTEM LOCKED — flip CH6 to ALT_HOLD")
        self.get_logger().info("  GUIDED → zigzag mission A→B→C→D→E → RTL")
        self.get_logger().info("  CH8 UP=TRACK (GUIDED only, person must be visible)")
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
        return self.current_alt > 0.1

    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2) -> float:
        """Distance in metres between two GPS coords."""
        dlat = (lat2 - lat1) * 111_111.0
        dlon = (lon2 - lon1) * 111_111.0 * math.cos(
            math.radians(lat1))
        return math.hypot(dlat, dlon)

    @staticmethod
    def _make_zigzag(lat, lon) -> list:
        """Build 5 zigzag waypoints from home GPS."""
        m_lat = 1.0 / 111_111.0
        m_lon = 1.0 / (111_111.0 * math.cos(math.radians(lat)))
        return [
            ('A', lat + 1*STEP_FORWARD*m_lat,
                  lon + STEP_SIDE*m_lon),
            ('B', lat + 2*STEP_FORWARD*m_lat,
                  lon - STEP_SIDE*m_lon),
            ('C', lat + 3*STEP_FORWARD*m_lat,
                  lon + STEP_SIDE*m_lon),
            ('D', lat + 4*STEP_FORWARD*m_lat,
                  lon - STEP_SIDE*m_lon),
            ('E', lat + 5*STEP_FORWARD*m_lat,
                  lon + STEP_SIDE*m_lon),
        ]

    def _make_gps_cmd(self, lat, lon, alt) -> GlobalPositionTarget:
        msg = GlobalPositionTarget()
        msg.header = Header()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.coordinate_frame = MAV_FRAME_GLOBAL_RELATIVE_ALT
        msg.type_mask        = TYPE_MASK_POS_ONLY
        msg.latitude         = lat
        msg.longitude        = lon
        msg.altitude         = float(alt)
        return msg

    # ──────────────────────────────────────────────────────
    # Sensor Callbacks
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

    def _on_gps_raw(self, msg: GPSRAW):
        self.gps_satellites   = msg.satellites_visible
        self.gps_hdop         = msg.eph / 100.0
        self.gps_raw_received = True

    def _on_local_pose(self, msg: PoseStamped):
        self._local_pose    = msg
        self.last_pose_time = time.time()
        p = msg.pose.position
        self.ekf_healthy = not (
            p.x == 0.0 and p.y == 0.0 and p.z == 0.0)

    def _on_alt(self, msg: Float64):
        self.current_alt   = msg.data
        self.alt_received  = True
        self.last_alt_time = time.time()

    def _on_battery(self, msg: BatteryState):
        self.battery_voltage  = msg.voltage
        self.battery_received = True
        self.last_bat_time    = time.time()

    def _on_home(self, msg: HomePosition):
        self.home_set = True
        self.home_lat = msg.geo.latitude
        self.home_lon = msg.geo.longitude

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

        # ── CH6 flight mode switch ─────────────────────────
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
                        "ALT_HOLD — altitude held! "
                        "CH5 UP=ARM | CH8 UP=TRACK (in air)")
                elif self.stage == STAGE_ALT_HOLD:
                    pass  # already in ALT_HOLD
                else:
                    self.get_logger().warn(
                        f"ALT_HOLD noted — "
                        f"currently in {STAGE_NAMES[self.stage]}")

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
                    self._delay(2.0, self._run_checks_then_fly)
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
            self.mission_paused      = False
            self._do_arm(False)
            if not self.system_locked and \
                    self.rc_flight_mode == 'ALT_HOLD':
                self.stage = STAGE_ALT_HOLD
            else:
                self.stage = STAGE_IDLE

        # ── CH8 TRACK switch ───────────────────────────────
        elif cmd == 'TRACK_START':
            # Only works in GUIDED — ALT_HOLD ignores velocity cmds
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

            if self.stage == STAGE_MISSION:
                # During zigzag mission: only pause if person detected
                if self.has_target:
                    self.get_logger().info(
                        f"Person detected! Pausing mission at "
                        f"WP {self.waypoints[self.current_wp_idx][0]}"
                        f" → tracking!")
                    self._pause_mission()
                else:
                    self.get_logger().warn(
                        "TRACK_START — no person detected, "
                        "continuing mission. "
                        "CH8 UP again when person is visible.")

            elif self.stage == STAGE_TRACK_WAIT:
                self.get_logger().info("Already in TRACK_WAIT...")

            elif self.stage in (STAGE_CLIMB, STAGE_FOLLOW):
                self.get_logger().info("TRACK_START — tracking!")
                self.rc_tracking_enabled = True
                self.is_holding          = False
                self.smooth_vx           = 0.0
                self.smooth_vy           = 0.0
                self.smooth_yaw          = 0.0
                self.stage               = STAGE_FOLLOW

            elif self._is_in_air():
                # In air in some other GUIDED stage
                self.get_logger().info("TRACK_START — tracking!")
                self.rc_tracking_enabled = True
                self.is_holding          = False
                self.smooth_vx           = 0.0
                self.smooth_vy           = 0.0
                self.smooth_yaw          = 0.0
                self.stage               = STAGE_FOLLOW

        elif cmd == 'TRACK_CANCEL':
            self.rc_tracking_enabled = False
            self.smooth_vx           = 0.0
            self.smooth_vy           = 0.0
            self.smooth_yaw          = 0.0
            self._send_velocity(0.0, 0.0, 0.0)

            if self.mission_paused:
                # Was tracking during mission → resume from saved WP
                self.get_logger().info(
                    "TRACK_CANCEL — resuming mission!")
                self._resume_mission()
            elif self.stage in (STAGE_FOLLOW, STAGE_TRACK_WAIT):
                self.get_logger().info(
                    "TRACK_CANCEL — holding in GUIDED.")
                self.stage = STAGE_CLIMB
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
                "IDLE — CH6 HIGH=ALT_HOLD | CH6 MID=GUIDED mission",
                throttle_duration_sec=5.0)

        elif self.stage == STAGE_ALT_HOLD:
            self.get_logger().info(
                "ALT_HOLD — altitude held! "
                "CH5 UP=ARM | CH6 MID=GUIDED mission "
                "(tracking only in GUIDED)",
                throttle_duration_sec=3.0)

        elif self.stage == STAGE_CLIMB:
            self.get_logger().info(
                f"Climbing... {self.current_alt:.1f}m "
                f"/ {TAKEOFF_ALT:.1f}m",
                throttle_duration_sec=1.0)
            if self.current_alt >= TAKEOFF_ALT - 0.4:
                self.get_logger().info(
                    f"Reached {self.current_alt:.1f}m! "
                    f"Starting zigzag mission...")
                self._start_zigzag()

        elif self.stage == STAGE_MISSION:
            self._mission_tick()

        elif self.stage == STAGE_TRACK_WAIT:
            # Mission paused — person detected, now tracking
            elapsed = time.time() - self.track_wait_start
            if self.has_target:
                # Person visible → track
                if not self.rc_tracking_enabled:
                    self.rc_tracking_enabled = True
                    self.is_holding          = False
                    self.stage               = STAGE_FOLLOW
            elif elapsed > TRACK_WAIT_TIMEOUT:
                # Person disappeared too long → resume mission
                self.get_logger().warn(
                    f"Person lost for {TRACK_WAIT_TIMEOUT:.0f}s "
                    f"— resuming mission!")
                self._resume_mission()
            else:
                self.get_logger().info(
                    f"Tracking... lost person for "
                    f"{elapsed:.1f}s/{TRACK_WAIT_TIMEOUT:.0f}s",
                    throttle_duration_sec=1.0)
                self._send_velocity(0.0, 0.0, 0.0)

        elif self.stage == STAGE_FOLLOW:
            if self.rc_tracking_enabled:
                self._follow_target()
            else:
                self._send_velocity(0.0, 0.0, 0.0)
                if self.mission_paused:
                    self._resume_mission()
                else:
                    self.stage = STAGE_CLIMB

        elif self.stage == STAGE_RTL:
            self.get_logger().info(
                "RTL in progress... "
                "flip CH5 DOWN then UP to re-arm",
                throttle_duration_sec=2.0)

    # ──────────────────────────────────────────────────────
    # Mission Pause / Resume
    # ──────────────────────────────────────────────────────

    def _pause_mission(self):
        """Pause zigzag mission and start tracking."""
        self.mission_paused      = True
        self.resume_wp_idx       = self.current_wp_idx
        self.track_wait_start    = time.time()
        self.rc_tracking_enabled = True
        self.is_holding          = False
        self.smooth_vx           = 0.0
        self.smooth_vy           = 0.0
        self.smooth_yaw          = 0.0
        self.stage = STAGE_FOLLOW
        self.get_logger().info(
            f"Mission PAUSED at WP "
            f"{self.waypoints[self.resume_wp_idx][0]} "
            f"— tracking person. CH8 DOWN to resume.")

    def _resume_mission(self):
        """Resume zigzag from saved waypoint index."""
        self.rc_tracking_enabled = False
        self.mission_paused      = False
        self.current_wp_idx      = self.resume_wp_idx
        self.wp_start_time       = time.time()
        self.stage               = STAGE_MISSION
        name = (self.waypoints[self.current_wp_idx][0]
                if self.waypoints else '?')
        self.get_logger().info(
            f"Mission RESUMED from WP {name}!")

    # ──────────────────────────────────────────────────────
    # Person Tracking
    # ──────────────────────────────────────────────────────

    def _follow_target(self):
        """Tracking controller — FBRL strafe (default) or yaw+forward."""
        now  = self.get_clock().now().nanoseconds / 1e9
        lost = (not self.has_target or
                (now - self.last_seen) > LOST_TIMEOUT)

        if lost:
            self.smooth_vx  = self._smooth(self.smooth_vx,  0.0)
            self.smooth_vy  = self._smooth(self.smooth_vy,  0.0)
            self.smooth_yaw = self._smooth(self.smooth_yaw, 0.0)
            if self.fbrl_only:
                self._send_velocity(self.smooth_vx, self.smooth_vy,
                                    0.0, yaw=0.0)
            else:
                self._send_velocity(self.smooth_vx, 0.0,
                                    0.0, yaw=self.smooth_yaw)
            if (abs(self.smooth_vx)  < MIN_VEL and
                    abs(self.smooth_vy)  < MIN_VEL and
                    abs(self.smooth_yaw) < MIN_VEL):
                self.smooth_vx           = 0.0
                self.smooth_vy           = 0.0
                self.smooth_yaw          = 0.0
                self.rc_tracking_enabled = False
                self.get_logger().warn("Target lost!")
                if self.mission_paused:
                    self.stage            = STAGE_TRACK_WAIT
                    self.track_wait_start = (
                        self.get_clock().now().nanoseconds / 1e9)
                else:
                    self.stage = STAGE_CLIMB
            return

        err_x    = self.offset_x
        err_y    = self.offset_y
        distance = math.sqrt(err_x**2 + err_y**2)

        if self.is_holding:
            if distance > START_ZONE:
                self.is_holding = False
            else:
                self.smooth_vx  = self._smooth(self.smooth_vx,  0.0)
                self.smooth_vy  = self._smooth(self.smooth_vy,  0.0)
                self.smooth_yaw = self._smooth(self.smooth_yaw, 0.0)
                if self.fbrl_only:
                    self._send_velocity(self.smooth_vx, self.smooth_vy,
                                        0.0, yaw=0.0)
                else:
                    self._send_velocity(self.smooth_vx, 0.0,
                                        0.0, yaw=self.smooth_yaw)
                self.get_logger().info(
                    f"HOLD (dist={distance:.0f}px)",
                    throttle_duration_sec=1.0)
                return
        else:
            if distance < STOP_ZONE:
                self.is_holding = True
                self.smooth_vx  = self._smooth(self.smooth_vx,  0.0)
                self.smooth_vy  = self._smooth(self.smooth_vy,  0.0)
                self.smooth_yaw = self._smooth(self.smooth_yaw, 0.0)
                if self.fbrl_only:
                    self._send_velocity(self.smooth_vx, self.smooth_vy,
                                        0.0, yaw=0.0)
                else:
                    self._send_velocity(self.smooth_vx, 0.0,
                                        0.0, yaw=self.smooth_yaw)
                self.get_logger().info(
                    f"HOLD — centered! (dist={distance:.0f}px)")
                return

        h = "RIGHT" if err_x > 0 else "LEFT"
        v = "BACK"  if err_y > 0 else "FWD"

        if self.fbrl_only:
            # ── FBRL strafe — no yaw ───────────────────────
            raw_vx = max(-MAX_VEL_X, min(MAX_VEL_X, -KP_FWD * err_y))
            raw_vy = max(-MAX_VEL_X, min(MAX_VEL_X, -KP_FWD * err_x))
            self.smooth_vx = self._smooth(self.smooth_vx, raw_vx)
            self.smooth_vy = self._smooth(self.smooth_vy, raw_vy)
            self._send_velocity(self.smooth_vx, self.smooth_vy,
                                0.0, yaw=0.0)
            self.get_logger().info(
                f"STRAFE {h}/{v} | "
                f"err=({err_x:+.0f},{err_y:+.0f})px | "
                f"vx={self.smooth_vx:+.2f} vy={self.smooth_vy:+.2f}m/s | "
                f"dist={distance:.0f}px",
                throttle_duration_sec=0.3)
        else:
            # ── Yaw + forward ──────────────────────────────
            raw_yaw = max(-MAX_YAW, min(MAX_YAW, -KP_YAW * err_x))
            raw_vx  = max(-MAX_VEL_X, min(MAX_VEL_X, -KP_FWD * err_y))
            self.smooth_yaw = self._smooth(self.smooth_yaw, raw_yaw)
            self.smooth_vx  = self._smooth(self.smooth_vx,  raw_vx)
            self._send_velocity(self.smooth_vx, 0.0, 0.0,
                                yaw=self.smooth_yaw)
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
    # Pre-Arm Checks
    # ──────────────────────────────────────────────────────

    def _run_checks_then_fly(self):
        now      = time.time()
        items    = []
        failures = []
        warnings = []

        # [1] MAVROS Connection
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

        # [2] Not Already Armed
        if self.mavros_armed:
            s = FAIL; d = "Already ARMED"
            failures.append("Already armed")
        else:
            s = PASS; d = "Disarmed — safe to arm"
        items.append(("[2]  Not Already Armed ", s, d))

        # [3] System unlocked
        if self.system_locked:
            s = FAIL; d = "Locked — flip CH6 to ALT_HOLD"
            failures.append("System still locked")
        else:
            s = PASS; d = "Unlocked by ALT_HOLD flip"
        items.append(("[3]  System Unlocked   ", s, d))

        # [4] GPS Fix
        if not self.gps_received:
            s = UNKNOWN; d = "No GPS data yet"
            warnings.append("GPS not received")
        elif self.gps_fix_type == NavSatStatus.STATUS_NO_FIX:
            s = FAIL; d = "No fix — go outside"
            failures.append("No GPS fix")
        else:
            s = PASS; d = f"Fix type {self.gps_fix_type}"
        items.append(("[4]  GPS Fix           ", s, d))

        # [5] GPS Position Valid
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

        # [6] Satellites >= 10
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

        # [7] HDOP
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

        # [8] EKF Health
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

        # [9] On Ground
        if not self.alt_received:
            s = UNKNOWN; d = "No altitude data"
            warnings.append("Altitude unknown")
        elif self._is_in_air():
            s = FAIL; d = f"Alt {self.current_alt:.1f}m — in air!"
            failures.append(f"In air: {self.current_alt:.1f}m")
        else:
            s = PASS; d = f"Alt {self.current_alt:.2f}m on ground"
        items.append(("[9]  On Ground         ", s, d))

        # [10] Landed State
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

        # [11] Safe Flight Mode
        unsafe = {'AUTO', 'RTL', 'LAND', 'LOITER', 'GUIDED'}
        mode = self.mavros_mode
        if mode in unsafe:
            s = FAIL; d = f"'{mode}' unsafe for arming"
            failures.append(f"Unsafe mode: {mode}")
        else:
            s = PASS; d = f"Mode '{mode}' OK"
        items.append(("[11] Flight Mode       ", s, d))

        # [12] Battery
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

        # [13] Home Position
        if not self.home_set:
            s = WARN; d = "Not set (auto on GPS lock)"
            warnings.append("Home not set")
        else:
            s = PASS
            d = f"lat={self.home_lat:.6f} lon={self.home_lon:.6f}"
        items.append(("[13] Home Position     ", s, d))

        # [14] System Status
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

        # ── Print report ───────────────────────────────────
        n_fail  = len(failures)
        n_warn  = len(warnings)
        n_pass  = 14 - n_fail - n_warn
        cleared = (n_fail == 0)
        verdict = "PRE-ARM CLEARED" if cleared else "PRE-ARM BLOCKED"
        ICON = {PASS:"OK", FAIL:"XX", WARN:"!!", UNKNOWN:"??"}
        div = "-" * 56

        self.get_logger().info(div)
        self.get_logger().info(" LawinTracker PRE-ARM CHECK REPORT")
        self.get_logger().info(div)
        for label, status, detail in items:
            self.get_logger().info(
                f"  [{ICON.get(status,'??')}] {label}: {status}")
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
                "All checks passed — starting mission!")
            self._start_guided_sequence()
        else:
            self.get_logger().warn(
                f"BLOCKED — fix {n_fail} issue(s)!")
            self.stage = STAGE_ALT_HOLD

    # ──────────────────────────────────────────────────────
    # GUIDED → ARM → TAKEOFF Sequence
    # ──────────────────────────────────────────────────────

    def _start_guided_sequence(self):
        self.get_logger().info("Step 1/3: Setting GUIDED mode...")
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
                f"Armed! Step 3/3: Takeoff to {TAKEOFF_ALT}m...")
            self.stage = STAGE_TAKEOFF
            self._do_takeoff(TAKEOFF_ALT)
        else:
            self.get_logger().warn("Waiting for arm...")
            self._delay(1.0, self._check_armed_then_takeoff)

    def _on_takeoff_result(self, future):
        try:
            result = future.result()
            if result and result.success:
                self.get_logger().info(
                    f"Takeoff accepted! Climbing to {TAKEOFF_ALT}m...")
                self.stage = STAGE_CLIMB
            else:
                self.get_logger().error(
                    "Takeoff rejected! Retrying...")
                self._delay(
                    2.0, lambda: self._do_takeoff(TAKEOFF_ALT))
        except Exception as e:
            self.get_logger().error(f"Takeoff error: {e}")
            self._delay(2.0, lambda: self._do_takeoff(TAKEOFF_ALT))

    # ──────────────────────────────────────────────────────
    # Zigzag Mission
    # ──────────────────────────────────────────────────────

    def _start_zigzag(self):
        """Called once climb completes — snapshot home and build waypoints."""
        if self.gps_lat is None or self.gps_lon is None:
            self.get_logger().error(
                "No GPS at climb — aborting mission!")
            self._set_mode('RTL')
            self.stage = STAGE_RTL
            return

        self.home_gps  = (self.gps_lat, self.gps_lon)
        self.waypoints = self._make_zigzag(
            self.gps_lat, self.gps_lon)
        self.current_wp_idx = 0
        self.stage = STAGE_MISSION

        self.get_logger().info("=" * 52)
        self.get_logger().info(
            f"Home GPS: lat={self.home_gps[0]:.6f} "
            f"lon={self.home_gps[1]:.6f}")
        self.get_logger().info("Zigzag waypoints:")
        for name, wlat, wlon in self.waypoints:
            self.get_logger().info(
                f"  {name}: lat={wlat:.6f} lon={wlon:.6f}")
        self.get_logger().info("=" * 52)
        self.get_logger().info(
            "Flying to Waypoint A — mission started!")
        self.wp_start_time = time.time()

    def _mission_tick(self):
        """Called every 100ms during STAGE_MISSION."""
        if not self.waypoints or self.current_wp_idx >= len(
                self.waypoints):
            return

        name, wlat, wlon = self.waypoints[self.current_wp_idx]

        cmd = self._make_gps_cmd(wlat, wlon, TAKEOFF_ALT)
        self.gps_cmd_pub.publish(cmd)

        if self.gps_lat is None or self.gps_lon is None:
            return

        dist = self._haversine_m(
            self.gps_lat, self.gps_lon, wlat, wlon)

        self.get_logger().info(
            f"→ WP {name} | dist={dist:.2f}m | "
            f"alt={self.current_alt:.1f}m",
            throttle_duration_sec=1.0)

        if dist < WP_THRESHOLD:
            self.get_logger().info(f"✓ Reached Waypoint {name}!")
            self.current_wp_idx += 1

            if self.current_wp_idx >= len(self.waypoints):
                self.get_logger().info("=" * 52)
                self.get_logger().info(
                    "All waypoints complete — RTL!")
                self.get_logger().info("=" * 52)
                self.stage = STAGE_RTL
                self.arm_blocked = False
                self._set_mode('RTL')
            else:
                next_name = self.waypoints[
                    self.current_wp_idx][0]
                self.get_logger().info(
                    f"Flying to Waypoint {next_name}...")
                self.wp_start_time = time.time()
            return

        if time.time() - self.wp_start_time > WP_TIMEOUT:
            self.get_logger().warn(
                f"Timeout reaching WP {name} — skipping!")
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                self.get_logger().warn(
                    "Final waypoint skipped — RTL!")
                self.stage = STAGE_RTL
                self._set_mode('RTL')
            else:
                self.wp_start_time = time.time()

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
        if not self.mavros_connected:
            self.get_logger().error(
                f"{action}: MAVROS not connected!")
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
            self.get_logger().info(
                "Waiting for MAVROS...",
                throttle_duration_sec=10.0)
            return

        gps_ok = (self.gps_received and
                  self.gps_fix_type != NavSatStatus.STATUS_NO_FIX)
        bat_str = (
            f"{self.battery_voltage:.2f}V"
            if self.battery_received and
               self.battery_voltage is not None
            else "?V")
        landed = LANDED_NAMES.get(self.landed_state, "?")

        wp_info = ""
        if self.stage == STAGE_MISSION and self.waypoints:
            idx = min(self.current_wp_idx,
                      len(self.waypoints) - 1)
            wp_info = (f" WP:{self.waypoints[idx][0]}"
                       f"({self.current_wp_idx+1}/"
                       f"{len(self.waypoints)})")

        self.get_logger().info(
            f"Health | "
            f"{'LOCKED' if self.system_locked else 'UNLOCKED'} "
            f"Stage:{STAGE_NAMES.get(self.stage,'?')}{wp_info} "
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
            f"FCUMode:{self.mavros_mode} "
            f"TrackMode:{'FBRL' if self.fbrl_only else 'YAW'}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Test node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
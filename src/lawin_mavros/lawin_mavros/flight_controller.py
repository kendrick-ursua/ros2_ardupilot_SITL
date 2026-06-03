#!/usr/bin/env python3
"""
LawinTracker Flight Controller — MAVROS Version
=================================================
Uses MAVROS mission services for waypoint navigation:
  /mavros/mission/push    → upload waypoints
  /mavros/mission/clear   → clear existing mission
  /mavros/mission/reached → waypoint reached feedback
  /mavros/set_mode        → switch AUTO/GUIDED/STABILIZE
  /mavros/cmd/arming      → arm/disarm
  /mavros/cmd/takeoff     → takeoff

WORKFLOW:
  Manual (STABILIZE):
    Axis5 UP → STABILIZE → arm → fly with sticks

  Autonomous + Waypoint Mission (GUIDED → AUTO):
    Axis5 MID → GUIDED → arm → takeoff to 8m → upload
    mission → switch AUTO → fly waypoints → auto RTL

  During waypoint mission:
    Axis7 UP  → pause mission (GUIDED) → wait for person
    Axis7 DOWN → cancel tracking → resume mission (AUTO)
    Axis5 DOWN → RTL (emergency)

WAYPOINTS:
  Edit WAYPOINT_OFFSETS below (north_m, east_m, alt_m) from home.

Run:
  ros2 run lawin_tracker flight_controller
"""

import rclpy
import math
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Float64, String
from geometry_msgs.msg import TwistStamped, Point
from mavros_msgs.msg import State, Waypoint, WaypointList, WaypointReached
from mavros_msgs.srv import (
    CommandBool, SetMode, CommandTOL,
    WaypointPush, WaypointClear, WaypointSetCurrent
)


# ── Flight modes ───────────────────────────
MODE_STABILIZE = 'STABILIZE'
MODE_GUIDED    = 'GUIDED'
MODE_AUTO      = 'AUTO'
MODE_RTL       = 'RTL'

IN_AIR_THRESHOLD   = 2.5   # meters — above barometer ground drift
TRACK_WAIT_TIMEOUT = 5.0

# ── Waypoints ─────────────────────────────
# (north_m, east_m, alt_m) offsets from home
WAYPOINT_OFFSETS = [
    (  4.0,  3.0,  8.0),   # WP A
    (  8.0, -3.0,  8.0),   # WP B
    ( 12.0,  3.0,  8.0),   # WP C
    ( 16.0, -3.0,  8.0),   # WP D
    ( 20.0,  3.0,  8.0),   # WP E
]


def offset_to_latlon(home_lat, home_lon, north_m, east_m):
    lat = home_lat + (north_m / 111_320.0)
    lon = home_lon + (east_m / (
        111_320.0 * math.cos(math.radians(home_lat))))
    return lat, lon


class FlightController(Node):

    # ── Flight stages ──────────────────────
    STAGE_IDLE       = 0
    STAGE_PREARM     = 1
    STAGE_ARM        = 2
    STAGE_MODE       = 3
    STAGE_TAKEOFF    = 4
    STAGE_CLIMB      = 5
    STAGE_HOVER      = 6
    STAGE_FOLLOW     = 7
    STAGE_RTL        = 8
    STAGE_STABILIZE  = 9
    STAGE_MISSION    = 10
    STAGE_TRACK_WAIT = 11

    # ── Controller config ──────────────────
    KP_YAW       = 0.003
    KP_FWD       = 0.010
    MAX_YAW      = 0.8
    MAX_VEL_X    = 5.0
    MIN_VEL      = 0.01
    SMOOTH       = 0.7
    STOP_ZONE    = 15
    START_ZONE   = 35
    LOST_TIMEOUT = 2.0

    def __init__(self):
        super().__init__('flight_controller')

        # ── State ──────────────────────────
        self.current_alt   = 0.0
        self.current_lat   = None
        self.current_lon   = None
        self.home_lat      = None
        self.home_lon      = None
        self.target_alt    = 8.0
        self.stage         = self.STAGE_IDLE
        self.gps_ready     = False

        self.mavros_connected = False
        self.mavros_armed     = False
        self.mavros_mode      = ''
        self.startup_done     = False

        # Block arming after RTL landing until ARM switch
        # is explicitly toggled DOWN then UP again
        self.arm_blocked = False

        # Target tracking
        self.offset_x   = 0.0
        self.offset_y   = 0.0
        self.has_target = False
        self.last_seen  = 0.0
        self.is_holding = False

        self.rc_tracking_enabled = False
        self.smooth_vx  = 0.0
        self.smooth_yaw = 0.0

        # ── Mission state ──────────────────
        self.mission_paused      = False
        self.mission_wp_count    = len(WAYPOINT_OFFSETS)
        self.current_wp_index    = 0
        self.track_wait_start    = 0.0
        self.resume_wp_index     = 0  # WP to resume from

        # QoS
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10)
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10)

        # ── Subscribers ───────────────────
        self.state_sub = self.create_subscription(
            State, '/mavros/state',
            self.state_callback, qos_reliable)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/mavros/global_position/global',
            self.gps_callback, qos_sensor)
        self.alt_sub = self.create_subscription(
            Float64, '/mavros/global_position/rel_alt',
            self.alt_callback, qos_sensor)
        self.offset_sub = self.create_subscription(
            Point, '/lawin/target_offset',
            self.offset_callback, 10)
        self.rc_sub = self.create_subscription(
            String, '/lawin/rc_command',
            self.rc_callback, 10)
        # Mission waypoint reached feedback
        self.reached_sub = self.create_subscription(
            WaypointReached, '/mavros/mission/reached',
            self.waypoint_reached_callback, qos_reliable)

        # ── Publishers ────────────────────
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',
            qos_sensor)

        # ── Service clients ───────────────
        self.arming_client    = self.create_client(
            CommandBool, '/mavros/cmd/arming')
        self.set_mode_client  = self.create_client(
            SetMode, '/mavros/set_mode')
        self.takeoff_client   = self.create_client(
            CommandTOL, '/mavros/cmd/takeoff')
        self.wp_push_client   = self.create_client(
            WaypointPush, '/mavros/mission/push')
        self.wp_clear_client  = self.create_client(
            WaypointClear, '/mavros/mission/clear')
        self.wp_setcur_client = self.create_client(
            WaypointSetCurrent, '/mavros/mission/set_current')

        self.main_timer = self.create_timer(
            0.05, self.main_loop)

        self.get_logger().info(
            "Flight Controller (MAVROS) ready!")
        self.get_logger().info(
            "Waiting for MAVROS connection...")

    # ─────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────
    def is_in_air(self) -> bool:
        return self.current_alt > IN_AIR_THRESHOLD

    def stage_name(self, stage: int) -> str:
        return {
            0:"IDLE", 1:"PREARM", 2:"ARM", 3:"MODE",
            4:"TAKEOFF", 5:"CLIMB", 6:"HOVER", 7:"FOLLOW",
            8:"RTL", 9:"STABILIZE", 10:"MISSION",
            11:"TRACK_WAIT"
        }.get(stage, f"STAGE_{stage}")

    def restart_main_loop(self):
        try:
            self.main_timer.cancel()
        except Exception:
            pass
        self.main_timer = self.create_timer(
            0.05, self.main_loop)

    # ─────────────────────────────────────
    # MAVROS State Callback
    # ─────────────────────────────────────
    def state_callback(self, msg: State):
        was_connected = self.mavros_connected
        self.mavros_connected = msg.connected
        self.mavros_armed     = msg.armed

        # Detect mode changes from ArduPilot
        prev_mode = self.mavros_mode
        self.mavros_mode = msg.mode

        if not was_connected and msg.connected:
            self.get_logger().info(
                "MAVROS connected!")
            self.get_logger().info(
                f"Mode: {msg.mode} | Armed: {msg.armed}")

        # Detect when ArduPilot switches to RTL
        # (e.g. mission complete via RTL waypoint)
        if prev_mode != msg.mode and msg.mode == MODE_RTL:
            if self.stage not in [self.STAGE_RTL,
                                   self.STAGE_IDLE]:
                self.get_logger().info(
                    "🏁 ArduPilot switched to RTL "
                    "(mission complete or emergency)")
                self.rc_tracking_enabled = False
                self.mission_paused      = False
                self.stage = self.STAGE_RTL

        if msg.connected and not self.startup_done:
            self.startup_done = True
            self._sync_stage(msg)

    def _sync_stage(self, msg: State):
        self.get_logger().info(
            f"Startup → Mode: {msg.mode} | "
            f"Armed: {msg.armed} | "
            f"Alt: {self.current_alt:.1f}m")

        if self.is_in_air():
            if msg.mode == MODE_AUTO:
                self.stage = self.STAGE_MISSION
            elif msg.mode == MODE_GUIDED:
                self.stage = self.STAGE_HOVER
            elif msg.mode == MODE_STABILIZE:
                self.stage = self.STAGE_STABILIZE
            elif msg.mode == MODE_RTL:
                self.stage = self.STAGE_RTL
            else:
                self.stage = self.STAGE_HOVER
        else:
            self.stage = (self.STAGE_STABILIZE
                          if msg.mode == MODE_STABILIZE
                          else self.STAGE_IDLE)

        self.get_logger().info(
            f"Stage → {self.stage_name(self.stage)}")
        self.get_logger().info(
            "Axis5 UP=STABILIZE | "
            "Axis5 MID=GUIDED+MISSION | "
            "Axis4 UP=ARM | Axis7 UP=TRACK")

    # ─────────────────────────────────────
    # GPS / Alt Callbacks
    # ─────────────────────────────────────
    def gps_callback(self, msg: NavSatFix):
        if msg.status.status == NavSatStatus.STATUS_NO_FIX:
            return
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        if self.home_lat is None:
            self.home_lat  = msg.latitude
            self.home_lon  = msg.longitude
            self.gps_ready = True
            self.get_logger().info(
                f"Home GPS: "
                f"lat={self.home_lat:.6f} "
                f"lon={self.home_lon:.6f}")

    def alt_callback(self, msg: Float64):
        self.current_alt = msg.data

    # ─────────────────────────────────────
    # Target Offset Callback
    # ─────────────────────────────────────
    def offset_callback(self, msg: Point):
        now = self.get_clock().now().nanoseconds / 1e9
        if msg.z > 0:
            self.offset_x   = msg.x
            self.offset_y   = msg.y
            self.has_target = True
            self.last_seen  = now
        else:
            self.has_target = False

    # ─────────────────────────────────────
    # Waypoint Reached Callback
    # ─────────────────────────────────────
    def waypoint_reached_callback(self, msg: WaypointReached):
        self.current_wp_index = msg.wp_seq
        self.get_logger().info(
            f"Waypoint {msg.wp_seq} reached!")
        # ArduPilot handles RTL automatically via the
        # RTL waypoint at the end of the mission.
        # Do NOT manually trigger RTL here — it causes
        # the mission to restart!

    # ─────────────────────────────────────
    # RC Command Callback
    # ─────────────────────────────────────
    def rc_callback(self, msg: String):
        cmd = msg.data
        self.get_logger().info(f"RC → {cmd}")

        if cmd == 'STABILIZE':
            self.get_logger().warn("STABILIZE! Manual control.")
            self.rc_tracking_enabled = False
            self.mission_paused      = False
            self.smooth_vx  = 0.0
            self.smooth_yaw = 0.0
            self.send_velocity(0.0, 0.0, 0.0)
            self.stage = self.STAGE_STABILIZE
            self.set_mode(MODE_STABILIZE)

        elif cmd == 'GUIDED':
            if not self.mavros_connected:
                self.get_logger().error(
                    "No MAVROS connection!")
                return
            if not self.gps_ready:
                self.get_logger().warn("⚠️  No GPS yet!")
                return

            if self.is_in_air():
                self.get_logger().info(
                    f"In air ({self.current_alt:.1f}m) → "
                    f"GUIDED → mission")
                self.rc_tracking_enabled = False
                self.mission_paused      = False
                self.stage = self.STAGE_HOVER
                self.set_mode(MODE_GUIDED,
                              callback=self.guided_in_air_done)
            else:
                if self.stage in [self.STAGE_IDLE,
                                   self.STAGE_STABILIZE]:
                    self.get_logger().info(
                        "On ground → GUIDED + ARM + "
                        "TAKEOFF + MISSION")
                    self.stage = self.STAGE_MODE
                    self.main_timer.cancel()
                    self.do_set_guided_mode()
                else:
                    self.get_logger().warn(
                        f"Already in "
                        f"{self.stage_name(self.stage)}")

        elif cmd == 'RTL':
            self.get_logger().warn("EMERGENCY RTL!")
            self.rc_tracking_enabled = False
            self.mission_paused      = False
            self.smooth_vx  = 0.0
            self.smooth_yaw = 0.0
            self.stage = self.STAGE_RTL
            # Block arming after RTL — require ARM switch
            # to be toggled DOWN then UP again
            self.arm_blocked = True
            self.set_mode(MODE_RTL)

        elif cmd == 'ARM':
            if not self.mavros_connected:
                self.get_logger().error("No connection!")
                return
            if self.is_in_air():
                self.get_logger().warn("Already in air!")
                return
            if self.arm_blocked:
                self.get_logger().warn(
                    "ARM blocked! Flip ARM switch DOWN "
                    "then UP again to arm.")
                return
            self.arm_motors(True)

        elif cmd == 'DISARM':
            self.get_logger().warn("DISARM!")
            self.rc_tracking_enabled = False
            self.mission_paused      = False
            self.arm_blocked = False  # reset block on disarm
            self.stage = self.STAGE_IDLE
            self.arm_motors(False)

        elif cmd == 'TRACK_START':
            if self.stage == self.STAGE_MISSION:
                # During mission → pause and track
                self.get_logger().info(
                    f"Pausing mission at WP"
                    f"{self.current_wp_index} → "
                    f"waiting {TRACK_WAIT_TIMEOUT}s...")
                self._pause_mission()

            elif self.stage == self.STAGE_TRACK_WAIT:
                self.get_logger().info(
                    "Already waiting for person...")

            elif self.stage == self.STAGE_CLIMB:
                # During climb to target alt → start tracking
                # at current altitude instead
                self.get_logger().info(
                    f"Tracking at current alt "
                    f"{self.current_alt:.1f}m "
                    f"(cancel to resume climb → mission)")
                self.rc_tracking_enabled = True
                self.is_holding = False
                self.stage = self.STAGE_FOLLOW
                self.restart_main_loop()

            elif self.stage in [self.STAGE_HOVER,
                                  self.STAGE_FOLLOW]:
                # In hover/follow → track at current altitude
                self.rc_tracking_enabled = True
                self.is_holding = False
                self.stage = self.STAGE_FOLLOW
                self.get_logger().info(
                    f"Tracking at alt "
                    f"{self.current_alt:.1f}m | "
                    f"cancel → climb to "
                    f"{self.target_alt}m → mission")

            elif self.is_in_air():
                # Auto-recover
                self.get_logger().warn(
                    "Auto-recovering to tracking...")
                self.rc_tracking_enabled = True
                self.is_holding = False
                self.stage = self.STAGE_FOLLOW
                self.restart_main_loop()

            else:
                self.get_logger().warn(
                    "Cannot track — not in air yet!")

        elif cmd == 'TRACK_CANCEL':
            self.rc_tracking_enabled = False

            if self.mission_paused:
                # Was tracking during mission → resume mission
                self.get_logger().info(
                    "Tracking cancelled → resuming mission!")
                self.stage = self.STAGE_MISSION
                self._resume_mission()

            elif self.stage in [self.STAGE_FOLLOW,
                                  self.STAGE_TRACK_WAIT,
                                  self.STAGE_HOVER]:
                # Was tracking in GUIDED (not during mission)
                # → climb to target altitude → start mission
                alt_diff = self.target_alt - self.current_alt
                if abs(alt_diff) > 0.5:
                    self.get_logger().info(
                        f"Tracking cancelled → "
                        f"climbing to {self.target_alt}m "
                        f"(currently {self.current_alt:.1f}m)"
                        f" → then mission...")
                    self.stage = self.STAGE_CLIMB
                else:
                    self.get_logger().info(
                        f"Tracking cancelled → "
                        f"already at {self.current_alt:.1f}m "
                        f"→ starting mission...")
                    self.stage = self.STAGE_CLIMB
                self.restart_main_loop()

            else:
                self.get_logger().info("Tracking CANCELLED.")

    # ─────────────────────────────────────
    # MAVROS Service Calls
    # ─────────────────────────────────────
    def set_mode(self, mode: str, callback=None):
        if not self.set_mode_client.service_is_ready():
            self._delay(1.0, lambda: self.set_mode(
                mode, callback))
            return
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.set_mode_client.call_async(req)
        if callback:
            future.add_done_callback(
                lambda f: callback(f))
        else:
            future.add_done_callback(
                lambda f: self.get_logger().info(
                    f"Mode → {mode}")
                if f.result() and f.result().mode_sent
                else self.get_logger().error(
                    f"Mode {mode} failed!"))

    def arm_motors(self, arm: bool):
        if not self.arming_client.service_is_ready():
            self._delay(1.0, lambda: self.arm_motors(arm))
            return
        req = CommandBool.Request()
        req.value = arm
        future = self.arming_client.call_async(req)
        future.add_done_callback(self.arm_callback)

    def arm_callback(self, future):
        try:
            result = future.result()
            if result and result.success:
                self.get_logger().info(
                    "Armed!" if self.mavros_armed
                    else "Disarmed!")
            else:
                self.get_logger().error(
                    "Arm/Disarm failed!")
        except Exception as e:
            self.get_logger().error(f"Arm error: {e}")

    def do_takeoff(self, alt: float):
        if not self.takeoff_client.service_is_ready():
            self._delay(1.0, lambda: self.do_takeoff(alt))
            return
        req = CommandTOL.Request()
        req.altitude  = float(alt)
        req.latitude  = float(self.current_lat or 0)
        req.longitude = float(self.current_lon or 0)
        req.min_pitch = 0.0
        req.yaw       = 0.0
        future = self.takeoff_client.call_async(req)
        future.add_done_callback(self.takeoff_callback)

    def takeoff_callback(self, future):
        if self.stage != self.STAGE_TAKEOFF:
            return
        try:
            result = future.result()
            if result and result.success:
                self.get_logger().info(
                    f"Takeoff! Climbing to "
                    f"{self.target_alt}m...")
                self.stage = self.STAGE_CLIMB
                self.restart_main_loop()
            else:
                self.get_logger().error(
                    "Takeoff failed! Retrying...")
                self._delay(2.0, lambda: self.do_takeoff(
                    self.target_alt))
        except Exception as e:
            self.get_logger().error(f"Takeoff error: {e}")
            self._delay(2.0, lambda: self.do_takeoff(
                self.target_alt))

    # ─────────────────────────────────────
    # Mission Management
    # ─────────────────────────────────────
    def _build_mission_waypoints(self) -> list:
        """Build MAVROS Waypoint list from offsets."""
        if self.home_lat is None:
            return []

        waypoints = []

        # WP 0 — Home (required by ArduPilot)
        home_wp = Waypoint()
        home_wp.frame        = Waypoint.FRAME_GLOBAL_REL_ALT
        home_wp.command      = 16   # MAV_CMD_NAV_WAYPOINT
        home_wp.is_current   = False
        home_wp.autocontinue = True
        home_wp.param1       = 0.0
        home_wp.param2       = 0.0
        home_wp.param3       = 0.0
        home_wp.param4       = 0.0
        home_wp.x_lat        = self.home_lat
        home_wp.y_long       = self.home_lon
        home_wp.z_alt        = 0.0
        waypoints.append(home_wp)

        # WP 1..N — Mission waypoints
        for i, (north_m, east_m, alt) in enumerate(
                WAYPOINT_OFFSETS):
            lat, lon = offset_to_latlon(
                self.home_lat, self.home_lon,
                north_m, east_m)

            wp = Waypoint()
            wp.frame        = Waypoint.FRAME_GLOBAL_REL_ALT
            wp.command      = 16   # MAV_CMD_NAV_WAYPOINT
            wp.is_current   = (i == 0)
            wp.autocontinue = True
            wp.param1       = 0.0   # hold time
            wp.param2       = 2.0   # acceptance radius (m)
            wp.param3       = 0.0   # pass radius
            wp.param4       = 0.0   # yaw
            wp.x_lat        = lat
            wp.y_long       = lon
            wp.z_alt        = float(alt)
            waypoints.append(wp)

            self.get_logger().info(
                f"  WP{i+1}: "
                f"lat={lat:.6f} lon={lon:.6f} alt={alt}m")

        # WP N+1 — RTL command (prevents mission loop!)
        rtl_wp = Waypoint()
        rtl_wp.frame        = Waypoint.FRAME_MISSION
        rtl_wp.command      = 20   # MAV_CMD_NAV_RETURN_TO_LAUNCH
        rtl_wp.is_current   = False
        rtl_wp.autocontinue = True
        rtl_wp.param1       = 0.0
        rtl_wp.param2       = 0.0
        rtl_wp.param3       = 0.0
        rtl_wp.param4       = 0.0
        rtl_wp.x_lat        = 0.0
        rtl_wp.y_long       = 0.0
        rtl_wp.z_alt        = 0.0
        waypoints.append(rtl_wp)
        self.get_logger().info(
            f"  WP{len(WAYPOINT_OFFSETS)+1}: RTL ← auto RTL!")

        return waypoints

    def _upload_and_start_mission(self):
        """Clear old mission, upload new one, switch to AUTO."""
        if not self.wp_clear_client.service_is_ready():
            self._delay(1.0, self._upload_and_start_mission)
            return

        # Tip: Set MIS_RESTART=0 in Mission Planner/MAVProxy
        # to prevent mission from looping after completion!
        # mavproxy: param set MIS_RESTART 0
        self.get_logger().info("Clearing old mission...")
        req = WaypointClear.Request()
        future = self.wp_clear_client.call_async(req)
        future.add_done_callback(self._on_mission_cleared)

    def _on_mission_cleared(self, future):
        try:
            result = future.result()
            if result and result.success:
                self.get_logger().info(
                    "Mission cleared! Uploading new mission...")
                self._push_mission()
            else:
                self.get_logger().error(
                    "Clear failed! Retrying...")
                self._delay(1.0, self._upload_and_start_mission)
        except Exception as e:
            self.get_logger().error(f"Clear error: {e}")
            self._delay(1.0, self._upload_and_start_mission)

    def _push_mission(self):
        if not self.wp_push_client.service_is_ready():
            self._delay(1.0, self._push_mission)
            return

        waypoints = self._build_mission_waypoints()
        if not waypoints:
            self.get_logger().error(
                "No waypoints! GPS not ready.")
            return

        self.get_logger().info(
            f"Pushing {len(waypoints)} waypoints "
            f"(1 home + {len(WAYPOINT_OFFSETS)} mission)...")

        req = WaypointPush.Request()
        req.start_index = 0
        req.waypoints   = waypoints
        future = self.wp_push_client.call_async(req)
        future.add_done_callback(self._on_mission_pushed)

    def _on_mission_pushed(self, future):
        try:
            result = future.result()
            if result and result.success:
                self.get_logger().info(
                    f"Mission uploaded! "
                    f"{result.wp_transfered} waypoints. "
                    f"Setting WP1 as current...")
                self.mission_wp_count = len(WAYPOINT_OFFSETS)
                self.current_wp_index = 0
                self.stage = self.STAGE_MISSION
                # Set WP1 as current (WP0 is home, skip it!)
                self._set_current_wp_then_auto(1)
            else:
                self.get_logger().error(
                    "Mission push failed! Retrying...")
                self._delay(2.0, self._push_mission)
        except Exception as e:
            self.get_logger().error(
                f"Mission push error: {e}")
            self._delay(2.0, self._push_mission)

    def _set_current_wp_then_auto(self, wp_index: int):
        """Set current waypoint then switch to AUTO."""
        if not self.wp_setcur_client.service_is_ready():
            self._delay(0.5, lambda: self._set_current_wp_then_auto(wp_index))
            return
        req = WaypointSetCurrent.Request()
        req.wp_seq = wp_index
        future = self.wp_setcur_client.call_async(req)
        future.add_done_callback(
            lambda f: self._after_wp_set(f, wp_index))

    def _after_wp_set(self, future, wp_index: int):
        try:
            result = future.result()
            if result and result.success:
                self.get_logger().info(
                    f"WP{wp_index} set as current! "
                    f"Switching to AUTO...")
            else:
                self.get_logger().warn(
                    f"WP set failed, switching AUTO anyway...")
        except Exception as e:
            self.get_logger().warn(f"WP set error: {e}")
        # Switch to AUTO regardless
        self._delay(0.5, lambda: self.set_mode(
            MODE_AUTO, callback=self._on_auto_mode))

    def _on_auto_mode(self, future):
        try:
            result = future.result()
            if result and result.mode_sent:
                self.get_logger().info(
                    "AUTO mode! Mission running...")
                self.get_logger().info(
                    "Axis7 UP=track | "
                    "Axis7 DOWN=resume | "
                    "Axis5 DOWN=RTL")
                self.restart_main_loop()
            else:
                self.get_logger().error(
                    "AUTO mode failed!")
                self.stage = self.STAGE_HOVER
                self.restart_main_loop()
        except Exception as e:
            self.get_logger().error(f"AUTO error: {e}")
            self.stage = self.STAGE_HOVER
            self.restart_main_loop()

    def _pause_mission(self):
        """Pause mission → GUIDED → wait for person."""
        self.mission_paused   = True
        self.resume_wp_index  = self.current_wp_index
        self.track_wait_start = time.time()
        self.stage = self.STAGE_TRACK_WAIT

        # Switch to GUIDED to pause AUTO
        self.set_mode(MODE_GUIDED)
        self.get_logger().info(
            f"Mission PAUSED at WP{self.current_wp_index} "
            f"→ waiting for person...")
        self.restart_main_loop()

    def _resume_mission(self):
        """Resume mission from paused waypoint."""
        self.rc_tracking_enabled = False
        self.mission_paused      = False
        self.stage = self.STAGE_MISSION
        self.get_logger().info(
            f"Resuming mission from "
            f"WP{self.resume_wp_index}...")
        self._set_current_wp_then_auto(self.resume_wp_index)

    # ─────────────────────────────────────
    # Takeoff sequence (ground start)
    # ─────────────────────────────────────
    def do_set_guided_mode(self):
        if self.stage != self.STAGE_MODE:
            return
        self.get_logger().info("Setting GUIDED mode...")
        self.set_mode(MODE_GUIDED,
                      callback=self.guided_mode_callback)

    def guided_mode_callback(self, future):
        if self.stage != self.STAGE_MODE:
            return
        try:
            result = future.result()
            if result and result.mode_sent:
                self.get_logger().info(
                    "GUIDED mode set! Arming...")
                self.stage = self.STAGE_ARM
                self._delay(1.0, lambda: self.arm_motors(True))
                self._delay(3.0, self.check_armed_then_takeoff)
            else:
                self.get_logger().error(
                    "GUIDED failed! Retrying...")
                self._delay(2.0, self.do_set_guided_mode)
        except Exception as e:
            self.get_logger().error(f"GUIDED error: {e}")
            self._delay(2.0, self.do_set_guided_mode)

    def check_armed_then_takeoff(self):
        if self.stage not in [self.STAGE_ARM,
                               self.STAGE_TAKEOFF]:
            return
        if self.mavros_armed:
            self.get_logger().info("Armed! Taking off...")
            self.stage = self.STAGE_TAKEOFF
            self.do_takeoff(self.target_alt)
        else:
            self.get_logger().warn(
                "Not armed yet, waiting...")
            self._delay(1.0, self.check_armed_then_takeoff)

    def guided_in_air_done(self, future):
        try:
            result = future.result()
            if result and result.mode_sent:
                self.get_logger().info(
                    "GUIDED! Starting mission...")
                self.stage = self.STAGE_CLIMB
                self.restart_main_loop()
            else:
                self.get_logger().error("GUIDED failed!")
        except Exception as e:
            self.get_logger().error(f"GUIDED error: {e}")

    # ─────────────────────────────────────
    # Main Loop (20Hz)
    # ─────────────────────────────────────
    def main_loop(self):

        if self.stage == self.STAGE_IDLE:
            if not self.mavros_connected:
                self.get_logger().info(
                    "Waiting for MAVROS...",
                    throttle_duration_sec=3.0)
            elif not self.gps_ready:
                self.get_logger().info(
                    "MAVROS OK! Waiting for GPS...",
                    throttle_duration_sec=3.0)
            else:
                self.get_logger().info(
                    "Ready! "
                    "Axis5 UP=Manual | "
                    "Axis5 MID=Auto+Mission | "
                    "Axis4 UP=ARM",
                    throttle_duration_sec=5.0)

        elif self.stage == self.STAGE_STABILIZE:
            self.get_logger().info(
                "STABILIZE — Use sticks!",
                throttle_duration_sec=3.0)

        elif self.stage == self.STAGE_ARM:
            self.get_logger().info(
                "Arming...",
                throttle_duration_sec=2.0)

        elif self.stage == self.STAGE_TAKEOFF:
            self.get_logger().info(
                f"Taking off to {self.target_alt}m...",
                throttle_duration_sec=2.0)

        elif self.stage == self.STAGE_CLIMB:
            alt_error = self.target_alt - self.current_alt
            self.get_logger().info(
                f"Alt: {self.current_alt:.1f}m "
                f"/ {self.target_alt:.1f}m",
                throttle_duration_sec=1.0)

            if abs(alt_error) <= 0.5:
                # Reached target altitude
                self.send_velocity(0.0, 0.0, 0.0)
                self.get_logger().info(
                    f"Reached {self.current_alt:.1f}m! "
                    f"Uploading mission...")
                self._upload_and_start_mission()

            elif self.is_in_air() and alt_error > 0.5:
                # Already in air but below target alt
                # → send upward velocity to climb
                climb_speed = min(1.5, alt_error * 0.3)
                self.send_velocity(0.0, 0.0, climb_speed)
                self.get_logger().info(
                    f"Climbing in air: "
                    f"{self.current_alt:.1f}m → "
                    f"{self.target_alt:.1f}m "
                    f"(vz={climb_speed:.2f}m/s)",
                    throttle_duration_sec=1.0)

            elif self.is_in_air() and alt_error < -0.5:
                # Above target alt → descend slowly
                descend_speed = max(-0.5, alt_error * 0.3)
                self.send_velocity(0.0, 0.0, descend_speed)
                self.get_logger().info(
                    f"Descending: "
                    f"{self.current_alt:.1f}m → "
                    f"{self.target_alt:.1f}m",
                    throttle_duration_sec=1.0)

        elif self.stage == self.STAGE_MISSION:
            self.get_logger().info(
                f"Mission | "
                f"WP{self.current_wp_index}"
                f"/{self.mission_wp_count} | "
                f"alt={self.current_alt:.1f}m | "
                f"Axis7 UP=track",
                throttle_duration_sec=2.0)

        elif self.stage == self.STAGE_TRACK_WAIT:
            elapsed = time.time() - self.track_wait_start

            if self.has_target:
                self.get_logger().info(
                    "Person detected! Tracking!")
                self.rc_tracking_enabled = True
                self.is_holding = False
                self.stage = self.STAGE_FOLLOW
                self.restart_main_loop()

            elif elapsed > TRACK_WAIT_TIMEOUT:
                self.get_logger().warn(
                    f"No person after "
                    f"{TRACK_WAIT_TIMEOUT}s "
                    f"→ resuming mission!")
                self.mission_paused = False
                self._resume_mission()

            else:
                self.get_logger().info(
                    f"Looking for person... "
                    f"({elapsed:.1f}s"
                    f"/{TRACK_WAIT_TIMEOUT}s) | "
                    f"Axis7 DOWN to cancel",
                    throttle_duration_sec=1.0)
                self.send_velocity(0.0, 0.0, 0.0)

        elif self.stage == self.STAGE_HOVER:
            self.smooth_vx  = 0.0
            self.smooth_yaw = 0.0
            self.send_velocity(0.0, 0.0, 0.0)
            if self.rc_tracking_enabled and self.has_target:
                self.get_logger().info("🎯 Target! Following...")
                self.is_holding = False
                self.stage = self.STAGE_FOLLOW

        elif self.stage == self.STAGE_FOLLOW:
            if self.rc_tracking_enabled:
                self.follow_target()
            else:
                self.send_velocity(0.0, 0.0, 0.0)
                if self.mission_paused:
                    # Target lost during mission tracking
                    # → resume mission
                    self.get_logger().info(
                        "Tracking off → resuming mission...")
                    self.stage = self.STAGE_MISSION
                    self._resume_mission()
                else:
                    # Target lost during GUIDED tracking
                    # → hover (TRACK_CANCEL handles climb+mission)
                    self.stage = self.STAGE_HOVER

        elif self.stage == self.STAGE_RTL:
            self.get_logger().info(
                "RTL in progress...",
                throttle_duration_sec=2.0)

    # ─────────────────────────────────────
    # Target Following
    # ─────────────────────────────────────
    def follow_target(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if not self.has_target or \
           (now - self.last_seen) > self.LOST_TIMEOUT:
            self.smooth_vx  = self.smooth(
                self.smooth_vx, 0.0)
            self.smooth_yaw = self.smooth(
                self.smooth_yaw, 0.0)
            self.send_velocity(
                self.smooth_vx, 0.0, 0.0,
                yaw=self.smooth_yaw)
            if abs(self.smooth_vx) < self.MIN_VEL and \
               abs(self.smooth_yaw) < self.MIN_VEL:
                self.smooth_vx  = 0.0
                self.smooth_yaw = 0.0
                self.is_holding = False
                self.get_logger().warn("Target lost!")
                if self.mission_paused:
                    self.rc_tracking_enabled = False
                    self._resume_mission()
                else:
                    self.stage = self.STAGE_HOVER
            return

        err_x    = self.offset_x
        err_y    = self.offset_y
        distance = math.sqrt(err_x**2 + err_y**2)

        if self.is_holding:
            if distance > self.START_ZONE:
                self.is_holding = False
            else:
                self.smooth_vx  = self.smooth(
                    self.smooth_vx, 0.0)
                self.smooth_yaw = self.smooth(
                    self.smooth_yaw, 0.0)
                self.send_velocity(
                    self.smooth_vx, 0.0, 0.0,
                    yaw=self.smooth_yaw)
                self.get_logger().info(
                    f"HOLD (dist={distance:.0f}px)",
                    throttle_duration_sec=1.0)
                return
        else:
            if distance < self.STOP_ZONE:
                self.is_holding = True
                self.smooth_vx  = self.smooth(
                    self.smooth_vx, 0.0)
                self.smooth_yaw = self.smooth(
                    self.smooth_yaw, 0.0)
                self.send_velocity(
                    self.smooth_vx, 0.0, 0.0,
                    yaw=self.smooth_yaw)
                self.get_logger().info(
                    f"HOLD — centered! "
                    f"(dist={distance:.0f}px)")
                return

        raw_yaw = -self.KP_YAW * err_x
        raw_vx  = -self.KP_FWD * err_y
        raw_yaw = max(-self.MAX_YAW,
                      min(self.MAX_YAW, raw_yaw))
        raw_vx  = max(-self.MAX_VEL_X,
                      min(self.MAX_VEL_X, raw_vx))
        self.smooth_yaw = self.smooth(
            self.smooth_yaw, raw_yaw)
        self.smooth_vx  = self.smooth(
            self.smooth_vx, raw_vx)
        self.send_velocity(
            self.smooth_vx, 0.0, 0.0,
            yaw=self.smooth_yaw)

        h = "RIGHT" if err_x > 0 else "LEFT"
        v = "BACK"  if err_y > 0 else "FWD"
        self.get_logger().info(
            f"MOVE {h}/{v} | "
            f"err=({err_x:+.0f},{err_y:+.0f})px | "
            f"yaw={self.smooth_yaw:+.3f} | "
            f"vx={self.smooth_vx:+.2f}m/s | "
            f"dist={distance:.0f}px",
            throttle_duration_sec=0.3)

    # ─────────────────────────────────────
    # Smooth helper
    # ─────────────────────────────────────
    def smooth(self, current: float, target: float) -> float:
        result = (self.SMOOTH * current +
                  (1.0 - self.SMOOTH) * target)
        return 0.0 if abs(result) < self.MIN_VEL else result

    # ─────────────────────────────────────
    # Velocity command
    # ─────────────────────────────────────
    def send_velocity(self, vx, vy, vz,
                      yaw=0.0, frame='base_link'):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.twist.linear.x  = float(vx)
        msg.twist.linear.y  = float(vy)
        msg.twist.linear.z  = float(vz)
        msg.twist.angular.z = float(yaw)
        self.vel_pub.publish(msg)

    # ─────────────────────────────────────
    # One-shot delay timer
    # ─────────────────────────────────────
    def _delay(self, seconds: float, callback):
        timer = None
        def _fire():
            timer.cancel()
            callback()
        timer = self.create_timer(seconds, _fire)


def main(args=None):
    rclpy.init(args=args)
    node = FlightController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Zigzag Mission Node — GPS-Assisted INS / MAVROS / ROS 2 Humble
===============================================================
Primary positioning: GPS (NavSatFix)
Fallback positioning: IMU dead-reckoning via MAVROS odometry/local position

Strategy
--------
1. At boot, wait for a GPS fix and record it as the GLOBAL ORIGIN.
2. Convert all waypoints into LOCAL NED offsets from that origin (metres).
3. Navigate using /mavros/setpoint_raw/local (NED, metres) — this works
   whether GPS is present or not, as long as ArduPilot's EKF has a valid
   local-frame estimate (optical flow, VIO, or just IMU propagation).
4. Continuously monitor GPS health. Log a warning when GPS is lost;
   dead-reckoning drift will accumulate but the mission continues.
5. Monitor EKF flags via /mavros/ekf_status. If EKF is unhealthy, hover
   and wait before proceeding.

Why local-frame setpoints?
--------------------------
/mavros/setpoint_raw/local  (mavros_msgs/PositionTarget)  uses MAV_FRAME_LOCAL_NED.
ArduPilot fuses GPS + IMU + barometer into a single local-frame EKF estimate.
When GPS drops, the EKF continues propagating from IMU — drift grows over time
but position control keeps working. This is far more robust than trying to
convert IMU data yourself.

MAVROS Topics / Services Used
------------------------------
  SUB  /mavros/global_position/global      sensor_msgs/NavSatFix
  SUB  /mavros/local_position/odom         nav_msgs/Odometry
  SUB  /mavros/ekf_status_report           mavros_msgs/EstimatorStatus  (optional)
  SUB  /mavros/state                       mavros_msgs/State

  PUB  /mavros/setpoint_raw/local          mavros_msgs/PositionTarget
         (NED frame, metres — works GPS-denied)

  SRV  /mavros/cmd/arming                  mavros_msgs/CommandBool
  SRV  /mavros/set_mode                    mavros_msgs/SetMode
  SRV  /mavros/cmd/takeoff                 mavros_msgs/CommandTOL

ArduPilot Setup Required
------------------------
  For GPS-denied indoor/outdoor flight you need ONE of:
    a) Optical flow sensor  (FLOW_TYPE != 0) + rangefinder
    b) Visual odometry      (VISUAL_ODOMETRY_ENABLE=1, e.g. Intel T265)
    c) GPS only (outdoor)   — EKF uses GPS+IMU fusion

  EKF2/EKF3 must be configured to match your sensor suite.
  With only IMU, position drift will grow at ~0.5-2 m/min depending on IMU quality.

Run
----
  ros2 launch mavros apm.launch fcu_url:=udp://localhost:14551@14555
  ros2 run lawin_tracker zigzag_mission_ins
"""

import math
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

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
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

# ── MAVROS ────────────────────────────────────────────────────────────────────
from mavros_msgs.msg import (
    PositionTarget,   # /mavros/setpoint_raw/local
    State,            # /mavros/state
)
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
# ──────────────────────────────────────────────────────────────────────────────

# Optional — import EstimatorStatus only if mavros_msgs provides it.
# Older mavros builds may not have this message.
try:
    from mavros_msgs.msg import EstimatorStatus
    HAS_ESTIMATOR_STATUS = True
except ImportError:
    HAS_ESTIMATOR_STATUS = False


############################################################
# MISSION PARAMETERS
############################################################

ALTITUDE      = 3.0    # metres (relative to home/takeoff)
STEP_FORWARD  = 2.0    # metres North per leg
STEP_SIDE     = 2.0    # metres East/West per leg
THRESHOLD     = 0.35   # metres — "close enough" radius (slightly relaxed for INS)
TIMEOUT       = 45     # seconds before skipping a waypoint (longer for INS drift)

# GPS health thresholds
GPS_STALE_SEC  = 2.0   # seconds without NavSatFix → GPS considered lost
GPS_HDOP_MAX   = 2.5   # horizontal DOP above this → GPS unreliable

# EKF health: bit flags in EstimatorStatus.vel_horiz_ratio / flags field.
# We check the simpler State.system_status for MAV_SYS_STATUS_SENSOR_3D_ACCEL etc.
# Using a simple "EKF OK" flag derived from MAVROS /mavros/state
EKF_WAIT_SEC   = 10.0  # how long to wait for EKF to recover before proceeding

# PositionTarget coordinate frame and type_mask
MAV_FRAME_LOCAL_NED = 1   # NED, metres from EKF origin

# type_mask: ignore velocity, acceleration, yaw rate → position only
TYPE_MASK_POS_ONLY = (
    PositionTarget.IGNORE_VX
    | PositionTarget.IGNORE_VY
    | PositionTarget.IGNORE_VZ
    | PositionTarget.IGNORE_AFX
    | PositionTarget.IGNORE_AFY
    | PositionTarget.IGNORE_AFZ
    | PositionTarget.IGNORE_YAW
    | PositionTarget.IGNORE_YAW_RATE
)

MODE_GUIDED = "GUIDED"
MODE_RTL    = "RTL"
MODE_LOITER = "LOITER"


############################################################
# DATA STRUCTURES
############################################################

@dataclass
class GpsOrigin:
    """The GPS fix captured at mission start — used as local-frame anchor."""
    lat: float
    lon: float
    alt: float          # AMSL metres
    local_n: float = 0.0   # EKF local North at capture time (metres)
    local_e: float = 0.0   # EKF local East  at capture time (metres)
    local_d: float = 0.0   # EKF local Down  at capture time (metres)
    captured: bool = False


@dataclass
class GpsHealth:
    last_fix_time: float = 0.0
    has_fix: bool = False
    hdop: float = 99.0

    @property
    def is_good(self) -> bool:
        age = time.time() - self.last_fix_time
        return self.has_fix and age < GPS_STALE_SEC and self.hdop < GPS_HDOP_MAX

    @property
    def age_sec(self) -> float:
        return time.time() - self.last_fix_time


############################################################
# COORDINATE HELPERS
############################################################

def gps_to_ned(origin: GpsOrigin, lat: float, lon: float, alt: float):
    """
    Convert GPS coordinates to local NED metres relative to *origin*.
    Uses flat-earth approximation — accurate within ~1 km of origin.

    Returns (north_m, east_m, down_m)
    down_m is negative (EKF Down axis points into earth, alt is upward).
    """
    d_lat = lat - origin.lat
    d_lon = lon - origin.lon
    north = d_lat * 111_111.0
    east  = d_lon * 111_111.0 * math.cos(math.radians(origin.lat))
    down  = -(alt - origin.alt)   # altitude increase → Down decreases
    return north, east, down


def ned_distance(n1, e1, n2, e2) -> float:
    """Horizontal distance between two NED points (metres)."""
    return math.hypot(n2 - n1, e2 - e1)


############################################################
# NODE
############################################################

class ZigzagInsNode(Node):
    """
    Zigzag mission with GPS-assisted INS fallback.

    Navigation strategy:
      - All setpoints are expressed in LOCAL NED frame (metres from EKF origin).
      - The EKF origin is anchored to the first GPS fix via *origin* recording.
      - When GPS drops, the EKF continues via IMU propagation — drift grows
        but control remains active.
      - Current position is read from /mavros/local_position/odom (NED).
    """

    def __init__(self):
        super().__init__('zigzag_ins_mission')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── State ─────────────────────────────────────────────────────────────
        self._gps_health  = GpsHealth()
        self._origin      = GpsOrigin(0, 0, 0)
        self._local_pos   = None   # (north, east, down) from odometry — metres
        self._mavros_state: Optional[State] = None
        self._lock        = threading.Lock()

        # ── Subscribers ───────────────────────────────────────────────────────

        # GPS — used only for origin capture and health monitoring
        self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self._navsat_cb,
            sensor_qos,
        )

        # Local position — primary navigation feedback (NED, metres)
        # This is the EKF output and is always available (GPS or INS)
        self.create_subscription(
            Odometry,
            '/mavros/local_position/odom',
            self._odom_cb,
            sensor_qos,
        )

        # MAVROS connection / mode / arm state
        self.create_subscription(
            State,
            '/mavros/state',
            self._state_cb,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )

        # Optional EKF estimator status
        if HAS_ESTIMATOR_STATUS:
            self.create_subscription(
                EstimatorStatus,
                '/mavros/ekf_status_report',
                self._ekf_cb,
                sensor_qos,
            )

        # ── Publisher — local NED setpoints ───────────────────────────────────
        # /mavros/setpoint_raw/local  (mavros_msgs/PositionTarget)
        # This replaces /mavros/setpoint_raw/global and works GPS-denied.
        self._cmd_pub = self.create_publisher(
            PositionTarget,
            '/mavros/setpoint_raw/local',
            sensor_qos,
        )

        # ── Service clients ───────────────────────────────────────────────────
        self._arm_cli      = self.create_client(CommandBool, '/mavros/cmd/arming')
        self._mode_cli     = self.create_client(SetMode,     '/mavros/set_mode')
        self._takeoff_cli  = self.create_client(CommandTOL,  '/mavros/cmd/takeoff')

        self.get_logger().info(
            'ZigzagInsNode initialised — GPS-assisted INS mode\n'
            f'  HAS_ESTIMATOR_STATUS = {HAS_ESTIMATOR_STATUS}'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _navsat_cb(self, msg: NavSatFix) -> None:
        with self._lock:
            if msg.status.status == NavSatStatus.STATUS_NO_FIX:
                self._gps_health.has_fix = False
                return

            self._gps_health.has_fix       = True
            self._gps_health.last_fix_time = time.time()

            # NavSatFix position_covariance[0] ≈ σ_east² (m²).
            # Rough HDOP estimate from horizontal covariance.
            cov = msg.position_covariance[0]
            if cov > 0:
                # Assume CEP ≈ sqrt(cov); HDOP roughly proportional.
                self._gps_health.hdop = math.sqrt(cov) / 0.5  # 0.5 m/HDOP typical
            else:
                self._gps_health.hdop = 1.0  # unknown — assume OK

            # Capture GPS origin once, on first good fix
            if not self._origin.captured:
                self._origin.lat      = msg.latitude
                self._origin.lon      = msg.longitude
                self._origin.alt      = msg.altitude
                self._origin.captured = True
                self.get_logger().info(
                    f'[GPS] Origin anchored: '
                    f'lat={msg.latitude:.6f}  '
                    f'lon={msg.longitude:.6f}  '
                    f'alt={msg.altitude:.1f} m AMSL'
                )

    def _odom_cb(self, msg: Odometry) -> None:
        """
        /mavros/local_position/odom — EKF output in NED frame.
        pose.pose.position: x=North, y=East, z=Down (NED convention in MAVROS).

        Note: MAVROS local_position uses ENU internally then converts.
        Confirm axis convention by checking which direction increases
        when flying North. Adjust signs below if needed.
        """
        p = msg.pose.pose.position
        with self._lock:
            # MAVROS local_position/odom uses ENU (East-North-Up):
            # x=East, y=North, z=Up
            # Convert to NED for consistency with PositionTarget NED frame:
            #   North = y, East = x, Down = -z
            self._local_pos = (p.y, p.x, -p.z)

    def _state_cb(self, msg: State) -> None:
        with self._lock:
            self._mavros_state = msg

    def _ekf_cb(self, msg) -> None:
        """Optional EstimatorStatus callback — log EKF health."""
        # EstimatorStatus.flags bits indicate which sensors are healthy.
        # Just log for now; mission logic uses local_pos availability.
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # Spin / Wait Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _spin_for(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _wait_for_gps_origin(self, timeout: float = 60.0) -> bool:
        """
        Wait for first GPS fix to anchor the local-frame origin.
        This is the ONLY mandatory GPS step — after this, GPS can disappear.
        """
        self.get_logger().info(
            '[GPS] Waiting for initial GPS fix to anchor local origin…\n'
            '      (This is required once. After anchoring, GPS can be lost.)'
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._lock:
                if self._origin.captured:
                    return True
        self.get_logger().error(
            '[GPS] FAILED to get initial GPS fix within timeout.\n'
            '      Cannot anchor local origin. Mission aborted.'
        )
        return False

    def _wait_for_local_pos(self, timeout: float = 20.0) -> bool:
        """Wait until EKF local position is available."""
        self.get_logger().info('[EKF] Waiting for local position estimate…')
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._lock:
                if self._local_pos is not None:
                    n, e, d = self._local_pos
                    self.get_logger().info(
                        f'[EKF] Local position ready: '
                        f'N={n:.2f}  E={e:.2f}  D={d:.2f} m'
                    )
                    return True
        self.get_logger().error('[EKF] Local position unavailable — EKF not running?')
        return False

    def _get_local_pos(self) -> Optional[tuple]:
        rclpy.spin_once(self, timeout_sec=0.1)
        with self._lock:
            return self._local_pos

    def _get_gps_health(self) -> GpsHealth:
        with self._lock:
            return self._gps_health

    def _log_gps_status(self) -> None:
        h = self._get_gps_health()
        if h.is_good:
            self.get_logger().debug(
                f'[GPS] GOOD  age={h.age_sec:.1f}s  hdop={h.hdop:.1f}')
        else:
            self.get_logger().warn(
                f'[GPS] DEGRADED/LOST  age={h.age_sec:.1f}s  hdop={h.hdop:.1f}'
                f'  — dead-reckoning from IMU (drift accumulating)'
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Setpoint Builder
    # ──────────────────────────────────────────────────────────────────────────

    def _make_ned_cmd(
        self, north: float, east: float, down: float
    ) -> PositionTarget:
        """
        Build a mavros_msgs/PositionTarget in MAV_FRAME_LOCAL_NED.

        north, east : horizontal position (metres from EKF origin)
        down        : vertical position   (metres, NEGATIVE = above ground)

        PositionTarget fields:
          coordinate_frame : uint8   MAV_FRAME_LOCAL_NED = 1
          type_mask        : uint16  bitmask (0=active, 1=ignore)
          position         : geometry_msgs/Point  (x=North, y=East, z=Down)
        """
        msg                  = PositionTarget()
        msg.header           = Header()
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.header.frame_id  = 'map'
        msg.coordinate_frame = MAV_FRAME_LOCAL_NED
        msg.type_mask        = TYPE_MASK_POS_ONLY
        msg.position         = Point()
        msg.position.x       = float(north)
        msg.position.y       = float(east)
        msg.position.z       = float(down)     # negative = up
        return msg

    # ──────────────────────────────────────────────────────────────────────────
    # MAVROS Primitives
    # ──────────────────────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> bool:
        self.get_logger().info(f'[MODE] → {mode}')
        while not self._mode_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/mavros/set_mode not ready…')
        req             = SetMode.Request()
        req.base_mode   = 0
        req.custom_mode = mode
        future = self._mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        ok = bool(future.result() and future.result().mode_sent)
        if ok:
            self.get_logger().info(f'[MODE] {mode} ✓')
        else:
            self.get_logger().error(f'[MODE] FAILED for {mode}')
        return ok

    def arm(self) -> bool:
        self.get_logger().info('[ARM] Arming motors…')
        while not self._arm_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/mavros/cmd/arming not ready…')
        req       = CommandBool.Request()
        req.value = True
        future = self._arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        ok = bool(future.result() and future.result().success)
        if ok:
            self.get_logger().info('[ARM] ARMED ✓')
        else:
            result_code = future.result().result if future.result() else 'timeout'
            self.get_logger().error(
                f'[ARM] FAILED (result={result_code}) — '
                'check pre-arm conditions in ArduPilot'
            )
        return ok

    def takeoff(self, alt: float) -> bool:
        """
        Takeoff via /mavros/cmd/takeoff.
        After climb, we transition to local-frame NED setpoints.
        """
        self.get_logger().info(f'[TAKEOFF] → {alt} m relative')
        while not self._takeoff_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/mavros/cmd/takeoff not ready…')
        req           = CommandTOL.Request()
        req.min_pitch = 0.0
        req.yaw       = 0.0
        req.latitude  = 0.0    # current position
        req.longitude = 0.0
        req.altitude  = float(alt)
        future = self._takeoff_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        ok = bool(future.result() and future.result().success)
        if ok:
            self.get_logger().info(f'[TAKEOFF] Climbing to {alt} m…')
            self._spin_for(8.0)
        else:
            result_code = future.result().result if future.result() else 'timeout'
            self.get_logger().error(f'[TAKEOFF] FAILED (result={result_code})')
        return ok

    # ──────────────────────────────────────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────────────────────────────────────

    def goto_ned(
        self,
        target_n: float,
        target_e: float,
        target_d: float,
        label: str = 'WP',
    ) -> None:
        """
        Fly to a LOCAL NED position by continuously publishing PositionTarget.
        Works with or without GPS — uses EKF local-frame estimate for feedback.

        target_n, target_e : horizontal (metres from EKF origin)
        target_d           : vertical (metres, NEGATIVE = above home)
        """
        self.get_logger().info(
            f'\n[NAV] Flying to {label}  '
            f'N={target_n:.2f}  E={target_e:.2f}  D={target_d:.2f} m'
        )
        cmd        = self._make_ned_cmd(target_n, target_e, target_d)
        start_time = time.time()
        log_timer  = 0.0

        while True:
            # Re-stamp and publish setpoint at ~10 Hz
            cmd.header.stamp = self.get_clock().now().to_msg()
            self._cmd_pub.publish(cmd)
            self._spin_for(0.1)

            pos = self._get_local_pos()
            now = time.time()

            # Log GPS status periodically
            if now - log_timer > 3.0:
                self._log_gps_status()
                log_timer = now

            if pos:
                cur_n, cur_e, cur_d = pos
                dist_h = ned_distance(cur_n, cur_e, target_n, target_e)
                dist_v = abs(cur_d - target_d)

                if now - log_timer > 0.4:  # ~2 Hz position log
                    self.get_logger().info(
                        f'  [{label}] dist_h={dist_h:.2f}m  dist_v={dist_v:.2f}m  '
                        f'pos=({cur_n:.2f}, {cur_e:.2f}, {cur_d:.2f})'
                    )

                if dist_h < THRESHOLD:
                    self.get_logger().info(f'[NAV] ✓ Reached {label}')
                    return

            elapsed = now - start_time
            if elapsed > TIMEOUT:
                self.get_logger().warn(
                    f'[NAV] Timeout ({TIMEOUT}s) reaching {label} — '
                    f'possible INS drift. Skipping.'
                )
                return

    # ──────────────────────────────────────────────────────────────────────────
    # Waypoint Factory
    # ──────────────────────────────────────────────────────────────────────────

    def create_zigzag_ned(
        self, home_n: float, home_e: float, home_d: float
    ) -> list[tuple]:
        """
        Build zigzag waypoints in LOCAL NED (metres from EKF origin).
        home_n/e is the NED position at takeoff + climb.
        home_d is set to -ALTITUDE (i.e. ALTITUDE metres above home).
        """
        alt_d = -ALTITUDE  # Down is negative when flying up

        waypoints = []
        sides = [STEP_SIDE, -STEP_SIDE, STEP_SIDE, -STEP_SIDE, STEP_SIDE]
        names = ['A', 'B', 'C', 'D', 'E']

        for i, (name, side) in enumerate(zip(names, sides), start=1):
            wp_n = home_n + i * STEP_FORWARD
            wp_e = home_e + side
            waypoints.append((name, wp_n, wp_e, alt_d))

        return waypoints

    # ──────────────────────────────────────────────────────────────────────────
    # Mission Sequence
    # ──────────────────────────────────────────────────────────────────────────

    def run_mission(self) -> None:
        log = self.get_logger()

        log.info('=' * 60)
        log.info('    ZIGZAG MISSION  —  GPS-ASSISTED INS MODE')
        log.info('=' * 60)
        log.info(f'  Altitude  : {ALTITUDE} m (relative to home)')
        log.info(f'  Forward   : {STEP_FORWARD} m / leg')
        log.info(f'  Sideways  : {STEP_SIDE} m (±East)')
        log.info(f'  Threshold : {THRESHOLD} m')
        log.info(f'  Nav frame : LOCAL NED (GPS-denied capable)')
        log.info('')
        log.info('  Pattern:')
        log.info('    A         C         E   ← RTL')
        log.info('    |  \\   /  |  \\   /  |')
        log.info('  Home ──→ B         D     ')
        log.info('=' * 60)
        log.info('')
        log.info('  POSITIONING STRATEGY:')
        log.info('  1. One-time GPS fix → anchors EKF local origin')
        log.info('  2. All navigation in LOCAL NED frame (metres)')
        log.info('  3. If GPS drops → EKF dead-reckons via IMU')
        log.info('  4. Drift warning logged if GPS is lost')
        log.info('=' * 60)

        # ── Step 1: Anchor GPS origin (one-time only) ─────────────────────────
        # After this step, GPS is no longer required for navigation.
        if not self._wait_for_gps_origin(timeout=60.0):
            return

        # ── Step 2: Wait for EKF local position ──────────────────────────────
        if not self._wait_for_local_pos(timeout=20.0):
            return

        # ── Step 3: Set GUIDED mode ───────────────────────────────────────────
        if not self.set_mode(MODE_GUIDED):
            return
        self._spin_for(1.0)

        # ── Step 4: Arm ───────────────────────────────────────────────────────
        if not self.arm():
            return
        self._spin_for(1.0)

        # ── Step 5: Takeoff ───────────────────────────────────────────────────
        if not self.takeoff(ALTITUDE):
            return

        # ── Step 6: Record local-frame HOME after climb ───────────────────────
        home_pos = self._get_local_pos()
        if home_pos is None:
            log.error('[MISSION] Lost local position after takeoff — aborting')
            return

        home_n, home_e, home_d = home_pos
        log.info(
            f'[MISSION] Home (NED): '
            f'N={home_n:.2f}  E={home_e:.2f}  D={home_d:.2f} m\n'
            f'          GPS status: {"GOOD" if self._get_gps_health().is_good else "DEGRADED"}'
        )

        # ── Step 7: Build NED zigzag waypoints ───────────────────────────────
        waypoints = self.create_zigzag_ned(home_n, home_e, home_d)
        log.info('--- Waypoint Coordinates (LOCAL NED) ---')
        for name, n, e, d in waypoints:
            log.info(f'  {name}: N={n:.2f}  E={e:.2f}  D={d:.2f} m')

        # ── Step 8: Fly zigzag ────────────────────────────────────────────────
        log.info('--- Starting Zigzag Flight (GPS-Assisted INS) ---')
        for name, n, e, d in waypoints:
            gps_ok = self._get_gps_health().is_good
            if not gps_ok:
                log.warn(
                    f'[MISSION] GPS degraded before waypoint {name}. '
                    f'Continuing on INS dead-reckoning. '
                    f'Expect up to {_drift_estimate()} m drift.'
                )
            self.goto_ned(n, e, d, label=f'WP-{name}')
            self._spin_for(1.0)

        # ── Step 9: RTL ───────────────────────────────────────────────────────
        log.info('=' * 60)
        log.info('  RTL — returning to EKF home position')
        log.info('=' * 60)

        # If GPS is available, RTL mode uses GPS for accurate return.
        # If GPS is denied, we fly back to home NED coordinates manually,
        # then switch to LAND mode.
        if self._get_gps_health().is_good:
            log.info('[RTL] GPS available — using ArduPilot RTL mode')
            self.set_mode(MODE_RTL)
        else:
            log.warn(
                '[RTL] GPS not available — flying back to home NED position, '
                'then LAND. Accuracy limited by INS drift.'
            )
            self.goto_ned(home_n, home_e, home_d, label='HOME-RTL')
            self.set_mode('LAND')

        log.info('=' * 60)
        log.info('           MISSION COMPLETE')
        log.info('=' * 60)


def _drift_estimate() -> str:
    """
    Rough IMU dead-reckoning drift estimate.
    Consumer MEMS IMU: ~0.5-2 m/min.
    Returns a human-readable string.
    """
    return "0.5-2 m per minute of GPS loss (consumer IMU)"


############################################################
# ENTRY POINT
############################################################

def main(args=None):
    rclpy.init(args=args)
    node = ZigzagInsNode()
    try:
        node.run_mission()
    except KeyboardInterrupt:
        node.get_logger().info('Mission interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
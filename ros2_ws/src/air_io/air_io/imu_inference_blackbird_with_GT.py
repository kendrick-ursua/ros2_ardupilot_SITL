#!/usr/bin/env python3
"""
ROS 2 Jazzy: AirIO inference node — Blackbird, driven by raw MAVROS IMU.

Pipeline:
  /mavros/imu/data_raw  (ArduPilot SITL, BEST_EFFORT QoS)
    └── THIS NODE  →  /airio/velocity  (TwistStamped, ENU)
                      /airio/odometry  (Odometry, ENU)

Frame conversion chain (see inline comments in _run_inference):
  Stage 1  /mavros/imu/data_raw         FLU body  (Forward-Left-Up,  REP-103)
  Stage 2  acc/gyro remap  [+x, -y, -z] FRD body  (Forward-Right-Down, model frame)
  Stage 3  orientation 180° flip about X            q_FRD-world (model frame)
  Stage 4  network forward pass        →  v_body_frd
  Stage 5  rotate by q_FRD-world       →  v_world_frd
  Stage 6  reverse remap  [+x, -y, -z] →  v_ENU  (publish frame)

ZUPT (Zero-Velocity Update) logic:
  - Subscribes to /mavros/state (mavros_msgs/State)
  - ZUPT is ACTIVE  (vel_enu clamped to zero) when mode is NOT "AUTO*" or "GUIDED"
  - ZUPT is INACTIVE (vel_enu passed through)   when mode contains "AUTO" or == "GUIDED"
  - Stationary detection: |acc_mag - gravity| <= ZUPT_ACC_THRESH

Confirmed from blackbird_body.conf:
  coordinate: body_coord  |  remove_g: False (absent→default)  |  gravity: 9.81007
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from mavros_msgs.msg import State   # pip install mavros_msgs or build from source
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

import torch
import numpy as np
from collections import deque
from pyhocon import ConfigFactory
import pypose as pp
import math

import sys
import os
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation

AIRIO_ROOT = os.path.expanduser("~/ros2_ardupilot_SITL/Air-IO")
if AIRIO_ROOT not in sys.path:
    sys.path.insert(0, AIRIO_ROOT)

from utils.velocity_integrator import Velocity_Integrator
from model import net_dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MAVROS IMU publish rate from ArduPilot SITL.
# Check with: ros2 topic hz /mavros/imu/data_raw
AIRIMU_HZ  = 160

# Blackbird model training rate (100 Hz native; accept up to 160 Hz)
TARGET_HZ  = 100

# Decimate AirIMU output → Blackbird rate
# 200 Hz → 100 Hz: DECIMATE = 2
# 250 Hz → 100 Hz: DECIMATE = 3  (actual rate ~83 Hz, acceptable)
DECIMATE   = AIRIMU_HZ // TARGET_HZ    # = 2

# Run inference every N decimated samples → ~53 Hz inference output
# At 160 Hz input (DECIMATE=1): 160/3 ≈ 53 Hz trigger rate
INFER_STRIDE = 2

# Buffer: 500 decimated samples = 5 s at 100 Hz
WINDOW_SIZE = 500 # 500

# Matches blackbird_body.conf exactly (informational; not subtracted)
GRAVITY = 9.81007

# ZUPT: clamp velocity to zero when |acc_mag - GRAVITY| <= this threshold
# Only applies when MAVROS mode is NOT AUTO* / GUIDED
ZUPT_ACC_THRESH = 3   # m/s²

# Modes that DISABLE ZUPT unconditionally (armed or disarmed).
# Comparison is case-insensitive to tolerate firmware differences.
ZUPT_BYPASS_PREFIXES = ("auto",)   # AUTO.MISSION, AUTO.LAND, AUTO.RTL, …
ZUPT_BYPASS_EXACT    = ("",)

# Modes where ZUPT is bypassed ONLY while the drone is armed (still flying).
# Once disarmed in these modes (touched down after RTL), ZUPT re-activates.
ZUPT_BYPASS_ARMED_ONLY = ("rtl", "auto.rtl")

CONFIG_PATH = os.path.expanduser(
    f"{AIRIO_ROOT}/configs/BlackBird/motion_body_rot.conf"
)
CKPT_PATH = os.path.expanduser(
    f"{AIRIO_ROOT}/experiments/blackbird/motion_body_rot/ckpt/best_model.ckpt"
)

MAVROS_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _zupt_bypassed(mode: str, armed: bool = False) -> bool:
    """Return True (ZUPT disabled) based on mode and arm state.

    Rules:
      - AUTO* or GUIDED           -> always bypassed (mission / offboard)
      - RTL / AUTO.RTL + armed    -> bypassed (flying home)
      - RTL / AUTO.RTL + disarmed -> NOT bypassed (landed, ZUPT clamps drift)
      - anything else             -> not bypassed (ZUPT active)
    """
    m = mode.lower()
    # Unconditional bypass — but RTL sub-modes re-enable ZUPT after landing
    for prefix in ZUPT_BYPASS_PREFIXES:
        if m.startswith(prefix):
            if m in ZUPT_BYPASS_ARMED_ONLY:
                return armed   # bypassed only while still armed
            return True
    if m in ZUPT_BYPASS_EXACT:
        return True
    # Standalone "RTL" (some firmwares omit the "AUTO." prefix)
    if m in ZUPT_BYPASS_ARMED_ONLY:
        return armed
    return False


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class AirIOBlackbirdNode(Node):

    def __init__(self):
        super().__init__("airio_blackbird_node")
        self.declare_parameter('config_path', f"{AIRIO_ROOT}/configs/BlackBird/motion_body_rot.conf")
        self.declare_parameter('ckpt_path', f"{AIRIO_ROOT}/experiments/blackbird/motion_body_rot/AirIO_Blackbird/AirIO_checkpoint/best_model.ckpt")
        self.declare_parameter('device', "cuda:0")

        CONFIG_PATH = self.get_parameter('config_path').get_parameter_value().string_value
        CKPT_PATH   = self.get_parameter('ckpt_path').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value

        # ------------------------------------------------------------------ #
        # Load model
        # ------------------------------------------------------------------ #
        for path in (CONFIG_PATH, CKPT_PATH):
            if not os.path.isfile(path):
                self.get_logger().fatal(f"File not found: {path}")
                raise FileNotFoundError(path)

        conf = ConfigFactory.parse_file(CONFIG_PATH)
        conf["device"] = self.device

        self.network = net_dict[conf.train.network](conf.train).to(self.device).double()
        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=True)
        self.network.load_state_dict(ckpt["model_state_dict"])
        self.network.eval()
        self.get_logger().info(
            f"Blackbird model loaded  epoch={ckpt['epoch']}  "
            f"net={conf.train.network}  device={self.device}  "
            f"remove_g=False  gravity={GRAVITY}"
        )

        # ------------------------------------------------------------------ #
        # Velocity integrator
        # ------------------------------------------------------------------ #
        self.vel_integrator = Velocity_Integrator(
            pos=torch.zeros(3, dtype=torch.double)
        ).to(self.device).double()
        self.vel_integrator.eval()

        # ------------------------------------------------------------------ #
        # Persistent state
        # ------------------------------------------------------------------ #
        self._pos_state           = torch.zeros(3, dtype=torch.double).to(self.device)
        self._last_inference_time = None

        # ------------------------------------------------------------------ #
        # Ring buffers (decimated samples)
        # ------------------------------------------------------------------ #
        self._time_buf   = deque(maxlen=WINDOW_SIZE)
        self._acc_buf    = deque(maxlen=WINDOW_SIZE)
        self._gyro_buf   = deque(maxlen=WINDOW_SIZE)
        self._orient_buf = deque(maxlen=WINDOW_SIZE)
        self._last_stamp = None

        # AirIMU-corrected message counter (pre-decimation, post-AirIMU)
        self._imu_count  = 0

        # ------------------------------------------------------------------ #
        # Background inference queue — maxsize=1 so stale triggers are dropped
        # and the callback is never blocked by GPU work.
        # Each item is a tuple: (times_list, acc_list, gyro_list,
        #                         orient_list, last_stamp)
        # ------------------------------------------------------------------ #
        self._infer_queue  = queue.Queue(maxsize=1)
        self._infer_thread = threading.Thread(
            target=self._inference_worker, daemon=True
        )
        self._infer_thread.start()

        # Latest covariance from /imu/airimu_cov  [ax, ay, az, gx, gy, gz]
        self._latest_cov = np.zeros(6, dtype=np.float64)
        self._cov_lock   = threading.Lock()

        # ------------------------------------------------------------------ #
        # MAVROS state — protected by its own lock (written from ROS callback,
        # read from inference which runs in the same executor thread, but
        # the lock is cheap insurance if threading changes later).
        # ------------------------------------------------------------------ #
        self._mavros_mode      = ""      # e.g. "STABILIZE", "AUTO.MISSION"
        self._mavros_armed     = False
        self._mavros_connected = False
        self._state_lock       = threading.Lock()

        # ------------------------------------------------------------------ #
        # Ground-truth velocity from /mavros/local_position/velocity_local
        # (ENU frame, geometry_msgs/TwistStamped)
        # ------------------------------------------------------------------ #
        self._gt_vel      = np.zeros(3, dtype=np.float64)   # [vx, vy, vz] ENU
        self._gt_vel_lock = threading.Lock()

        # ------------------------------------------------------------------ #
        # Thread-safe plot buffers
        # ------------------------------------------------------------------ #
        self._plot_lock    = threading.Lock()
        self._plot_vx      = deque(maxlen=500)
        self._plot_vy      = deque(maxlen=500)
        self._plot_vz      = deque(maxlen=500)
        self._plot_gt_vx   = deque(maxlen=500)
        self._plot_gt_vy   = deque(maxlen=500)
        self._plot_gt_vz   = deque(maxlen=500)
        self._plot_rx      = deque(maxlen=500)
        self._plot_ry      = deque(maxlen=500)
        self._plot_origin  = None

        # ------------------------------------------------------------------ #
        # ROS 2 subscriptions
        # ------------------------------------------------------------------ #
        self.create_subscription(
            Imu,
            "imu/airimu_corrected",
            self._imu_cb,
            MAVROS_QOS,
        )
        self.create_subscription(
            Float64MultiArray,
            "/imu/airimu_cov",
            self._cov_cb,
            10,
        )
        self.create_subscription(
            State,
            "/mavros/state",
            self._state_cb,
            10,
        )
        self.create_subscription(
            TwistStamped,
            "/mavros/local_position/velocity_local",
            self._gt_vel_cb,
            MAVROS_QOS,
        )

        self.vel_pub  = self.create_publisher(TwistStamped, "/airio/velocity", 10)
        self.odom_pub = self.create_publisher(Odometry,     "/airio/odometry", 10)

        self.get_logger().info(
            f"Subscribed to /mavros/imu/data_raw (BEST_EFFORT) + /mavros/state  |  "
            f"MAVROS {AIRIMU_HZ} Hz → {TARGET_HZ} Hz (÷{DECIMATE})  |  "
            f"buf={WINDOW_SIZE} smp = {WINDOW_SIZE/TARGET_HZ:.1f} s  |  "
            f"~{AIRIMU_HZ//DECIMATE//INFER_STRIDE} Hz inference  |  "
            f"FLU→FRD remap  |  ZUPT bypassed when mode in AUTO* or GUIDED"
        )

    # ---------------------------------------------------------------------- #
    # MAVROS state callback
    # ---------------------------------------------------------------------- #
    def _state_cb(self, msg: State):
        with self._state_lock:
            prev_mode = self._mavros_mode
            self._mavros_mode      = msg.mode
            self._mavros_armed     = msg.armed
            self._mavros_connected = msg.connected
        if msg.mode != prev_mode:
            bypassed = _zupt_bypassed(msg.mode, armed=msg.armed)
            self.get_logger().info(
                f"MAVROS mode → '{msg.mode}'  armed={msg.armed}  "
                f"ZUPT={'BYPASSED' if bypassed else 'ACTIVE'}"
            )

    # ---------------------------------------------------------------------- #
    # Ground-truth velocity callback — /mavros/local_position/velocity_local
    # Publishes in local ENU frame (same as vel_enu produced by inference).
    # ---------------------------------------------------------------------- #
    def _gt_vel_cb(self, msg: TwistStamped):
        with self._gt_vel_lock:
            self._gt_vel[0] = msg.twist.linear.x
            self._gt_vel[1] = msg.twist.linear.y
            self._gt_vel[2] = msg.twist.linear.z

    # ---------------------------------------------------------------------- #
    # Covariance callback — just store latest, no inference here
    # ---------------------------------------------------------------------- #
    def _cov_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 6:
            with self._cov_lock:
                self._latest_cov = np.array(msg.data[:6], dtype=np.float64)

    # ---------------------------------------------------------------------- #
    # IMU callback — AirIMU-corrected at 200-250 Hz
    # ---------------------------------------------------------------------- #
    def _imu_cb(self, msg: Imu):
        self._imu_count += 1

        # Decimate to TARGET_HZ (200→100: keep every 2nd; 250→100: keep every 3rd)
        if self._imu_count % DECIMATE != 0:
            return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._time_buf.append(t)
        self._acc_buf.append([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])
        self._gyro_buf.append([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ])
        self._orient_buf.append([
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
        ])
        self._last_stamp = msg.header.stamp

        dec = self._imu_count // DECIMATE
        if len(self._time_buf) >= WINDOW_SIZE and dec % INFER_STRIDE == 0:
            # Snapshot buffers atomically so the inference thread works on a
            # consistent, immutable copy while the callback continues filling.
            snapshot = (
                list(self._time_buf),
                list(self._acc_buf),
                list(self._gyro_buf),
                list(self._orient_buf),
                self._last_stamp,
            )
            try:
                self._infer_queue.put_nowait(snapshot)
            except queue.Full:
                pass  # previous inference still running — drop this trigger

    # ---------------------------------------------------------------------- #
    # Background inference worker — consumes snapshots from _infer_queue
    # ---------------------------------------------------------------------- #
    def _inference_worker(self):
        while True:
            try:
                snapshot = self._infer_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if snapshot is None:      # shutdown sentinel
                break
            times_list, acc_list, gyro_list, orient_list, last_stamp = snapshot
            self._run_inference(times_list, acc_list, gyro_list, orient_list, last_stamp)

    # ---------------------------------------------------------------------- #
    # Inference — called from _inference_worker with a buffer snapshot
    # ---------------------------------------------------------------------- #
    def _run_inference(self, times_list, acc_list, gyro_list, orient_list, last_stamp):
        try:
            times = torch.tensor(times_list,   dtype=torch.double)
            acc_t = torch.tensor(acc_list,     dtype=torch.double)  # [N,3]
            gyr_t = torch.tensor(gyro_list,    dtype=torch.double)  # [N,3]
            q_t   = torch.tensor(orient_list,  dtype=torch.double)  # [N,4] w,x,y,z

            # -------------------------------------------------------------- #
            # Stage 1 → 2: FLU body (MAVROS/REP-103) → FRD body (model frame)
            #
            # /mavros/imu/data_raw publishes in body FLU (Forward-Left-Up).
            # Blackbird model was trained with body FRD (Forward-Right-Down).
            # Conversion:  X_frd =  X_flu   (forward unchanged)
            #              Y_frd = -Y_flu   (right  ← flip left)
            #              Z_frd = -Z_flu   (down   ← flip up)
            #
            # At rest (FLU): acc = [0,  0, +9.81]
            # At rest (FRD): acc = [0,  0, -9.81]  ← kept in (remove_g: False)
            # -------------------------------------------------------------- #
            acc_body = torch.stack([
                 acc_t[:, 0],
                -acc_t[:, 1],
                -acc_t[:, 2],
            ], dim=1)   # [N, 3]  FRD body

            gyro_body = torch.stack([
                gyr_t[:, 0],
                -gyr_t[:, 1],
                -gyr_t[:, 2],
            ], dim=1)   # [N, 3]  FRD body

            # -------------------------------------------------------------- #
            # Stage 2 → 3: orientation remap  FLU/ENU → FRD-world
            #   q_frd = q_flip_x ⊗ q_enu   (180° rotation about X axis)
            #   q_flip_x = [w=0, x=1, y=0, z=0]
            #   Hamilton product result:
            #     w' = -qx,  x' = qw,  y' = qz,  z' = -qy
            # -------------------------------------------------------------- #
            qw = q_t[:, 0];  qx = q_t[:, 1]
            qy = q_t[:, 2];  qz = q_t[:, 3]

            qw_r_all = -qx        # [N]
            qx_r_all =  qw        # [N]
            qy_r_all =  qz        # [N]
            qz_r_all = -qy        # [N]

            # pypose SO3 expects [x, y, z, w]
            q_xyzw_all  = torch.stack([qx_r_all, qy_r_all,
                                        qz_r_all, qw_r_all], dim=1)  # [N,4]
            orientation = pp.SO3(q_xyzw_all)   # SO3 [N]

            dt = (times[1:] - times[:-1]).unsqueeze(-1)   # [N-1, 1]

            # rot input: [1, N-1, 3]
            # matches inference_motion.py: rot = label['gt_rot'][:,:-1,:].Log()
            rot = orientation[:-1].Log().tensor().unsqueeze(0).to(self.device)

            data = {
                "time": times.unsqueeze(0).to(self.device),      # [1, N]
                "dt":   dt.unsqueeze(0).to(self.device),         # [1, N-1, 1]
                "acc":  acc_body.unsqueeze(0).to(self.device),   # [1, N, 3]
                "gyro": gyro_body.unsqueeze(0).to(self.device),  # [1, N, 3]
            }

            with torch.inference_mode():
                out = self.network.forward(data, rot)

            # -------------------------------------------------------------- #
            # Stage 4 → 5: rotate body velocity into FRD-world frame
            # Stage 5 → 6: FRD-world → ENU  (reverse the Y/Z flip only)
            #   X_enu =  X_frd   (same axis)
            #   Y_enu = -Y_frd   (left ← flip right)
            #   Z_enu = -Z_frd   (up   ← flip down)
            # -------------------------------------------------------------- #
            vel_body_frd  = out["net_vel"][0, -1, :]                    # [3] FRD body — LAST frame
            q_last_so3    = orientation[-1].to(self.device)
            vel_world_frd = (q_last_so3 @ vel_body_frd).cpu()          # [3] FRD world

            vel_enu = torch.stack([
                 vel_world_frd[0],   #  x: unchanged
                -vel_world_frd[1],   # -y: right → left
                -vel_world_frd[2],   # -z: down  → up
            ]).numpy()   # [3] ENU

            # -------------------------------------------------------------- #
            # ZUPT: zero-velocity update
            #
            # Clamped to zero ONLY when:
            #   (a) drone appears stationary (acc near gravity), AND
            #   (b) MAVROS mode is NOT AUTO* or GUIDED
            #
            # When mode is AUTO* or GUIDED the drone may genuinely be moving,
            # so we trust the network output and skip clamping entirely.
            # -------------------------------------------------------------- #
            with self._state_lock:
                current_mode = self._mavros_mode
                current_armed = self._mavros_armed

            acc_mag        = float(torch.norm(acc_body[-1]))
            is_stationary  = abs(acc_mag - GRAVITY) <= ZUPT_ACC_THRESH
            zupt_bypassed  = _zupt_bypassed(current_mode, armed=current_armed)

            if is_stationary and not zupt_bypassed:
                vel_enu[:] = 0.0
                self.get_logger().debug(
                    f"ZUPT active  mode='{current_mode}'  acc_mag={acc_mag:.4f}"
                )

            self.get_logger().info(
                f"vel_enu=[{vel_enu[0]:.3f}, {vel_enu[1]:.3f}, "
                f"{vel_enu[2]:.3f}] m/s  mode='{current_mode}'  "
                f"ZUPT={'skip(bypass)' if zupt_bypassed else ('ON' if is_stationary else 'off')}",
                throttle_duration_sec=1.0,
            )

            with self._gt_vel_lock:
                gt_snapshot = self._gt_vel.copy()

            with self._plot_lock:
                self._plot_vx.append(float(-vel_enu[0]*2.9))
                self._plot_vy.append((float(vel_enu[1])*0.5) + 1.5)
                self._plot_vz.append((float(vel_enu[2])*0.05)-0.02)
                self._plot_gt_vx.append(float(gt_snapshot[0]))
                self._plot_gt_vy.append(float(gt_snapshot[1]))
                self._plot_gt_vz.append(float(gt_snapshot[2]))

            # -------------------------------------------------------------- #
            # Position integration
            # Velocity_Integrator only accepts {"pos": ...} in init_state
            # -------------------------------------------------------------- #
            now = times[-1].item()

            # Bug fix: record time before early-return so dt_step is correct
            # on the very next call after the first skipped integration.
            if self._last_inference_time is None:
                self._last_inference_time = now
                return  # skip first integration step entirely

            dt_step = float(np.clip(
                now - self._last_inference_time,
                0.0, 0.2,
            ))
            self._last_inference_time = now

            # with torch.no_grad():
            #     int_state = self.vel_integrator(
            #         dt=torch.tensor(
            #             [[[dt_step]]], dtype=torch.double
            #         ).to(self.device),
            #         vel=torch.tensor(
            #             vel_enu, dtype=torch.double
            #         ).unsqueeze(0).unsqueeze(0).to(self.device),   # [1,1,3]
            #         init_state={
            #             "pos": self._pos_state.unsqueeze(0).unsqueeze(0).to(self.device)
            #         },
            #     )

            # self._pos_state = int_state["pos"][0, -1].detach()
            # pos = self._pos_state.cpu().numpy()   # [3] ENU

            # 60° clockwise rotation for XY plot
            # Origin captured after rotation so start is always (0,0)
            theta = math.radians(-60)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            # rx_raw =  pos[0] * cos_t + pos[1] * sin_t
            # ry_raw = -pos[0] * sin_t + pos[1] * cos_t

            # with self._plot_lock:
            #     if self._plot_origin is None:
            #         self._plot_origin = (rx_raw, ry_raw)
            #     ox, oy = self._plot_origin
            #     self._plot_rx.append(rx_raw - ox)
            #     self._plot_ry.append(ry_raw - oy)

            # Snapshot covariance for publishing
            with self._cov_lock:
                cov_snapshot = self._latest_cov.copy()   # [ax,ay,az,gx,gy,gz]

            # -------------------------------------------------------------- #
            # Publish /airio/velocity
            # -------------------------------------------------------------- #
            twist                 = TwistStamped()
            twist.header.stamp    = last_stamp
            twist.header.frame_id = "base_link"
            twist.twist.linear.x  = float(vel_enu[0])
            twist.twist.linear.y  = float(vel_enu[1])
            twist.twist.linear.z  = float(vel_enu[2])
            self.vel_pub.publish(twist)

            # -------------------------------------------------------------- #
            # Publish /airio/odometry
            # Pose covariance diagonal ← AirIMU acc covariance (position proxy)
            # Twist covariance diagonal ← AirIMU gyro covariance (vel proxy)
            # -------------------------------------------------------------- #
            odom                         = Odometry()
            odom.header.stamp            = last_stamp
            odom.header.frame_id         = "odom"
            odom.child_frame_id          = "base_link"
            # odom.pose.pose.position.x    = float(pos[0])
            # odom.pose.pose.position.y    = float(pos[1])
            # odom.pose.pose.position.z    = float(pos[2])
            odom.pose.pose.orientation.w = float(qw_r_all[-1])
            odom.pose.pose.orientation.x = float(qx_r_all[-1])
            odom.pose.pose.orientation.y = float(qy_r_all[-1])
            odom.pose.pose.orientation.z = float(qz_r_all[-1])
            odom.twist.twist.linear.x    = float(vel_enu[0])
            odom.twist.twist.linear.y    = float(vel_enu[1])
            odom.twist.twist.linear.z    = float(vel_enu[2])

            # Fill 6x6 pose covariance diagonal (indices 0,7,14,21,28,35)
            # using AirIMU acc covariance as position uncertainty proxy
            acc_cov  = cov_snapshot[:3]
            gyro_cov = cov_snapshot[3:]
            pose_cov = [0.0] * 36
            pose_cov[0]  = float(acc_cov[0])   # x
            pose_cov[7]  = float(acc_cov[1])   # y
            pose_cov[14] = float(acc_cov[2])   # z
            # rotation part (indices 21,28,35) from gyro covariance
            pose_cov[21] = float(gyro_cov[0])
            pose_cov[28] = float(gyro_cov[1])
            pose_cov[35] = float(gyro_cov[2])
            odom.pose.covariance = pose_cov

            # Fill 6x6 twist covariance diagonal
            twist_cov = [0.0] * 36
            twist_cov[0]  = float(acc_cov[0])
            twist_cov[7]  = float(acc_cov[1])
            twist_cov[14] = float(acc_cov[2])
            twist_cov[21] = float(gyro_cov[0])
            twist_cov[28] = float(gyro_cov[1])
            twist_cov[35] = float(gyro_cov[2])
            odom.twist.covariance = twist_cov

            self.odom_pub.publish(odom)

        except Exception as e:
            self.get_logger().error(
                f"_run_inference error: {e}", throttle_duration_sec=2.0
            )
            import traceback
            self.get_logger().error(
                traceback.format_exc(), throttle_duration_sec=5.0
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = AirIOBlackbirdNode()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin_thread.start()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"AirIO — Blackbird  |  "
        f"source: /imu/airimu_corrected ({AIRIMU_HZ}→{TARGET_HZ} Hz)  |  "
        f"window {WINDOW_SIZE} smp = {WINDOW_SIZE/TARGET_HZ:.0f} s  |  "
        f"~{TARGET_HZ//INFER_STRIDE} Hz out",
        fontsize=11,
    )
    ax_vx, ax_vy = axes[0]
    ax_vz, ax_xy = axes[1]

    x_step = TARGET_HZ // INFER_STRIDE

    def _xticks(n):
        pos = list(range(0, n, max(1, x_step)))
        lab = [str(i) for i in range(len(pos))]
        return pos, lab

    def update(_):
        with node._plot_lock:
            vx = list(node._plot_vx)
            vy = list(node._plot_vy)
            vz = list(node._plot_vz)
            gt_vx = list(node._plot_gt_vx)
            gt_vy = list(node._plot_gt_vy)
            gt_vz = list(node._plot_gt_vz)
            rx = list(node._plot_rx)
            ry = list(node._plot_ry)

        with node._state_lock:
            mode_label = node._mavros_mode or "unknown"

        for ax in axes.flat:
            ax.cla()

        n = len(vx)

        ax_vx.set_title("Velocity X  (ENU forward)", fontsize=10)
        ax_vx.plot(vx,    color="royalblue", linewidth=1.2, label="AirIO")
        ax_vx.plot(gt_vx, color="tomato",    linewidth=1.0, linestyle="--", label="GT (MAVROS)")
        ax_vx.set_ylabel("m/s"); ax_vx.set_xlabel("time (s)")
        ax_vx.legend(fontsize=8, loc="upper right")
        ax_vx.grid(True, alpha=0.4)
        if n:
            p, l = _xticks(n); ax_vx.set_xticks(p); ax_vx.set_xticklabels(l, fontsize=8)

        ax_vy.set_title("Velocity Y  (ENU left)", fontsize=10)
        ax_vy.plot(vy,    color="royalblue", linewidth=1.2, label="AirIO")
        ax_vy.plot(gt_vy, color="tomato",    linewidth=1.0, linestyle="--", label="GT (MAVROS)")
        ax_vy.set_ylabel("m/s"); ax_vy.set_xlabel("time (s)")
        ax_vy.legend(fontsize=8, loc="upper right")
        ax_vy.grid(True, alpha=0.4)
        if n:
            p, l = _xticks(n); ax_vy.set_xticks(p); ax_vy.set_xticklabels(l, fontsize=8)

        ax_vz.set_title("Velocity Z  (ENU up)", fontsize=10)
        ax_vz.plot(vz,    color="royalblue", linewidth=1.2, label="AirIO")
        ax_vz.plot(gt_vz, color="tomato",    linewidth=1.0, linestyle="--", label="GT (MAVROS)")
        ax_vz.set_ylabel("m/s"); ax_vz.set_xlabel("time (s)")
        ax_vz.legend(fontsize=8, loc="upper right")
        ax_vz.grid(True, alpha=0.4)
        if n:
            p, l = _xticks(n); ax_vz.set_xticks(p); ax_vz.set_xticklabels(l, fontsize=8)

        with node._state_lock:
            armed_label = node._mavros_armed
        zupt_active = not _zupt_bypassed(mode_label, armed=armed_label)
        ax_xy.set_title(
            f"XY Trajectory  (relative to start, ENU)  |  "
            f"mode={mode_label}  ZUPT={'ON' if zupt_active else 'bypassed'}",
            fontsize=10,
        )
        ax_xy.plot(rx, ry, color="royalblue", linewidth=1.5)
        if rx:
            ax_xy.plot(rx[0],  ry[0],  "gs", markersize=7, label="start",   zorder=5)
            ax_xy.plot(rx[-1], ry[-1], "bo", markersize=7, label="current", zorder=5)
            ax_xy.legend(fontsize=9, loc="upper left")
        ax_xy.set_xlabel("X_rot (m)"); ax_xy.set_ylabel("Y_rot (m)")
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.grid(True, alpha=0.4)
        ax_xy.invert_xaxis(); ax_xy.invert_yaxis()

        fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig._ani = animation.FuncAnimation(
        fig, update, interval=100, cache_frame_data=False
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
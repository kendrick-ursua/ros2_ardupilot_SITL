#!/usr/bin/env python3
"""
ROS 2 Jazzy: AirIO inference node — runs independently of ArduPilot EKF.
Publishes estimated velocity and integrated position to custom topics.
ArduPilot flies normally via SITL GPS.

Fixes applied:
  1. rot sliced to [:-1] to match dt shape [N-1]
  2. Gravity NOT subtracted — model expects raw IMU (gravity included)
  3. Gazebo ENU convention: acc.z = +9.81 at rest → only flip Y axis
  4. Orientation quaternion remapped consistently with acc/gyro axis remap

Added:
  5. MAVROS ground truth overlay from /mavros/local_position/odom
     with QoS compatibility (BEST_EFFORT sensor profile)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

import torch
import numpy as np
from collections import deque
from pyhocon import ConfigFactory
import pypose as pp
import math

import sys, os
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation

AIRIO_ROOT = os.path.expanduser("~/Air-IO")
if AIRIO_ROOT not in sys.path:
    sys.path.insert(0, AIRIO_ROOT)

from utils.velocity_integrator import Velocity_Integrator
from model import net_dict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW_SIZE = 200
CONFIG_PATH = "/home/kenders/Air-IO/configs/EuRoC/motion_body_rot.conf"
CKPT_PATH   = "/home/kenders/Air-IO/experiments/euroc/motion_body_rot/ckpt/best_model.ckpt"
DEVICE      = "cuda:0"

# Gazebo /imu @ 1000 Hz → decimate to 200 Hz to match EuRoC training
IMU_HZ       = 1000
TARGET_HZ    = 200
DECIMATE     = IMU_HZ // TARGET_HZ   # = 5
INFER_STRIDE = 8                     # run inference every 10 decimated samples ≈ 20 Hz

# ---------------------------------------------------------------------------
# QoS Profiles
# ---------------------------------------------------------------------------
# MAVROS local_position topics are published with BEST_EFFORT reliability.
# Subscribing with RELIABLE would cause a QoS incompatibility and no messages
# would be received.  Match the publisher profile exactly.
MAVROS_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)

# Standard /imu topics are typically RELIABLE + VOLATILE — use default QoS.
IMU_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class AirIONode(Node):
    def __init__(self):
        super().__init__("airIO_node")

        # --- Load model ---
        conf = ConfigFactory.parse_file(CONFIG_PATH)
        conf["device"] = DEVICE
        self.network = net_dict[conf.train.network](conf.train).to(DEVICE).double()
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
        self.network.load_state_dict(ckpt["model_state_dict"])
        self.network.eval()
        self.get_logger().info(f"AirIO loaded — epoch {ckpt['epoch']}")

        # --- Velocity integrator ---
        self.vel_integrator = Velocity_Integrator(
            pos=torch.zeros(3, dtype=torch.double)
        ).to(DEVICE).double()
        self.vel_integrator.eval()

        # --- Persistent state ---
        self._pos_state           = torch.zeros(3, dtype=torch.double).to(DEVICE)
        self._last_inference_time = None

        # --- Ring buffers (200 samples = 1.0 s at 200 Hz) ---
        self._time_buf   = deque(maxlen=WINDOW_SIZE)
        self._acc_buf    = deque(maxlen=WINDOW_SIZE)
        self._gyro_buf   = deque(maxlen=WINDOW_SIZE)
        self._orient_buf = deque(maxlen=WINDOW_SIZE)
        self._last_stamp = None

        # --- IMU decimation counter ---
        self._imu_count = 0

        # --- AirIO plot buffers ---
        self._plot_lock = threading.Lock()
        self._plot_vx = deque(maxlen=500)
        self._plot_vy = deque(maxlen=500)
        self._plot_vz = deque(maxlen=500)
        self._plot_px = deque(maxlen=500)
        self._plot_py = deque(maxlen=500)
        self._plot_origin = None

        # --- MAVROS ground truth buffers ---
        self._gt_px     = deque(maxlen=500)
        self._gt_py     = deque(maxlen=500)
        self._gt_origin = None   # set on first message to align origins

        # --- Publishers / Subscribers ---
        self.create_subscription(Imu, "/ekf/imu_for_airio", self._imu_ekf_cb, 400) #IMU_QOS
        self.create_subscription(Imu, "/imu", self._imu_cb, IMU_QOS) #IMU_QOS
        self.vel_pub  = self.create_publisher(TwistStamped, "/airio/velocity",  10)
        self.odom_pub = self.create_publisher(Odometry,     "/airio/odometry",  10)

        # MAVROS local position odometry — BEST_EFFORT QoS required
        self.create_subscription(
            Odometry,
            "/mavros/local_position/odom",
            self._mavros_odom_cb,
            MAVROS_QOS,
        )

        self.get_logger().info(
            f"AirIO ready | buffer: {TARGET_HZ} Hz ({WINDOW_SIZE} samples = "
            f"{WINDOW_SIZE / TARGET_HZ:.1f}s) | inference: "
            f"~{TARGET_HZ // INFER_STRIDE} Hz"
        )
        self.get_logger().info(
            "Subscribed to /mavros/local_position/odom with BEST_EFFORT QoS"
        )

    # -----------------------------------------------------------------------
    def _mavros_odom_cb(self, msg: Odometry):
        """
        Receive MAVROS local position odometry and store as ground truth.
        The MAVROS local_position/odom frame is ENU (same as AirIO output),
        so no axis remapping is needed — just track origin offset.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        with self._plot_lock:
            if self._gt_origin is None:
                self._gt_origin = (x, y)
            ox, oy = self._gt_origin
            self._gt_px.append(x - ox)
            self._gt_py.append(y - oy)

    # -----------------------------------------------------------------------
    def _imu_cb(self, msg: Imu):
        self._imu_count += 1

        # Decimate 1000 Hz → 200 Hz
        if self._imu_count % DECIMATE != 0:
            return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._time_buf.append(t)

        self._acc_buf.append([msg.linear_acceleration.x,
                               msg.linear_acceleration.y,
                               msg.linear_acceleration.z])

        self._gyro_buf.append([msg.angular_velocity.x,
                                msg.angular_velocity.y,
                                msg.angular_velocity.z])

        # self._orient_buf.append([msg.orientation.w,
        #                           msg.orientation.x,
        #                           msg.orientation.y,
        #                           msg.orientation.z])
        self._last_stamp = msg.header.stamp

        # Run inference once buffer is full, every INFER_STRIDE decimated samples
        if (len(self._time_buf) >= WINDOW_SIZE and
                (self._imu_count // DECIMATE) % INFER_STRIDE == 0):
            self._run_inference()

    # -----------------------------------------------------------------------
    def _imu_ekf_cb(self, msg: Imu):
        """
        Receive IMU data from ArduPilot EKF node (with gravity removed and
        possibly other preprocessing) for debugging and comparison.
        """
        # For now just log the EKF IMU data to compare with raw IMU input.
        # In the future we could run inference on this data as well to see how
        # it differs from the raw IMU input.
        self._orient_buf.append([msg.orientation.w,
                                  msg.orientation.x,
                                  msg.orientation.y,
                                  msg.orientation.z])
    # -----------------------------------------------------------------------
    def _run_inference(self):
        try:
            times = torch.tensor(list(self._time_buf),   dtype=torch.double)
            acc_t = torch.tensor(list(self._acc_buf),    dtype=torch.double)
            gyr_t = torch.tensor(list(self._gyro_buf),   dtype=torch.double)
            q_t   = torch.tensor(list(self._orient_buf), dtype=torch.double)

            # ------------------------------------------------------------------
            # FIX 3: Gazebo uses ENU (REP-103): X=fwd, Y=left, Z=up.
            #   At rest: acc = [0, 0, +9.81]
            #   EuRoC:   X=fwd, Y=right, Z=down
            #   Mapping: X unchanged, Y flipped, Z flipped
            # ------------------------------------------------------------------
            acc_remap = torch.stack([ acc_t[:, 0],
                                      -acc_t[:, 1],
                                      -acc_t[:, 2]], dim=1)
            gyr_remap = torch.stack([ gyr_t[:, 0],
                                      gyr_t[:, 1],
                                      gyr_t[:, 2]], dim=1)

            acc_body  = acc_remap
            gyro_body = gyr_remap

            # ------------------------------------------------------------------
            # FIX 4: Quaternion axis remap (flip Y and Z via q_flip_x ⊗ q_enu)
            # ------------------------------------------------------------------
            qw = q_t[:, 0]
            qx = q_t[:, 1]
            qy = q_t[:, 2]
            qz = q_t[:, 3]

            qw_r = -qx
            qx_r =  qw
            qy_r =  qz
            qz_r = -qy

            q_remap_xyzw = torch.stack([qx_r, qy_r, qz_r, qw_r], dim=1)
            orientation  = pp.SO3(q_remap_xyzw)

            dt = (times[1:] - times[:-1]).unsqueeze(-1)

            # FIX 1: Slice orientation to [:-1]
            rot = orientation[:-1].Log().tensor().unsqueeze(0).to(DEVICE)

            data = {
                "time": times.unsqueeze(0).to(DEVICE),
                "dt":   dt.unsqueeze(0).to(DEVICE),
                "acc":  acc_body.unsqueeze(0).to(DEVICE),
                "gyro": gyro_body.unsqueeze(0).to(DEVICE),
            }

            with torch.no_grad():
                out = self.network.forward(data, rot)

            vel_euroc_body  = out["net_vel"][0, -1]
            q_last          = orientation[-1].to(DEVICE)
            vel_euroc_world = (q_last @ vel_euroc_body).cpu().numpy()
            vel = np.array([
                 vel_euroc_world[0],
                -vel_euroc_world[1],
                -vel_euroc_world[2],
            ])

            self.get_logger().info(
                f"vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]",
                throttle_duration_sec=1.0
            )

            with self._plot_lock:
                self._plot_vx.append(float(vel[0]))
                self._plot_vy.append(float(vel[1]))
                self._plot_vz.append(float(vel[2]))

            # --- Integrate position ---
            now     = times[-1].item()
            dt_step = float(np.clip(
                now - self._last_inference_time
                    if self._last_inference_time is not None
                    else 1.0 / (TARGET_HZ / INFER_STRIDE),
                0.0, 0.2
            ))
            self._last_inference_time = now

            with torch.no_grad():
                int_state = self.vel_integrator(
                    dt=torch.tensor(
                        [[[dt_step]]], dtype=torch.double
                    ).to(DEVICE),
                    vel=torch.tensor(
                        vel, dtype=torch.double
                    ).unsqueeze(0).unsqueeze(0).to(DEVICE),
                    init_state={"pos": self._pos_state.unsqueeze(0).unsqueeze(0)},
                )

            self._pos_state = int_state["pos"][0, -1].detach()
            pos = self._pos_state.cpu().numpy()

            with self._plot_lock:
                if self._plot_origin is None:
                    self._plot_origin = (float(pos[0]), float(pos[1]))
                ox, oy = self._plot_origin
                self._plot_px.append((float(pos[0]) - ox))
                self._plot_py.append((float(pos[1]) - oy))

            # --- Publish velocity ---
            twist = TwistStamped()
            twist.header.stamp    = self._last_stamp
            twist.header.frame_id = "base_link"
            twist.twist.linear.x  = float(vel[0])
            twist.twist.linear.y  = float(vel[1])
            twist.twist.linear.z  = float(vel[2])
            self.vel_pub.publish(twist)

            # --- Publish odometry ---
            odom = Odometry()
            odom.header.stamp            = self._last_stamp
            odom.header.frame_id         = "odom"
            odom.child_frame_id          = "base_link"
            odom.pose.pose.position.x    = float(pos[0])
            odom.pose.pose.position.y    = float(pos[1])
            odom.pose.pose.position.z    = float(pos[2])
            odom.pose.pose.orientation.w = float(qw_r[-1])
            odom.pose.pose.orientation.x = float(qx_r[-1])
            odom.pose.pose.orientation.y = float(qy_r[-1])
            odom.pose.pose.orientation.z = float(qz_r[-1])
            odom.twist.twist.linear.x    = float(vel[0])
            odom.twist.twist.linear.y    = float(vel[1])
            odom.twist.twist.linear.z    = float(vel[2])
            self.odom_pub.publish(odom)

        except Exception as e:
            self.get_logger().error(
                f"_run_inference error: {e}", throttle_duration_sec=2.0
            )
            import traceback
            self.get_logger().error(
                traceback.format_exc(), throttle_duration_sec=5.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = AirIONode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("AirIO — realtime inference (from /imu)", fontsize=14)
    ax_vx, ax_vy = axes[0]
    ax_vz, ax_xy = axes[1]

    def update(_):
        with node._plot_lock:
            vx   = list(node._plot_vx)
            vy   = list(node._plot_vy)
            vz   = list(node._plot_vz)
            px   = list(node._plot_px)
            py   = list(node._plot_py)
            gt_x = list(node._gt_px)
            gt_y = list(node._gt_py)

        for ax in axes.flat:
            ax.cla()

        ax_vx.set_title("Velocity X (ENU fwd)")
        ax_vx.plot(vx, color="royalblue", linewidth=1.2)
        ax_vx.set_ylabel("m/s"); ax_vx.grid(True, alpha=0.4)

        ax_vy.set_title("Velocity Y (ENU left)")
        ax_vy.plot(vy, color="royalblue", linewidth=1.2)
        ax_vy.set_ylabel("m/s"); ax_vy.grid(True, alpha=0.4)

        ax_vz.set_title("Velocity Z (ENU up)")
        ax_vz.plot(vz, color="royalblue", linewidth=1.2)
        ax_vz.set_ylabel("m/s"); ax_vz.grid(True, alpha=0.4)

        # ------------------------------------------------------------------
        # XY trajectory — AirIO estimate vs MAVROS ground truth
        # Both series are rotated by the same theta so they stay aligned.
        # ------------------------------------------------------------------
        ax_xy.set_title("XY Trajectory — AirIO vs MAVROS GT")

        theta = math.radians(95)   #(-60 - 180) # -60
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def rotate(xs, ys):
            rx = [ x * cos_t + y * sin_t for x, y in zip(xs, ys)]
            ry = [-x * sin_t + y * cos_t for x, y in zip(xs, ys)]
            return rx, ry

        # AirIO estimate
        if px:
            rx, ry = rotate(px, py)
            ax_xy.plot(rx, ry, color="royalblue", linewidth=1.5, label="AirIO")
            ax_xy.plot(rx[-1], ry[-1], "bo", markersize=7)

        # MAVROS ground truth
        if gt_x:
            # gt_rx, gt_ry = rotate(gt_x, gt_y)
            ax_xy.plot(gt_x, gt_y,
                       color="tomato", linewidth=1.5,
                       linestyle="--", label="MAVROS GT")
            ax_xy.plot(gt_x[-1], gt_y[-1], "r^", markersize=7)

        ax_xy.set_xlabel("X_rot (m)")
        ax_xy.set_ylabel("Y_rot (m)")
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.grid(True, alpha=0.4)
        ax_xy.invert_xaxis()
        ax_xy.invert_yaxis()
        ax_xy.legend(loc="upper left", fontsize=8)

        fig.tight_layout()

    fig._ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
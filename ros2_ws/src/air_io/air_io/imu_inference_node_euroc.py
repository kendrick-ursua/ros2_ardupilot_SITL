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
"""
import rclpy
from rclpy.node import Node
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
DEVICE      = "cpu" #cuda:0

# Gazebo /imu @ 1000 Hz → decimate to 400 Hz to match EuRoC training
IMU_HZ       = 1000
TARGET_HZ    = 200
DECIMATE     = IMU_HZ // TARGET_HZ   # = 5
INFER_STRIDE = 8                 # run inference every 2 buffered samples ≈ 100 Hz


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

        # --- Plot buffers ---
        self._plot_lock = threading.Lock()
        self._plot_vx = deque(maxlen=500)
        self._plot_vy = deque(maxlen=500)
        self._plot_vz = deque(maxlen=500)
        self._plot_px = deque(maxlen=500)
        self._plot_py = deque(maxlen=500)
        self._plot_origin = None

        # --- Publishers / Subscribers ---
        self.create_subscription(Imu, "/imu", self._imu_cb, 10)
        self.vel_pub  = self.create_publisher(TwistStamped, "/airio/velocity",  10)
        self.odom_pub = self.create_publisher(Odometry,     "/airio/odometry",  10)

        self.get_logger().info(
            f"AirIO ready | buffer: {TARGET_HZ} Hz ({WINDOW_SIZE} samples = "
            f"{WINDOW_SIZE / TARGET_HZ:.1f}s) | inference: "
            f"~{TARGET_HZ // INFER_STRIDE} Hz"
        )

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
        self._orient_buf.append([msg.orientation.w,
                                  msg.orientation.x,
                                  msg.orientation.y,
                                  msg.orientation.z])
        self._last_stamp = msg.header.stamp

        # Run inference once buffer is full, every INFER_STRIDE decimated samples
        if (len(self._time_buf) >= WINDOW_SIZE and
                (self._imu_count // DECIMATE) % INFER_STRIDE == 0):
            self._run_inference()

    # -----------------------------------------------------------------------
    def _run_inference(self):
        try:
            times = torch.tensor(list(self._time_buf),   dtype=torch.double)
            acc_t = torch.tensor(list(self._acc_buf),    dtype=torch.double)
            gyr_t = torch.tensor(list(self._gyro_buf),   dtype=torch.double)
            q_t   = torch.tensor(list(self._orient_buf), dtype=torch.double)

            # ------------------------------------------------------------------
            # FIX 3: Gazebo uses ENU (REP-103): X=fwd, Y=left, Z=up.
            #   At rest: acc = [0, 0, +9.81]  (specific force, reaction to gravity)
            #   EuRoC:   X=fwd, Y=right, Z=down
            #   Mapping: X unchanged, Y flipped, Z flipped
            # ------------------------------------------------------------------
            acc_remap = torch.stack([ acc_t[:, 0],
                                      -acc_t[:, 1],
                                      -acc_t[:, 2]], dim=1)   # [N, 3]
            gyr_remap = torch.stack([ gyr_t[:, 0],
                                      gyr_t[:, 1],
                                      gyr_t[:, 2]], dim=1)   # [N, 3]

            # ------------------------------------------------------------------
            # FIX 2: Do NOT subtract gravity.
            #   Air-IO is trained on raw EuRoC IMU data, which includes gravity
            #   (acc.z ≈ +9.81 at rest in EuRoC frame).  The network handles
            #   gravity internally via the rotation input.  Subtracting gravity
            #   here puts the model out-of-distribution.
            # ------------------------------------------------------------------
            acc_body  = acc_remap   # raw remapped — gravity stays in
            gyro_body = gyr_remap

            # ------------------------------------------------------------------
            # FIX 4: Remap orientation quaternion axes consistently with acc/gyro.
            #   Gazebo orientation: [w, x, y, z] in ENU frame.
            #   After axis remap (flip Y and Z), reconstruct the quaternion so
            #   it represents the same physical attitude in EuRoC frame.
            #   For a 180° rotation about X that maps (Y,Z)→(-Y,-Z):
            #     q_euroc = q_flip_x ⊗ q_enu
            #   where q_flip_x = [0, 1, 0, 0]  (w=0, x=1, y=0, z=0)
            # ------------------------------------------------------------------
            # Unpack ENU quaternion components
            qw = q_t[:, 0]
            qx = q_t[:, 1]
            qy = q_t[:, 2]
            qz = q_t[:, 3]

            # q_flip_x ⊗ q_enu  (Hamilton product, q_flip = [w=0,x=1,y=0,z=0])
            qw_r =  0*qw - 1*qx - 0*qy - 0*qz   # = -qx
            qx_r =  0*qx + 1*qw + 0*qz - 0*qy   # = +qw  (corrected sign)
            qy_r =  0*qy - 0*qz + 1*qx + 0*qw   # wait — let's use explicit form:
            # Hamilton: (a⊗b) where a=[0,1,0,0]:
            #   w' =  0*qw - 1*qx - 0*qy - 0*qz = -qx
            #   x' =  0*qx + 1*qw + 0*qz - 0*qy =  qw
            #   y' =  0*qy - 0*qz + 1*(-qz)...  — use matrix form below
            # Use the clean matrix form:
            qw_r = -qx
            qx_r =  qw
            qy_r =  qz
            qz_r = -qy

            # Stack remapped quaternion [w, x, y, z] → reorder to [x, y, z, w] for pypose
            q_remap_xyzw = torch.stack([qx_r, qy_r, qz_r, qw_r], dim=1)  # [N, 4]
            orientation  = pp.SO3(q_remap_xyzw)

            dt = (times[1:] - times[:-1]).unsqueeze(-1)   # [N-1, 1]

            # ------------------------------------------------------------------
            # FIX 1: Slice orientation to [:-1] so rot shape = [1, N-1, 3],
            #   matching dt shape [1, N-1, 1].
            #   Upstream inference_motion.py: rot = label['gt_rot'][:,:-1,:].Log()
            # ------------------------------------------------------------------
            rot = orientation[:-1].Log().tensor().unsqueeze(0).to(DEVICE)  # [1, N-1, 3]

            data = {
                "time": times.unsqueeze(0).to(DEVICE),            # [1, N]
                "dt":   dt.unsqueeze(0).to(DEVICE),               # [1, N-1, 1]
                "acc":  acc_body.unsqueeze(0).to(DEVICE),         # [1, N, 3]
                "gyro": gyro_body.unsqueeze(0).to(DEVICE),        # [1, N, 3]
            }

            with torch.no_grad():
                out = self.network.forward(data, rot)

            # ------------------------------------------------------------------
            # Network output is body-frame velocity in EuRoC coords (the model
            # is trained with coordinate: body_coord).
            # Must rotate body→world BEFORE remapping back to ENU, otherwise
            # integrating body-frame velocity directly produces a straight line.
            #   orientation[-1]: SO3 q_W_B  (body → world, EuRoC frame)
            #   v_world_euroc = R_W_B @ v_body_euroc
            # Then reverse the Y/Z flip to get ENU world-frame velocity.
            # ------------------------------------------------------------------
            vel_euroc_body  = out["net_vel"][0, -1]                    # [3] CUDA, body EuRoC
            q_last          = orientation[-1].to(DEVICE)               # SO3  q_W_B, EuRoC
            vel_euroc_world = (q_last @ vel_euroc_body).cpu().numpy()  # [3] world EuRoC
            vel = np.array([
                vel_euroc_world[0],    # X — unchanged
                -vel_euroc_world[1],    # Y — flip back to ENU
                -vel_euroc_world[2],    # Z — flip back to ENU
            ])

            self.get_logger().info(
                f"vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]",
                throttle_duration_sec=1.0
            )

            # --- Update plot buffers ---
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
            pos = self._pos_state.cpu().numpy()   # [3]

            with self._plot_lock:
                if self._plot_origin is None:
                    self._plot_origin = (float(pos[0]), float(pos[1]))
                ox, oy = self._plot_origin
                self._plot_px.append(float(pos[0]) - ox)
                self._plot_py.append(float(pos[1]) - oy)

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
            # Publish remapped orientation in odom (EuRoC frame)
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
            vx = list(node._plot_vx)
            vy = list(node._plot_vy)
            vz = list(node._plot_vz)
            px = list(node._plot_px)
            py = list(node._plot_py)

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

        ax_xy.set_title("XY Trajectory (relative to start)")

        #### DEFAULT ORIENTATION ####
        # ax_xy.plot(px, py, color="royalblue", linewidth=1.5)
        # if px:
        #     ax_xy.plot(px[-1], py[-1], "bo", markersize=7)
        # ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")

        #### REMAPPED ORIENTATION (flip Y for ENU) ####
        # ax_xy.plot([-y for y in py], px, color="royalblue", linewidth=1.5)
        # if px:
        #     ax_xy.plot(-py[-1], px[-1], "bo", markersize=7)
        # ax_xy.set_xlabel("-Y (m)"); ax_xy.set_ylabel("X (m)")

        # 45° clockwise rotation
        theta = math.radians(-60)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # Rotate each (px, py) point
        rx = [ x * cos_t + y * sin_t for x, y in zip(px, py)]
        ry = [-x * sin_t + y * cos_t for x, y in zip(px, py)]
        ax_xy.plot(rx, ry, color="royalblue", linewidth=1.5)
        if rx:
            ax_xy.plot(rx[-1], ry[-1], "bo", markersize=7)
        ax_xy.set_xlabel("X_rot (m)")
        ax_xy.set_ylabel("Y_rot (m)")

        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.grid(True, alpha=0.4)
        ax_xy.invert_xaxis()  # ENU: X forward, Y left → invert X for typical view
        ax_xy.invert_yaxis()  # ENU: Y left, Z up → invert Y for typical view

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
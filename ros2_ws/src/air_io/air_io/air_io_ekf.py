#!/usr/bin/env python3
"""
airio_ekf_ros2_node.py
======================
Direct ROS2 port of Air-IO/EKF/IMUofflinerunner.py

Subscribes:
  /imu/airimu_corrected   sensor_msgs/Imu
      └─ linear_acceleration  →  corrected acc  (â_t)
      └─ angular_velocity     →  corrected gyro (ŵ_t)
      └─ linear_acceleration_covariance[0,4,8]  → acc_cov  diagonal
      └─ angular_velocity_covariance[0,4,8]     → gyro_cov diagonal

  /imu/airimu_cov         std_msgs/Float64MultiArray
      └─ data[0:3]  → gyro_cov  (η̂_t^g)  — overrides IMU msg cov if present
      └─ data[3:6]  → acc_cov   (η̂_t^a)

  /airio/velocity         geometry_msgs/TwistStamped
      └─ twist.linear.{x,y,z} → net_vel (body frame)

  /airio/velocity_cov     std_msgs/Float64MultiArray
      └─ data[0:3]  → velocity measurement covariance

Publishes:
  /ekf/odom               nav_msgs/Odometry     pos + rot + vel + cov
  /ekf/pose               geometry_msgs/PoseStamped
  /ekf/twist              geometry_msgs/TwistStamped  (world frame)
  /ekf/bias               std_msgs/Float64MultiArray  [bg(3) ba(3)]
  /ekf/state_raw          std_msgs/Float64MultiArray  15-element EKF state

Usage:
  python3 airio_ekf_ros2_node.py \\
      --airio_root ~/Air-IO \\
      --init_rot   0 0 0 \\
      --init_vel   0 0 0 \\
      --init_pos   0 0 0 \\
      --gravity    9.81007 \\
      --bias_weight    1e-12 \\
      --input_weight   1e2 \\
      --obs_weight     1e-1
"""

import os
import sys
import argparse
import threading

import numpy as np
import torch
import pypose as pp

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64MultiArray


def stamp2sec(s) -> float:
    return s.sec + s.nanosec * 1e-9


# ─────────────────────────────────────────────────────────────────────────────
#  ROS2 Node
# ─────────────────────────────────────────────────────────────────────────────

class AirIOEKFNode(Node):
    """
    Exact online equivalent of the offline IMUofflinerunner EKF loop.

    State layout (15-dim, matching SingleIMU / EKF_runner):
      [0:3]   rot   — so(3) tangent  (Log of rotation)
      [3:6]   vel   — world frame (m/s)
      [6:9]   pos   — world frame (m)
      [9:12]  bg    — gyro  bias (rad/s)
      [12:15] ba    — accel bias (m/s²)
    """

    def __init__(self, args):
        super().__init__("airio_ekf")

        # ── Load Air-IO EKF classes ───────────────────────────────────────────
        airio_root = os.path.expanduser(args.airio_root)
        ekf_root   = os.path.join(airio_root, "EKF")
        for p in [airio_root, ekf_root]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from EKF.IMUofflinerunner import SingleIMU, EKF_runner  # noqa: E402

        # ── EKF weights (exactly as in offline script) ────────────────────────
        self.bias_weight  = args.bias_weight   # 1e-12
        self.input_weight = args.input_weight  # 1e2
        self.obs_weight   = args.obs_weight    # 1e-1

        # ── Gravity vector ────────────────────────────────────────────────────
        self.gravity = torch.tensor([0.0, 0.0, -args.gravity], dtype=torch.float64)

        # ── Build EKF ─────────────────────────────────────────────────────────
        model     = SingleIMU().double()
        self.ekf  = EKF_runner(model)

        # Initial state vector (15-dim)
        init_rot = torch.tensor(args.init_rot, dtype=torch.float64)   # so3 log
        init_vel = torch.tensor(args.init_vel, dtype=torch.float64)
        init_pos = torch.tensor(args.init_pos, dtype=torch.float64)

        initial_state = torch.zeros(15, dtype=torch.float64)
        initial_state[0:3]  = init_rot
        initial_state[3:6]  = init_vel
        initial_state[6:9]  = init_pos
        # biases start at zero (same as offline script)
        self.ekf.state = initial_state

        self.get_logger().info(
            f"[AirIO-EKF] Initialised\n"
            f"  init_rot={args.init_rot}  init_vel={args.init_vel}  init_pos={args.init_pos}\n"
            f"  bias_weight={self.bias_weight}  input_weight={self.input_weight}  "
            f"obs_weight={self.obs_weight}"
        )

        # ── Runtime state ─────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._last_imu_stamp: float | None = None

        # Latest AirIMU covariance (from /imu/airimu_cov)
        self._gyro_cov: torch.Tensor | None = None   # (3,)
        self._acc_cov:  torch.Tensor | None = None   # (3,)

        # Latest AirIO velocity observation + its covariance
        self._pending_vel:     torch.Tensor | None = None   # (3,)
        self._pending_vel_cov: torch.Tensor | None = None   # (3,)
        self._pending_vel_stamp: float = -1.0

        self.frame_id       = args.frame_id
        self.child_frame_id = args.child_frame_id

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Imu, "/imu/airimu_corrected", self._imu_cb, 1000)
        self.create_subscription(
            Float64MultiArray, "/imu/airimu_cov", self._airimu_cov_cb, 400)
        self.create_subscription(
            TwistStamped, "/airio/velocity", self._vel_cb, 50)
        self.create_subscription(
            Float64MultiArray, "/airio/velocity_cov", self._vel_cov_cb, 400)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_odom    = self.create_publisher(Odometry,          "/ekf/odom",               10)
        self.pub_pose    = self.create_publisher(PoseStamped,       "/ekf/pose",               10)
        self.pub_twist   = self.create_publisher(TwistStamped,      "/ekf/twist",              10)
        self.pub_bias    = self.create_publisher(Float64MultiArray, "/ekf/bias",               10)
        self.pub_raw     = self.create_publisher(Float64MultiArray, "/ekf/state_raw",          10)
        # ── AirIO inputs ──────────────────────────────────────────────────────
        # R̂_t as 9-float row-major 3×3 matrix  → feed to AirIO as 'rot'
        self.pub_rotmat  = self.create_publisher(Float64MultiArray, "/ekf/orientation_matrix", 10)
        # Filtered IMU pass-through → feed to AirIO as 'acc' / 'gyro'
        self.pub_imu_mux = self.create_publisher(Imu,               "/ekf/imu_for_airio",      10)

        # Cache the last filtered IMU msg so we can re-publish it after EKF
        self._last_imu_msg: Imu | None = None

    # ── /imu/airimu_cov  →  η̂_t^g, η̂_t^a ────────────────────────────────────

    def _airimu_cov_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 6:
            with self._lock:
                self._gyro_cov = torch.tensor(msg.data[0:3], dtype=torch.float64)
                self._acc_cov  = torch.tensor(msg.data[3:6], dtype=torch.float64)

    # ── /airio/velocity  →  net_vel observation ───────────────────────────────

    def _vel_cb(self, msg: TwistStamped):
        with self._lock:
            self._pending_vel = torch.tensor(
                [msg.twist.linear.x,
                 msg.twist.linear.y,
                 msg.twist.linear.z], dtype=torch.float64)
            self._pending_vel_stamp = stamp2sec(msg.header.stamp)

    def _vel_cov_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 3:
            with self._lock:
                self._pending_vel_cov = torch.tensor(
                    msg.data[0:3], dtype=torch.float64)

    # ── /imu/airimu_corrected  →  EKF predict + update ───────────────────────

    def _imu_cb(self, msg: Imu):
        stamp = stamp2sec(msg.header.stamp)

        acc  = torch.tensor([msg.linear_acceleration.x,
                              msg.linear_acceleration.y,
                              msg.linear_acceleration.z], dtype=torch.float64)
        gyro = torch.tensor([msg.angular_velocity.x,
                              msg.angular_velocity.y,
                              msg.angular_velocity.z], dtype=torch.float64)

        # dt
        with self._lock:
            if self._last_imu_stamp is None:
                self._last_imu_stamp = stamp
                return
            dt = stamp - self._last_imu_stamp
            if dt <= 0.0 or dt > 1.0:
                self._last_imu_stamp = stamp
                return
            self._last_imu_stamp = stamp

            # ── Grab covariances ───────────────────────────────────────────
            # AirIMU covariance from /imu/airimu_cov (preferred)
            # Fallback: read diagonal from Imu msg covariance fields
            if self._gyro_cov is not None:
                gyro_cov = self._gyro_cov.clone()
            else:
                gc = msg.angular_velocity_covariance
                gyro_cov = torch.tensor([gc[0], gc[4], gc[8]], dtype=torch.float64)
                if gyro_cov.sum() == 0:
                    gyro_cov = torch.full((3,), 1e-4, dtype=torch.float64)

            if self._acc_cov is not None:
                acc_cov = self._acc_cov.clone()
            else:
                ac = msg.linear_acceleration_covariance
                acc_cov = torch.tensor([ac[0], ac[4], ac[8]], dtype=torch.float64)
                if acc_cov.sum() == 0:
                    acc_cov = torch.full((3,), 1e-2, dtype=torch.float64)

            # ── Q matrix — exactly as offline script ──────────────────────
            # Offline: data["gyro_cov"][0] has shape (T=1, 3)
            # q[:3]  = gyro_cov  (T=1, 3) → assigned into q[0:3] which is (3,)
            # The offline script assigns a (1,3) slice into q[:3] (shape (3,))
            # which works because PyTorch broadcasts the single row.
            # We replicate by keeping cov as (3,) scalars (already done above).
            q = torch.ones(12, dtype=torch.float64) * self.bias_weight
            q[0:3] = gyro_cov.squeeze()   # ensure (3,) — squeeze any extra dim
            q[3:6] = acc_cov.squeeze() * self.input_weight
            Q = torch.diag(q)             # (12×12)

            # ── Observation — match by timestamp (20 ms realtime window) ──
            # Offline: observation = io_result["net_vel"][io_index]  shape (3,)
            observation = None
            R_meas      = torch.eye(3, dtype=torch.float64) * 0.001

            if (self._pending_vel is not None and
                    abs(self._pending_vel_stamp - stamp) < 0.02):
                observation = self._pending_vel.clone()   # (3,)

                if self._pending_vel_cov is not None:
                    r_diag = self._pending_vel_cov.clone().squeeze() * self.obs_weight
                else:
                    r_diag = torch.full((3,), 0.001, dtype=torch.float64)

                R_meas = torch.diag(r_diag)
                self._pending_vel = None   # consume

        # Cache filtered IMU msg for re-publishing alongside R̂_t
        with self._lock:
            self._last_imu_msg = msg

        # ── imu_data dict ─────────────────────────────────────────────────
        # EKF_runner.propogate_update/state does:
        #   input = torch.cat([gyro, acc, d_bias_gyro, d_bias_acc], dim=-1)
        # d_bias_* = self.state[9:12]  →  shape (3,)  flat 1-D.
        # ALL tensors in the cat must be same ndim → gyro/acc must be (3,) flat.
        # dt is used as a scalar multiplier inside state_transition: w * dt.
        # Inside EKF_runner.propogate_update/state:
        #   input = torch.cat([gyro, acc, d_bias_gyro, d_bias_acc], dim=-1)
        # where d_bias_* = self.state[9:12] shape (3,).
        # cat requires ALL tensors to be same ndim → must be (3,), not (1,3).
        # dt is used as scalar multiplier: w * dt  inside state_transition.
        imu_data = {
            "gyro": gyro,                                        # (3,)  flat
            "acc":  acc,                                         # (3,)  flat
            "dt":   torch.tensor(dt, dtype=torch.float64),      # scalar ()
        }

        # ── Run EKF step ──────────────────────────────────────────────────
        with self._lock:
            try:
                self.ekf.run(imu_data, observation=observation, Q=Q, R=R_meas)
            except RuntimeError as e:
                # Print exact shapes on dimension mismatch to aid debugging
                self.get_logger().error(
                    f"EKF run error: {e}\n"
                    f"  imu_data shapes — gyro:{imu_data['gyro'].shape}  "
                    f"acc:{imu_data['acc'].shape}  dt:{imu_data['dt'].shape}\n"
                    f"  Q:{Q.shape}  R:{R_meas.shape}  "
                    f"obs:{observation.shape if observation is not None else None}\n"
                    f"  gyro_cov:{gyro_cov.shape}  acc_cov:{acc_cov.shape}"
                )
                return
            except Exception as e:
                self.get_logger().error(f"EKF run error: {e}")
                return

            state = self.ekf.state.clone()   # (15,)

        self._publish(state, msg.header.stamp)

    # ── Publish ───────────────────────────────────────────────────────────────

    def _publish(self, state: torch.Tensor, stamp):
        """
        State layout (matching offline script):
          [0:3]   rot  so(3) log  → convert to quaternion via pp.so3().Exp()
          [3:6]   vel  world frame
          [6:9]   pos  world frame
          [9:12]  bg   gyro bias
          [12:15] ba   accel bias
        """
        rot_so3 = state[0:3]
        vel     = state[3:6].numpy()
        pos     = state[6:9].numpy()
        bg      = state[9:12].numpy()
        ba      = state[12:15].numpy()

        # so3 log → quaternion (w, x, y, z)  AND  3×3 rotation matrix
        rot_quat = pp.so3(rot_so3).Exp()   # pp.SO3
        q = rot_quat.numpy()               # [w, x, y, z]

        # 3×3 rotation matrix  R̂_t  (body → world)
        w, x, y, z = q
        R_mat = np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])

        # ── Odometry ──────────────────────────────────────────────────────
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id  = self.child_frame_id

        odom.pose.pose.position.x    = float(pos[0])
        odom.pose.pose.position.y    = float(pos[1])
        odom.pose.pose.position.z    = float(pos[2])
        odom.pose.pose.orientation.w = float(q[0])
        odom.pose.pose.orientation.x = float(q[1])
        odom.pose.pose.orientation.y = float(q[2])
        odom.pose.pose.orientation.z = float(q[3])

        odom.twist.twist.linear.x = float(vel[0])
        odom.twist.twist.linear.y = float(vel[1])
        odom.twist.twist.linear.z = float(vel[2])

        # Covariance from EKF — try to get it if EKF_runner exposes it
        try:
            _, cov = self.ekf.get_result()
            if cov is not None and len(cov) > 0:
                last_cov = cov[-1]                    # (15, 15) or similar
                # Pose covariance (pos + rot) → 6×6 row-major
                pcov = np.zeros(36)
                pcov[0]  = float(last_cov[6, 6])     # pos x
                pcov[7]  = float(last_cov[7, 7])     # pos y
                pcov[14] = float(last_cov[8, 8])     # pos z
                pcov[21] = float(last_cov[0, 0])     # rot x
                pcov[28] = float(last_cov[1, 1])     # rot y
                pcov[35] = float(last_cov[2, 2])     # rot z
                odom.pose.covariance = pcov.tolist()

                vcov = np.zeros(36)
                vcov[0]  = float(last_cov[3, 3])
                vcov[7]  = float(last_cov[4, 4])
                vcov[14] = float(last_cov[5, 5])
                odom.twist.covariance = vcov.tolist()
        except Exception:
            pass   # covariance is optional

        self.pub_odom.publish(odom)

        # ── PoseStamped ───────────────────────────────────────────────────
        ps = PoseStamped()
        ps.header = odom.header
        ps.pose   = odom.pose.pose
        self.pub_pose.publish(ps)

        # ── TwistStamped ──────────────────────────────────────────────────
        tw = TwistStamped()
        tw.header = odom.header
        tw.twist.linear.x = float(vel[0])
        tw.twist.linear.y = float(vel[1])
        tw.twist.linear.z = float(vel[2])
        self.pub_twist.publish(tw)

        # ── Bias ──────────────────────────────────────────────────────────
        bias_msg = Float64MultiArray()
        bias_msg.data = (bg.tolist() + ba.tolist())   # [bg_x bg_y bg_z ba_x ba_y ba_z]
        self.pub_bias.publish(bias_msg)

        # ── Raw state (15-element) ─────────────────────────────────────────
        raw_msg = Float64MultiArray()
        raw_msg.data = state.numpy().tolist()
        self.pub_raw.publish(raw_msg)

        # ── R̂_t  — rotation matrix for AirIO 'rot' input ─────────────────
        # 9 floats, row-major 3×3  (body → world)
        rotmat_msg = Float64MultiArray()
        rotmat_msg.data = R_mat.flatten().tolist()
        self.pub_rotmat.publish(rotmat_msg)

        # ── /ekf/imu_for_airio  — filtered IMU + updated orientation ──────
        # Re-publish the latest AirIMU-corrected Imu msg but with the EKF
        # quaternion stamped in the orientation field so AirIO always receives
        # a consistent (IMU data, R̂_t) pair on a single topic.
        with self._lock:
            last_imu = self._last_imu_msg
        if last_imu is not None:
            mux = Imu()
            mux.header                         = last_imu.header
            mux.linear_acceleration            = last_imu.linear_acceleration
            mux.linear_acceleration_covariance = last_imu.linear_acceleration_covariance
            mux.angular_velocity               = last_imu.angular_velocity
            mux.angular_velocity_covariance    = last_imu.angular_velocity_covariance
            # Inject EKF orientation (R̂_t as quaternion)
            mux.orientation.w = float(q[0])
            mux.orientation.x = float(q[1])
            mux.orientation.y = float(q[2])
            mux.orientation.z = float(q[3])
            mux.orientation_covariance[0]  = float(R_mat[0, 0])
            mux.orientation_covariance[1]  = float(R_mat[0, 1])
            mux.orientation_covariance[2]  = float(R_mat[0, 2])
            mux.orientation_covariance[3]  = float(R_mat[1, 0])
            mux.orientation_covariance[4]  = float(R_mat[1, 1])
            mux.orientation_covariance[5]  = float(R_mat[1, 2])
            mux.orientation_covariance[6]  = float(R_mat[2, 0])
            mux.orientation_covariance[7]  = float(R_mat[2, 1])
            mux.orientation_covariance[8]  = float(R_mat[2, 2])
            self.pub_imu_mux.publish(mux)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Air-IO EKF ROS2 node (direct port of IMUofflinerunner)")

    parser.add_argument("--airio_root", default="~/Air-IO",
                        help="Path to Air-IO repo root")

    # Initial state
    parser.add_argument("--init_rot", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help="Initial rotation as so(3) log [rx ry rz] (default identity)")
    parser.add_argument("--init_vel", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help="Initial velocity in world frame [vx vy vz] (m/s)")
    parser.add_argument("--init_pos", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help="Initial position in world frame [px py pz] (m)")

    # EKF weights — same names and defaults as offline script
    parser.add_argument("--gravity",      type=float, default=9.81007)
    parser.add_argument("--bias_weight",  type=float, default=1e-12,
                        help="Q diagonal for bias states (default 1e-12)")
    parser.add_argument("--input_weight", type=float, default=1e2,
                        help="Scaling on acc_cov in Q (default 1e2)")
    parser.add_argument("--obs_weight",   type=float, default=1e-1,
                        help="Scaling on AirIO vel_cov in R (default 1e-1)")

    # Frames
    parser.add_argument("--frame_id",       default="odom")
    parser.add_argument("--child_frame_id", default="base_link")

    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = AirIOEKFNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
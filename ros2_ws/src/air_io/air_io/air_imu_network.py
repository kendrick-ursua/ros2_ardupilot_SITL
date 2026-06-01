"""
AirIMU ROS2 Realtime Inference Node
====================================
Subscribes to /imu (sensor_msgs/Imu), buffers a sliding window of IMU data,
runs AirIMU correction inference, and publishes corrected IMU + covariance.

Topics:
  SUB  /imu                  sensor_msgs/Imu
  PUB  /imu/airimu_corrected sensor_msgs/Imu   (bias-corrected accel & gyro)
  PUB  /imu/airimu_cov       std_msgs/Float64MultiArray  (6-element cov vector)

Usage:
  python3 airimu_ros2_node.py \
      --config /path/to/AirIMU/configs/exp/EuRoC/codenet.conf \
      --ckpt   /path/to/best_model.ckpt \
      --window 200 \
      --device cpu
"""

import sys
import os
import argparse
import threading
import collections
import time

import numpy as np
import torch

# ── ROS2 imports ──────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Time as RosTime


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def load_airimu(config_path: str, ckpt_path: str, device: str):
    """Load AirIMU network from config + checkpoint."""
    # AirIMU must be on PYTHONPATH (or installed)
    from pyhocon import ConfigFactory
    from model import net_dict

    conf = ConfigFactory.parse_file(config_path)
    conf.train.device = device

    network = net_dict[conf.train.network](conf.train).to(device).double()
    network.eval()

    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    network.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", "?")
    print(f"[AirIMU] Loaded checkpoint (epoch {epoch}) from {ckpt_path}")
    return network, conf


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────────────────────

class AirIMUNode(Node):

    def __init__(self):
        super().__init__("airimu_realtime")

        home_dir = os.path.expanduser('~')

        self.declare_parameter('device', "cpu")
        self.declare_parameter('window', 200)
        self.declare_parameter('step', 1)
        self.declare_parameter('topic', "/imu")
        self.declare_parameter('config', f"{home_dir}/ros2_ardupilot_SITL/AirIMU/configs/exp/EuRoC/codenet.conf")
        self.declare_parameter('ckpt', f"{home_dir}/ros2_ardupilot_SITL/AirIMU/EuRoCWholeaug/ckpt/best_model.ckpt")
        self.declare_parameter('airimu_path', f"{home_dir}/ros2_ardupilot_SITL/AirIMU")

        self.window_size = self.get_parameter('window').get_parameter_value().integer_value
        self.imu_topic = self.get_parameter('topic').get_parameter_value().string_value
        self.step_size = self.get_parameter('step').get_parameter_value().integer_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        airimu_path = self.get_parameter('airimu_path').get_parameter_value().string_value
        if airimu_path:
            sys.path.insert(0, os.path.abspath(airimu_path))

        # ── Load model ────────────────────────────────────────────────────────
        self.get_logger().info("Loading AirIMU model …")
        self.network, self.conf = load_airimu(
            self.get_parameter('config').get_parameter_value().string_value,
            self.get_parameter('ckpt').get_parameter_value().string_value,
            self.get_parameter('device').get_parameter_value().string_value
        )
        self.get_logger().info("Model ready.")

        # ── Buffers ───────────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._acc_buf  = collections.deque(maxlen=self.window_size)   # (3,) each
        self._gyro_buf = collections.deque(maxlen=self.window_size)
        self._dt_buf   = collections.deque(maxlen=self.window_size)
        self._stamp_buf = collections.deque(maxlen=self.window_size)
        self._last_stamp: float | None = None
        self._since_last_infer: int = 0

        # ── ROS subscribers / publishers ──────────────────────────────────────
        self.sub_imu = self.create_subscription(
            Imu, self.imu_topic, self._imu_cb, 1000
        )
        self.pub_corrected = self.create_publisher(
            Imu, "/imu/airimu_corrected", 1000
        )
        self.pub_cov = self.create_publisher(
            Float64MultiArray, "/imu/airimu_cov", 1000
        )

        self.get_logger().info(
            f"Subscribed to '{self.imu_topic}' | window={self.window_size} "
            f"step={self.step_size} device={self.device}"
        )

    # ── IMU callback ──────────────────────────────────────────────────────────

    def _imu_cb(self, msg: Imu):
        stamp = stamp_to_sec(msg.header.stamp)
        a = msg.linear_acceleration
        g = msg.angular_velocity

        acc  = np.array([a.x, a.y, a.z], dtype=np.float64)
        gyro = np.array([g.x, g.y, g.z], dtype=np.float64)

        with self._lock:
            if self._last_stamp is not None:
                dt = stamp - self._last_stamp
                if dt <= 0:
                    dt = 1.0 / 200.0   # fall back to 200 Hz default
            else:
                dt = 1.0 / 200.0

            self._last_stamp = stamp
            self._acc_buf.append(acc)
            self._gyro_buf.append(gyro)
            self._dt_buf.append(dt)
            self._stamp_buf.append(stamp)
            self._since_last_infer += 1

            if (len(self._acc_buf) == self.window_size and
                    self._since_last_infer >= self.step_size):
                self._since_last_infer = 0
                # snapshot current buffers for inference (no copies needed for deque)
                acc_arr  = np.array(self._acc_buf,  dtype=np.float64)   # (W, 3)
                gyro_arr = np.array(self._gyro_buf, dtype=np.float64)
                dt_arr   = np.array(self._dt_buf,   dtype=np.float64)   # (W,)
                stamp_arr = list(self._stamp_buf)
                # run inference in same thread (fast enough for CPU/GPU)
                self._run_inference(acc_arr, gyro_arr, dt_arr, stamp_arr, msg)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference(self, acc_arr, gyro_arr, dt_arr, stamps, latest_msg: Imu):
        """
        Build the data dict AirIMU expects, run network.inference(), publish.

        AirIMU's network.inference() expects a dict with keys:
          'acc'     : Tensor (1, W, 3)  double
          'gyro'    : Tensor (1, W, 3)  double
          'dt'      : Tensor (1, W, 1)  double
        and returns an inte_state dict with at least:
          'correction_acc'  : Tensor  (1, W, 3)
          'correction_gyro' : Tensor  (1, W, 3)
          'acc_cov'         : Tensor  (1, W, 3)  (may be absent)
          'gyro_cov'        : Tensor  (1, W, 3)  (may be absent)
        """
        W = len(acc_arr)

        acc_t  = torch.from_numpy(acc_arr).unsqueeze(0).to(self.device)    # (1,W,3)
        gyro_t = torch.from_numpy(gyro_arr).unsqueeze(0).to(self.device)
        dt_t   = torch.from_numpy(dt_arr).unsqueeze(0).unsqueeze(-1).to(self.device)  # (1,W,1)

        data = {
            "acc":  acc_t,
            "gyro": gyro_t,
            "dt":   dt_t,
        }

        t0 = time.perf_counter()
        with torch.no_grad():
            try:
                inte_state = self.network.inference(data)
            except Exception as e:
                self.get_logger().error(f"AirIMU inference error: {e}")
                return
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # ── Extract the LAST sample's correction (most recent) ────────────────
        corr_acc  = inte_state["correction_acc"].squeeze(0)[-1].cpu().numpy()   # (3,)
        corr_gyro = inte_state["correction_gyro"].squeeze(0)[-1].cpu().numpy()

        acc_cov  = (inte_state.get("acc_cov",  None))
        gyro_cov = (inte_state.get("gyro_cov", None))

        if acc_cov is not None:
            acc_cov_val  = acc_cov.squeeze(0)[-1].cpu().numpy()
        else:
            acc_cov_val  = np.zeros(3)
        if gyro_cov is not None:
            gyro_cov_val = gyro_cov.squeeze(0)[-1].cpu().numpy()
        else:
            gyro_cov_val = np.zeros(3)

        # corrected measurement = raw + correction  (AirIMU convention)
        raw_acc  = acc_arr[-1]
        raw_gyro = gyro_arr[-1]
        c_acc    = raw_acc  + corr_acc
        c_gyro   = raw_gyro + corr_gyro

        # ── Publish corrected IMU ─────────────────────────────────────────────
        out_msg = Imu()
        out_msg.header = latest_msg.header
        out_msg.orientation = latest_msg.orientation
        out_msg.orientation_covariance = latest_msg.orientation_covariance

        out_msg.linear_acceleration.x = float(c_acc[0])
        out_msg.linear_acceleration.y = float(c_acc[1])
        out_msg.linear_acceleration.z = float(c_acc[2])

        out_msg.angular_velocity.x = float(c_gyro[0])
        out_msg.angular_velocity.y = float(c_gyro[1])
        out_msg.angular_velocity.z = float(c_gyro[2])

        # Diagonal covariance from AirIMU uncertainty (var not std, fill diagonal)
        acov = acc_cov_val.tolist()
        gcov = gyro_cov_val.tolist()
        # ROS Imu uses row-major 3x3 — fill diagonal
        out_msg.linear_acceleration_covariance[0] = acov[0]
        out_msg.linear_acceleration_covariance[4] = acov[1]
        out_msg.linear_acceleration_covariance[8] = acov[2]
        out_msg.angular_velocity_covariance[0] = gcov[0]
        out_msg.angular_velocity_covariance[4] = gcov[1]
        out_msg.angular_velocity_covariance[8] = gcov[2]

        self.pub_corrected.publish(out_msg)

        # ── Publish raw covariance vector ─────────────────────────────────────
        cov_msg = Float64MultiArray()
        cov_msg.data = (acov + gcov)   # [ax, ay, az, gx, gy, gz]
        self.pub_cov.publish(cov_msg)

        self.get_logger().debug(
            f"Inference {elapsed_ms:.1f} ms | "
            f"corr_acc={corr_acc} corr_gyro={corr_gyro}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AirIMUNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

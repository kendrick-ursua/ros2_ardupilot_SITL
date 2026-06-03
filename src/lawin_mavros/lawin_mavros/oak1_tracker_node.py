#!/usr/bin/env python3
"""
LawinTracker OAK-1 Tracker Node  —  Hailo-8 Edition
=====================================================
Hardware : Raspberry Pi 5 + Hailo-8 AI Hat+
OS       : Ubuntu 24.04 (not RPi OS)
ROS2     : Jazzy
HailoRT  : 4.23.0

── INSTALL GUIDE ──────────────────────────────────────────────────
1. Download from https://hailo.ai/developer-zone/
     hailort-pcie-driver_4.23.0_all.deb
     hailort_4.23.0_arm64.deb
     hailort-4.23.0-cp312-cp312-linux_aarch64.whl

2. Install kernel headers + DKMS:
     sudo apt install -y dkms linux-headers-$(uname -r)

3. Install HailoRT:
     sudo dpkg -i hailort-pcie-driver_4.23.0_all.deb
     sudo dpkg -i hailort_4.23.0_arm64.deb

4. Install Python bindings into ROS2 Python:
     sudo pip3 install hailort-4.23.0-cp312-cp312-linux_aarch64.whl \
         --target=/opt/ros/jazzy/lib/python3.12/site-packages \
         --no-deps

5. Add user to hailo group and reboot:
     sudo usermod -aG hailo $USER
     sudo reboot

6. Verify:
     hailortcli fw-control identify
     source /opt/ros/jazzy/setup.bash
     python3 -c "from hailo_platform import HEF, VDevice; print('OK')"

7. Place your compiled HEF at MODEL_PATH below.

── ARCHITECTURE ───────────────────────────────────────────────────
  Terminal 1: ros2 launch depthai_ros_driver_v3 driver.launch.py
              params_file:=~/Lawin_ws/lawin_ws/src/ai_cam/config/oak1_params.yaml
  Terminal 2: ros2 run lawin_mavros oak1_tracker
  Terminal 3: ros2 run image_view image_view
              --ros-args -r image:=/lawin/tracking_image

── TOPICS ─────────────────────────────────────────────────────────
  SUB: /oak/rgb/image_raw     sensor_msgs/Image
  SUB: /lawin/rc_command      std_msgs/String
  PUB: /lawin/target_offset   geometry_msgs/Point
  PUB: /lawin/tracking_image  sensor_msgs/Image

── NOTES ──────────────────────────────────────────────────────────
  - Output tensor format: HAILO NMS BY CLASS (not a plain array)
    HailoRT returns a structured buffer parsed via hailo_platform's
    own NMS utilities — see _parse_outputs for details
  - Hailo input: UINT8 RGB 640×640, normalization baked into HEF
  - NMS baked in: score_th=0.3, iou_th=0.7, max_boxes=25, classes=1
  - Inference runs in a dedicated thread; ROS2 callback deposits
    frames into a queue so the spin thread is never blocked

Run:
  ros2 run lawin_mavros oak1_tracker
"""

import os
import time
import threading
import queue
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    raise RuntimeError("OpenCV (cv2) is required but not installed.")

try:
    from hailo_platform import (
        HEF,
        VDevice,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
    )
    HAILO_OK = True
except ImportError:
    HAILO_OK = False

# ── Config ──────────────────────────────────────────────────────
MODEL_PATH   = '/home/lawin/Lawin_ws/lawin_ws/yolo_hailo/lawin_detection.hef'
INFER_SIZE   = 640      # must match HEF compile input (image_dims in NMS json)
CONFIDENCE   = 0.50     # matches nms_scores_th baked into HEF; filter here too
DEAD_ZONE    = 50       # px dead zone

# Colors  (BGR)
DETECT_COLOR   = (255,   0,   0)
CENTROID_COLOR = (  0,   0, 255)
HUD_COLOR      = (  0, 255, 255)


# ────────────────────────────────────────────────────────────────
#  Hailo Inference Helper
# ────────────────────────────────────────────────────────────────

class HailoInference:
    """
    Wraps HailoRT 4.19 VDevice / InferVStreams for single-model inference.

    Usage:
        hi = HailoInference(hef_path)
        detections = hi.infer(rgb_uint8_hwc_640x640)
        # detections → list of (y1, x1, y2, x2, confidence)  pixel coords
        hi.release()

    Output parsing:
        With meta_arch=yolov8 + nms_postprocess(engine=cpu) baked in,
        HailoRT returns a dict keyed by output stream name whose value is
        a numpy array shaped (1, max_proposals, 5+) or a list of
        HailoDetection objects depending on the HailoRT version.

        HailoRT 4.18+ with CPU NMS returns raw float32 arrays shaped:
            (batch, num_detections, 5)   →  [x1, y1, x2, y2, score]
            coords are normalized [0, 1]

        We handle both the object API and the raw array API defensively.
    """

    def __init__(self, hef_path: str):
        if not HAILO_OK:
            raise RuntimeError(
                "hailo_platform not importable. "
                "Install python3-hailort_4.19.0_arm64.deb")

        self.hef    = HEF(hef_path)
        self.target = VDevice()

        configure_params = ConfigureParams.create_from_hef(
            self.hef,
            interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(
            self.hef, configure_params)
        self.network_group  = self.network_groups[0]
        self.network_group_params = \
            self.network_group.create_params()

        # Input stream params — uint8 so we skip float conversion;
        # normalization is baked into the HEF.
        self.input_vstream_params = InputVStreamParams.make(
            self.network_group,
            format_type=FormatType.UINT8)

        # Output stream params — float32 for decoded boxes / scores.
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group,
            format_type=FormatType.FLOAT32)

        # Cache output stream info (names, shapes) for parsing
        self.output_info = \
            self.hef.get_output_vstream_infos()

    # ------------------------------------------------------------------

    def infer(self, rgb_frame: np.ndarray) -> list:
        """
        Args:
            rgb_frame: uint8 numpy array, shape (640, 640, 3), RGB order

        Returns:
            list of (y1, x1, y2, x2, confidence) in PIXEL coordinates
            relative to the 640×640 inference frame.
            Scale back to original frame size in the caller.
        """
        # HailoRT expects a batch dict: {input_name: ndarray (B,H,W,C)}
        input_data = {
            self.hef.get_input_vstream_infos()[0].name:
                np.expand_dims(rgb_frame, axis=0)   # (1,640,640,3)
        }

        with InferVStreams(
                self.network_group,
                self.input_vstream_params,
                self.output_vstream_params) as pipeline:

            with self.network_group.activate(
                    self.network_group_params):

                raw_outputs = pipeline.infer(input_data)

        return self._parse_outputs(raw_outputs)

    # ------------------------------------------------------------------

    def _parse_outputs(self, raw_outputs: dict) -> list:
        """
        Parse HailoRT 4.23 NMS output.

        Confirmed structure:
            raw_outputs: dict
              key:   'lawin_detection_1024/yolov8_nms_postprocess'
              value: list[ batch ][ class ] → np.ndarray shape (N, 5)
                     columns: [x1, y1, x2, y2, score]
                     coords:  normalized [0, 1]

        We have batch=1, class=1 (person), so:
            raw_outputs[key][0][0]  →  (N, 5) array
        """
        detections = []

        for name, batch_list in raw_outputs.items():
            # batch_list[0] = list of per-class arrays for frame 0
            per_class = batch_list[0]

            for class_arr in per_class:
                # class_arr shape: (N, 5) or (0, 5) if no detections
                if class_arr.shape[0] == 0:
                    continue

                for row in class_arr:
                    y1_n, x1_n, y2_n, x2_n, score = (
                        float(row[0]), float(row[1]),
                        float(row[2]), float(row[3]),
                        float(row[4]))

                    if score < CONFIDENCE:
                        continue

                    x1 = int(x1_n * INFER_SIZE)
                    y1 = int(y1_n * INFER_SIZE)
                    x2 = int(x2_n * INFER_SIZE)
                    y2 = int(y2_n * INFER_SIZE)

                    detections.append((x1, y1, x2, y2, score))

        return detections

    # ------------------------------------------------------------------

    def release(self):
        """Release Hailo device resources."""
        try:
            self.target.release()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────
#  ROS2 Node
# ────────────────────────────────────────────────────────────────

class OAK1TrackerNode(Node):

    def __init__(self):
        super().__init__('oak1_tracker')

        # ── State ──────────────────────────────────────────
        self.tracking     = False
        self.fps          = 0.0
        self.prev_time    = time.time()
        self.bridge       = CvBridge()

        self.locked_cx    = None
        self.locked_cy    = None
        self.locked_err_x = 0
        self.locked_err_y = 0
        self.frame_count  = 0
        self.last_boxes   = []

        # ── Frame queue (ROS thread → inference thread) ────
        # maxsize=1 keeps latency low: if inference is busy,
        # the old frame is dropped and replaced by the newest.
        self._frame_queue = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._latest_result = []   # shared: inference → draw thread

        # ── Load Hailo ─────────────────────────────────────
        if not HAILO_OK:
            self.get_logger().error(
                "hailo_platform not importable! "
                "Install python3-hailort_4.19.0_arm64.deb")
            return

        self.get_logger().info(f"Loading HEF: {MODEL_PATH}")
        try:
            self.hailo = HailoInference(MODEL_PATH)
        except Exception as e:
            self.get_logger().error(f"Hailo init failed: {e}")
            return
        self.get_logger().info("Hailo inference ready!")

        # ── Start inference thread ─────────────────────────
        self._infer_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="hailo_infer")
        self._infer_thread.start()

        # ── QoS ────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1)

        # ── Subscribers ────────────────────────────────────
        self.create_subscription(
            Image, '/oak/rgb/image_raw',
            self._on_image, qos)
        self.create_subscription(
            String, '/lawin/rc_command',
            self._on_rc, 10)

        # ── Publishers ─────────────────────────────────────
        self.offset_pub = self.create_publisher(
            Point, '/lawin/target_offset', 10)
        self.image_pub  = self.create_publisher(
            Image, '/lawin/tracking_image', 10)

        self.get_logger().info("=" * 50)
        self.get_logger().info("  OAK-1 Tracker ready  [Hailo-8]")
        self.get_logger().info(
            "  Subscribing to /oak/rgb/image_raw")
        self.get_logger().info(
            "  Make sure ROS driver is running!")
        self.get_logger().info(
            "  CH8 UP=TRACK | CH8 DOWN=CANCEL")
        self.get_logger().info("=" * 50)

    # ──────────────────────────────────────────────────────
    # RC Handler
    # ──────────────────────────────────────────────────────

    def _on_rc(self, msg: String):
        cmd = msg.data
        if cmd == 'TRACK_START' and not self.tracking:
            self.tracking = True
            self.get_logger().info("Tracking STARTED!")
        elif cmd == 'TRACK_CANCEL' and self.tracking:
            self.tracking     = False
            self.locked_cx    = None
            self.locked_cy    = None
            self.locked_err_x = 0
            self.locked_err_y = 0
            self.offset_pub.publish(Point())
            self.get_logger().info(
                "Tracking CANCELLED — centroid cleared!")

    # ──────────────────────────────────────────────────────
    # Image Callback  (ROS2 spin thread)
    # Converts frame, drops into queue for inference thread.
    # Also reads latest results and publishes annotated image.
    # ──────────────────────────────────────────────────────

    def _on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge: {e}")
            return

        # FPS
        now            = time.time()
        self.fps       = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now

        # Kept from original: grayscale conversion + flip
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray,  cv2.COLOR_GRAY2BGR)
        frame = cv2.flip(frame, -1)

        fh, fw = frame.shape[:2]
        cam_cx = fw // 2
        cam_cy = fh // 2

        # ── Every-other-frame skip (kept from original) ────
        self.frame_count += 1
        if self.frame_count % 2 == 1:
            # Prepare 640×640 RGB for Hailo and push to queue
            infer_frame = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))
            infer_rgb   = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
            try:
                # Non-blocking put: if queue is full, discard old frame
                self._frame_queue.put_nowait(infer_rgb)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self._frame_queue.put_nowait(infer_rgb)

        # ── Read latest inference result ───────────────────
        with self._result_lock:
            boxes_raw = list(self._latest_result)

        # boxes_raw: (x1,y1,x2,y2,conf) in 640×640 space
        # Scale to actual frame size
        sx = fw / INFER_SIZE
        sy = fh / INFER_SIZE
        boxes = [
            (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy), c)
            for x1, y1, x2, y2, c in boxes_raw
        ]

        # ── Draw detections ────────────────────────────────
        for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                DETECT_COLOR, 1)
            label = f"ID:{i+1}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(
                frame,
                (x1, y1 - th - 8), (x1 + tw + 8, y1),
                DETECT_COLOR, -1)
            cv2.putText(
                frame, label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Centroid tracking ──────────────────────────────
        if self.tracking:
            if boxes:
                cx = int(np.mean(
                    [(x1+x2)/2 for x1,y1,x2,y2,_ in boxes]))
                cy = int(np.mean(
                    [(y1+y2)/2 for x1,y1,x2,y2,_ in boxes]))
                self.locked_cx    = cx
                self.locked_cy    = cy
                self.locked_err_x = cx - cam_cx
                self.locked_err_y = cy - cam_cy
            elif self.locked_cx is not None:
                cx = self.locked_cx
                cy = self.locked_cy
            else:
                cx = None
                cy = None

            if cx is not None:
                err_x = cx - cam_cx
                err_y = cy - cam_cy

                self.locked_err_x = err_x
                self.locked_err_y = err_y

                dz     = DEAD_ZONE
                inside = (abs(err_x) < dz and abs(err_y) < dz)
                dz_col = (0, 255, 0) if inside else (0, 0, 255)

                if boxes:
                    cv2.circle(frame, (cx, cy), 8,
                               CENTROID_COLOR, -1)
                    label = f"LOCK ({err_x:+d},{err_y:+d})"
                else:
                    cv2.circle(frame, (cx, cy), 10,
                               (0, 165, 255), 2)
                    cv2.circle(frame, (cx, cy), 5,
                               (0, 165, 255), -1)
                    label = f"GHOST ({err_x:+d},{err_y:+d})"
                    cv2.putText(
                        frame, "Person lost — tracking last pos",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 165, 255), 1, cv2.LINE_AA)

                cv2.line(frame, (cam_cx, cam_cy), (cx, cy),
                         (255, 255, 255), 1)
                cv2.putText(
                    frame, "HOLD" if inside else "MOVE",
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, dz_col, 2, cv2.LINE_AA)
                cv2.putText(
                    frame, label,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, CENTROID_COLOR, 1, cv2.LINE_AA)

                offset   = Point()
                offset.x = float(err_x)
                offset.y = float(err_y)
                offset.z = 1.0
                self.offset_pub.publish(offset)

                src_label = 'LIVE' if boxes else 'GHOST'
                h = 'RIGHT' if err_x > 0 else 'LEFT'
                v = 'DOWN'  if err_y > 0 else 'UP'
                self.get_logger().info(
                    f"TRACK[{src_label}] {h}/{v} "
                    f"offset=({err_x:+d},{err_y:+d})px "
                    f"{'HOLD' if inside else 'MOVE'}",
                    throttle_duration_sec=0.5)

                if inside and not boxes:
                    self.get_logger().info(
                        "Reached ghost centroid — lock cleared!")
                    self.locked_cx = None
                    self.locked_cy = None
            else:
                cv2.putText(
                    frame, "Waiting for person...",
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 165, 255), 2, cv2.LINE_AA)
                self.offset_pub.publish(Point())
        else:
            self.offset_pub.publish(Point())

        # ── HUD ────────────────────────────────────────────
        if self.tracking and self.locked_cx is not None:
            lock_str = f"LOCKED({self.locked_cx},{self.locked_cy})"
        elif self.tracking:
            lock_str = "SEARCHING"
        else:
            lock_str = "DETECTING"
        cv2.putText(
            frame,
            f"FPS:{self.fps:.1f} "
            f"People:{len(boxes)} {lock_str}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, HUD_COLOR, 1, cv2.LINE_AA)

        self._draw_reticle(frame, cam_cx, cam_cy)

        # ── Publish annotated image ────────────────────────
        try:
            out_msg = self.bridge.cv2_to_imgmsg(
                frame, encoding='bgr8')
            out_msg.header = msg.header
            self.image_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Publish error: {e}")

    # ──────────────────────────────────────────────────────
    # Inference Thread  — blocks on Hailo, never touches ROS
    # ──────────────────────────────────────────────────────

    def _inference_loop(self):
        self.get_logger().info(
            "Inference thread started.")
        while rclpy.ok():
            try:
                rgb_frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                detections = self.hailo.infer(rgb_frame)
            except Exception as e:
                self.get_logger().error(
                    f"Hailo infer error: {e}",
                    throttle_duration_sec=2.0)
                detections = []
            with self._result_lock:
                self._latest_result = detections
        self.get_logger().info("Inference thread exiting.")

    # ──────────────────────────────────────────────────────
    # Draw Reticle
    # ──────────────────────────────────────────────────────

    def _draw_reticle(self, frame, cx, cy):
        dz  = DEAD_ZONE
        gap = dz + 8
        fh, fw = frame.shape[:2]
        c   = HUD_COLOR
        cv2.line(frame, (0,  cy),      (cx-gap, cy),  c, 1)
        cv2.line(frame, (cx+gap, cy),  (fw,     cy),  c, 1)
        cv2.line(frame, (cx, 0),       (cx, cy-gap),  c, 1)
        cv2.line(frame, (cx, cy+gap),  (cx, fh),      c, 1)
        b = 12
        for px, py in [(cx-dz, cy-dz), (cx+dz, cy-dz),
                       (cx-dz, cy+dz), (cx+dz, cy+dz)]:
            sx = 1 if px < cx else -1
            sy = 1 if py < cy else -1
            cv2.line(frame, (px, py), (px+sx*b, py), c, 1)
            cv2.line(frame, (px, py), (px, py+sy*b), c, 1)
        cv2.circle(frame, (cx, cy), 2, c, -1)

    # ──────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────

    def destroy_node(self):
        if hasattr(self, 'hailo'):
            self.hailo.release()
            self.get_logger().info("Hailo device released.")
        super().destroy_node()


# ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = OAK1TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

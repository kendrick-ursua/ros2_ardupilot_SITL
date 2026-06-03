#!/usr/bin/env python3
"""
LawinTracker ROS2 Node — Centroid Tracking with RC Control
==========================================================
Tracks centroid of ALL detected persons.

Controls:
  RadioMaster Axis 7 UP   (2047) → Start tracking
  RadioMaster Axis 7 DOWN (0)    → Cancel tracking
  Left-click  → Start tracking (manual fallback)
  Right-click → Cancel tracking (manual fallback)
  Q           → Quit

Topics:
  SUB: /camera/image         (sensor_msgs/msg/Image)
  SUB: /lawin/rc_command     (std_msgs/msg/String)
  PUB: /lawin/target_offset  (geometry_msgs/msg/Point)
       x: horizontal offset from center (pixels)
       y: vertical offset from center (pixels)
       z: number of persons detected (0 = not tracking)
"""

import rclpy
import cv2
import numpy as np
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


class LawinTrackerNode(Node):

    # Config
    CONF          = 0.35
    DEAD_ZONE     = 35
    FONT          = 0.4
    FONT_THICK    = 1
    PADDING       = 4
    MARGIN        = 4

    DETECT_COLOR   = (255, 0,   0  )
    CENTROID_COLOR = (0,   0,   255)
    HUD_COLOR      = (0,   255, 255)
    STATUS_COLORS  = {
        "TRACKING":  (0,   0,   255),
        "DETECTING": (0,   255, 0  ),
    }

    def __init__(self):
        super().__init__('lawin_tracker')

        # Parameters
        self.declare_parameter('model_path',   'best.pt')
        self.declare_parameter('camera_topic', '/camera/image')
        self.declare_parameter('show_window',  True)

        model_path    = self.get_parameter('model_path').value
        camera_topic  = self.get_parameter('camera_topic').value
        self.show_win = self.get_parameter('show_window').value

        # YOLO
        self.get_logger().info(f"Loading YOLO model: {model_path}")
        self.model  = YOLO(model_path)
        self.bridge = CvBridge()

        # State
        self.tracking      = False
        self.frame_count   = 0
        self.fps           = 0.0
        self.prev_time     = time.time()
        self.current_frame = None
        self.current_boxes = []

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        # ── Subscribers ───────────────────
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, qos)

        self.rc_sub = self.create_subscription(
            String, '/lawin/rc_command',
            self.rc_callback, 10)

        # ── Publishers ────────────────────
        self.offset_pub = self.create_publisher(
            Point, '/lawin/target_offset', 10)

        # OpenCV Window
        if self.show_win:
            cv2.namedWindow("LawinTracker", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("LawinTracker", self.on_click)

        self.get_logger().info(
            f"LawinTracker ready! Subscribing to: {camera_topic}")
        self.get_logger().info(
            "RC: Axis 4 UP=track | Axis 4 DOWN=cancel")
        self.get_logger().info(
            "Mouse: Left-click=track | Right-click=cancel | Q=quit")

    # ─────────────────────────────────────
    # RC Callback
    # ─────────────────────────────────────
    def rc_callback(self, msg: String):
        """Handle RC commands from RadioMaster."""
        cmd = msg.data

        if cmd == 'TRACK_START':
            if not self.tracking:
                self.tracking = True
                self.get_logger().info(
                    "RC: Tracking STARTED!")

        elif cmd == 'TRACK_CANCEL':
            if self.tracking:
                self.tracking = False
                self.offset_pub.publish(Point())
                self.get_logger().info(
                    "RC: Tracking CANCELLED!")

    # ─────────────────────────────────────
    # Image Callback
    # ─────────────────────────────────────
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        self.frame_count += 1
        curr_time          = time.time()
        self.fps           = 1.0 / (curr_time - self.prev_time + 1e-6)
        self.prev_time     = curr_time
        self.current_frame = frame.copy()
        display_frame      = frame.copy()

        fh, fw = display_frame.shape[:2]
        cam_cx = fw // 2
        cam_cy = fh // 2

        # Always detect persons
        self.current_boxes = self.yolo_detect(frame)
        self.draw_detections(display_frame, self.current_boxes)

        # Person count
        count = len(self.current_boxes)
        cv2.putText(display_frame, f"People: {count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 180, 0), 2, cv2.LINE_AA)

        if self.tracking:
            status = self.process_centroid(
                display_frame, cam_cx, cam_cy)
        else:
            self.offset_pub.publish(Point())
            status = "DETECTING"

        self.draw_hud_reticle(display_frame, cam_cx, cam_cy)

        # Status + FPS
        s_color = self.STATUS_COLORS.get(status, (255, 255, 255))
        cv2.putText(display_frame,
                    f"FPS: {self.fps:.1f} | {status}",
                    (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 1)

        # Instructions
        if not self.tracking:
            cv2.putText(display_frame,
                        "Flip Axis 4 UP or Left-click to track",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(display_frame,
                        "Flip Axis 4 DOWN or Right-click to cancel",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

        if self.show_win:
            cv2.imshow("LawinTracker", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rclpy.shutdown()

    # ─────────────────────────────────────
    # Centroid Tracking
    # ─────────────────────────────────────
    def process_centroid(self, display_frame, cam_cx, cam_cy):
        if len(self.current_boxes) == 0:
            self.offset_pub.publish(Point())
            cv2.putText(display_frame, "No persons detected",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 165, 255), 2, cv2.LINE_AA)
            return "TRACKING"

        # Calculate centroid of all persons
        cx = int(np.mean([(x1 + x2) / 2
                          for x1, y1, x2, y2 in self.current_boxes]))
        cy = int(np.mean([(y1 + y2) / 2
                          for x1, y1, x2, y2 in self.current_boxes]))

        # Draw centroid
        cv2.circle(display_frame, (cx, cy), 8,
                   self.CENTROID_COLOR, -1)
        cv2.putText(display_frame, "C",
                    (cx + 10, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    self.CENTROID_COLOR, 2, cv2.LINE_AA)

        # Draw line from crosshair to centroid
        cv2.line(display_frame,
                 (cam_cx, cam_cy), (cx, cy),
                 (255, 255, 255), 1)

        # Calculate offset
        err_x = cx - cam_cx
        err_y = cy - cam_cy

        # Dead zone indicator
        dz     = self.DEAD_ZONE
        inside = (cam_cx-dz < cx < cam_cx+dz and
                  cam_cy-dz < cy < cam_cy+dz)
        dz_color   = (0, 255, 0) if inside else (0, 0, 255)
        hold_label = "HOLD"      if inside else "MOVE"

        cv2.putText(display_frame, hold_label,
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, dz_color, 2, cv2.LINE_AA)
        cv2.putText(display_frame,
                    f"Centroid: ({cx}, {cy})",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, self.CENTROID_COLOR, 1, cv2.LINE_AA)
        cv2.putText(display_frame,
                    f"Offset: ({err_x:+d}, {err_y:+d})px",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, self.CENTROID_COLOR, 1, cv2.LINE_AA)

        # Publish offset
        offset   = Point()
        offset.x = float(err_x)
        offset.y = float(err_y)
        offset.z = float(len(self.current_boxes))
        self.offset_pub.publish(offset)

        return "TRACKING"

    # ─────────────────────────────────────
    # YOLO Detection
    # ─────────────────────────────────────
    def yolo_detect(self, frame):
        results = self.model(frame, conf=self.CONF, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append([x1, y1, x2, y2])
        return boxes

    # ─────────────────────────────────────
    # Drawing
    # ─────────────────────────────────────
    def draw_detections(self, frame, boxes):
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          self.DETECT_COLOR, 1)
            label = f"P{i+1}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                self.FONT, self.FONT_THICK)
            bg_x1 = x1
            bg_x2 = x1 + tw + 2 * self.PADDING
            bg_y2 = y1
            bg_y1 = y1 - (th + 2 * self.MARGIN)
            cv2.rectangle(frame,
                          (bg_x1, bg_y1), (bg_x2, bg_y2),
                          self.DETECT_COLOR, -1)
            text_x = bg_x1 + ((bg_x2 - bg_x1) - tw) // 2
            text_y = bg_y1 + ((bg_y2 - bg_y1) + th) // 2 - 2
            cv2.putText(frame, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, self.FONT,
                        (255, 255, 255), self.FONT_THICK,
                        cv2.LINE_AA)

    def draw_hud_reticle(self, frame, cx, cy):
        dz      = self.DEAD_ZONE
        gap     = dz + 8
        tick    = 6
        mid_gap = dz - 10
        bracket = 12
        fh, fw  = frame.shape[:2]
        color   = self.HUD_COLOR

        cv2.line(frame, (0, cy),      (cx-gap, cy),  color, 1)
        cv2.line(frame, (cx+gap, cy), (fw, cy),      color, 1)
        cv2.line(frame, (cx, 0),      (cx, cy-gap),  color, 1)
        cv2.line(frame, (cx, cy+gap), (cx, fh),      color, 1)

        for dx in [-mid_gap, mid_gap]:
            cv2.line(frame, (cx+dx, cy-tick),
                     (cx+dx, cy+tick), color, 1)
        for dy in [-mid_gap, mid_gap]:
            cv2.line(frame, (cx-tick, cy+dy),
                     (cx+tick, cy+dy), color, 1)

        x1, y1, x2, y2 = cx-dz, cy-dz, cx+dz, cy+dz
        cv2.line(frame, (x1, y1), (x1+bracket, y1), color, 1)
        cv2.line(frame, (x1, y1), (x1, y1+bracket), color, 1)
        cv2.line(frame, (x2, y1), (x2-bracket, y1), color, 1)
        cv2.line(frame, (x2, y1), (x2, y1+bracket), color, 1)
        cv2.line(frame, (x1, y2), (x1+bracket, y2), color, 1)
        cv2.line(frame, (x1, y2), (x1, y2-bracket), color, 1)
        cv2.line(frame, (x2, y2), (x2-bracket, y2), color, 1)
        cv2.line(frame, (x2, y2), (x2, y2-bracket), color, 1)
        cv2.circle(frame, (cx, cy), 2, color, -1)

    # ─────────────────────────────────────
    # Mouse Callback (manual fallback)
    # ─────────────────────────────────────
    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.tracking:
                self.tracking = True
                self.get_logger().info(
                    "Mouse: Tracking STARTED!")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.tracking = False
            self.offset_pub.publish(Point())
            self.get_logger().info(
                "Mouse: Tracking CANCELLED!")


def main(args=None):
    rclpy.init(args=args)
    node = LawinTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
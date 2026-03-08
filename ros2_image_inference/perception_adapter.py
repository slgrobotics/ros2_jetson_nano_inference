#!/usr/bin/env python3

from math import pi
import json
import subprocess
import time
from typing import Dict, Set, Optional

import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Bool, String, Float32
from sensor_msgs.msg import Illuminance


class PerceptionAdapter(Node):
    """
    Perception -> Behavior adapter.

    Originally designed for face/gesture detections, now adapted to consume
    generic Detection2DArray messages from the image inference node.

    Expected input:
      /image_inference_detections   (vision_msgs/Detection2DArray)

    Published outputs:
      /bt/face_gesture_detect       (sensor_msgs/Illuminance)  # BT hack preserved
      /fgs/face_detected            (std_msgs/Bool)
      /fgs/face_yaw_error           (std_msgs/Float32)
      /fgs/gesture_command          (std_msgs/String)
    """

    def __init__(self):
        super().__init__("perception_adapter")

        # ---- parameters ----
        self.declare_parameter("ticker_interval_sec", 0.1)
        self.declare_parameter("detection_topic", "/image_inference_detections")
        self.declare_parameter("face_detected_sound", "")
        self.declare_parameter("face_detected_text", "")
        self.declare_parameter("min_confidence", 0.6)
        self.declare_parameter("face_cooldown_sec", 3.0)
        self.declare_parameter("camera_center_x", 320.0)

        # What label should be treated as the "trackable target" (old "FACE")
        self.declare_parameter("target_label", "person")

        # Optional mapping for publishers that send numeric class ids as strings.
        # Example: {"0":"person","1":"duckie","2":"stop"}
        self.declare_parameter("label_map_json", "{}")

        # Optional set of labels that should still be treated like gestures/commands
        # Example: ["like", "ok", "stop", "yes", "six"]
        self.declare_parameter("gesture_labels", ["LIKE", "OK", "STOP", "YES", "SIX"])

        self.detection_topic = self.get_parameter("detection_topic").value
        self.face_detected_sound = self.get_parameter("face_detected_sound").value
        self.face_detected_text = self.get_parameter("face_detected_text").value
        self.min_conf = float(self.get_parameter("min_confidence").value)
        self.face_cooldown = float(self.get_parameter("face_cooldown_sec").value)
        self.camera_center_x = float(self.get_parameter("camera_center_x").value)
        self.ticker_interval = float(self.get_parameter("ticker_interval_sec").value)
        self.target_label = str(self.get_parameter("target_label").value).upper()

        try:
            self.label_map: Dict[str, str] = {
                str(k): str(v).upper()
                for k, v in json.loads(self.get_parameter("label_map_json").value).items()
            }
        except Exception as e:
            self.get_logger().warn(f"Invalid label_map_json, using empty map: {e}")
            self.label_map = {}

        self.gesture_labels: Set[str] = {
            str(x).upper() for x in self.get_parameter("gesture_labels").value
        }

        # ---- state ----
        self.state = "idle"  # idle | tracking
        self.last_face_time = 0.0
        self.last_gesture = ""
        self.last_gesture_time = 0.0
        self.last_said = ""

        # ---- publishers ----
        self.face_gesture_detected_pub = self.create_publisher(
            Illuminance, "/bt/face_gesture_detect", 10
        )
        self.face_detected_pub = self.create_publisher(Bool, "/fgs/face_detected", 10)
        self.face_yaw_err_pub = self.create_publisher(Float32, "/fgs/face_yaw_error", 10)
        self.gesture_pub = self.create_publisher(String, "/fgs/gesture_command", 10)

        # ---- subscription ----
        self.sub = self.create_subscription(
            Detection2DArray,
            self.detection_topic,
            self._on_detection,
            10,
        )

        # ---- timer ----
        self.ticker_timer = self.create_timer(self.ticker_interval, self._ticker_callback)

        self.get_logger().info(
            f"Perception Adapter ready. topic={self.detection_topic}, target_label={self.target_label}"
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _normalize_label(self, raw_label: str) -> str:
        raw = str(raw_label).strip()
        mapped = self.label_map.get(raw, raw)
        return mapped.upper()

    def _say_something(self, text: str):
        if text and self.last_said != text:
            subprocess.Popen(["flite", "-t", text])
            self.last_said = text

    def _pub_to_bt(self, face_detected: bool, gesture: str, angle_error: float):
        combo_msg = Illuminance()
        combo_msg.header.stamp = self.get_clock().now().to_msg()
        combo_msg.header.frame_id = gesture
        combo_msg.illuminance = 1.0 if face_detected else 0.0
        combo_msg.variance = float(angle_error)
        self.face_gesture_detected_pub.publish(combo_msg)

    def _publish_target_lost(self):
        self.face_detected_pub.publish(Bool(data=False))
        self.face_yaw_err_pub.publish(Float32(data=0.0))
        bt_gesture = self.last_gesture if (time.time() - self.last_gesture_time) < 1.0 else ""
        self._pub_to_bt(face_detected=False, gesture=bt_gesture, angle_error=0.0)

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    def _on_detection(self, msg: Detection2DArray):
        """
        Consume generic Detection2DArray.

        Strategy:
          * Find the best detection whose label matches target_label
          * Treat certain labels as gesture commands
          * Publish one coherent BT update per callback
        """
        best_target_x: Optional[float] = None
        best_target_conf = -1.0

        for detection in msg.detections:
            for hypothesis_with_pose in detection.results:
                hypothesis = hypothesis_with_pose.hypothesis
                raw_label = hypothesis.class_id
                label = self._normalize_label(raw_label)
                confidence = float(hypothesis.score)

                if confidence < self.min_conf:
                    continue

                # Gesture-like semantic labels
                if label in self.gesture_labels:
                    self._handle_gesture(label)
                    continue

                # Target object (old FACE equivalent)
                if label == self.target_label:
                    obj_x = float(detection.bbox.center.position.x)
                    if confidence > best_target_conf:
                        best_target_conf = confidence
                        best_target_x = obj_x

        if best_target_x is not None:
            self._handle_face(best_target_x)

    def _ticker_callback(self):
        now = time.time()
        if self.state == "tracking" and (now - self.last_face_time) > self.face_cooldown:
            self.get_logger().info("Target disappeared, resetting state to idle")
            self.state = "idle"
            self._publish_target_lost()

    # --------------------------------------------------
    # Semantic handlers
    # --------------------------------------------------

    def _handle_face(self, face_x: float):
        now = time.time()
        self.last_face_time = now

        distance_px = face_x - self.camera_center_x
        angle_factor = pi / (6 * 320.0)  # assumes ~60 deg FOV over 640 px
        angle_error = distance_px * angle_factor

        self.face_detected_pub.publish(Bool(data=True))
        self.face_yaw_err_pub.publish(Float32(data=float(angle_error)))

        bt_gesture = self.last_gesture if (now - self.last_gesture_time) < 1.0 else ""
        self._pub_to_bt(face_detected=True, gesture=bt_gesture, angle_error=float(angle_error))

        if self.state == "idle":
            self.get_logger().info(
                f"{self.target_label} detected (first time) at x={face_x:.1f}, "
                f"distance_px={distance_px:.1f}, angle_error={angle_error:.3f}"
            )

            if self.face_detected_sound:
                subprocess.Popen(["aplay", self.face_detected_sound])

            if self.face_detected_text:
                self._say_something(self.face_detected_text)

            self.state = "tracking"

        else:
            self.get_logger().debug(
                f"{self.target_label} at x={face_x:.1f}, "
                f"distance_px={distance_px:.1f}, angle_error={angle_error:.3f}"
            )

    def _handle_gesture(self, gesture: str):
        gesture = gesture.upper()
        self.last_gesture = gesture
        self.last_gesture_time = time.time()

        self.get_logger().info(f"Gesture: {gesture}")
        self.gesture_pub.publish(String(data=gesture))

        spoken = {
            "LIKE": "Like",
            "OK": "Okay",
            "STOP": "Stop",
            "YES": "Yes",
            "SIX": "Six",
        }.get(gesture, gesture)

        self._say_something(spoken)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionAdapter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
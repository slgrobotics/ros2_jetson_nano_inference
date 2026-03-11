#!/usr/bin/env python3

import json
import subprocess
import time
from math import pi
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Illuminance
from std_msgs.msg import Bool, Float32, String
from vision_msgs.msg import Detection2DArray

#
# For the reasons of this designs see:
#  - https://github.com/slgrobotics/articubot_one/wiki/Behavior-Tree-for-Gesture-and-Face-Detection-Sensor
#

class PerceptionAdapter(Node):
    """
    Perception -> Behavior adapter.

    Input:
      /image_inference_detections   (vision_msgs/Detection2DArray)

    Published outputs:
      /bt/face_gesture_detect       (sensor_msgs/Illuminance)
      /fgs/face_detected            (std_msgs/Bool)
      /fgs/face_yaw_error           (std_msgs/Float32)
      /fgs/gesture_command          (std_msgs/String)

    Semantics:
      - target_label (default: PERSON) is treated as the old "face" target
      - one best target instance is selected for yaw tracking
      - all non-target labels are treated as gesture-like semantic commands
      - selected labels can be mapped to arbitrary gesture names via gesture_map_json
      - unmapped labels pass through as uppercase strings
      - mapped-label priority is determined by the order in gesture_map_json
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

        # Label treated as the old "FACE" target
        self.declare_parameter("target_label", "person")

        # Optional mapping of classes to "gesture names".
        # Example: {"cup":"STOP", "giraffe":"LIKE", "cat":"MEOW", "dog":"WOOF", "chair":"OK"}
        self.declare_parameter("label_map_json", "{}")

        # Ordered mapping:
        #   detected_label -> gesture name for BT / speech
        #
        # The order matters.
        # Earlier entries take priority over later ones, regardless of confidence.
        #
        # Example:
        #   {"DOG":"YES","CHAIR":"SIX","CAT":"STOP"}
        #
        # If both DOG and CHAIR are detected, DOG wins because it appears first.
        self.declare_parameter("gesture_map_json", "{}")

        # Optional speech overrides keyed by final gesture/command name
        # Example: {"STOP":"Stop","CUP":"Cup","FIRE_HYDRANT":"Fire hydrant"}
        self.declare_parameter("speech_map_json", "{}")

        # Suppress repeating the same command too frequently
        self.declare_parameter("gesture_cooldown_sec", 1.0)

        self.detection_topic = str(self.get_parameter("detection_topic").value)
        self.face_detected_sound = str(self.get_parameter("face_detected_sound").value)
        self.face_detected_text = str(self.get_parameter("face_detected_text").value)
        self.min_conf = float(self.get_parameter("min_confidence").value)
        self.face_cooldown = float(self.get_parameter("face_cooldown_sec").value)
        self.camera_center_x = float(self.get_parameter("camera_center_x").value)
        self.ticker_interval = float(self.get_parameter("ticker_interval_sec").value)
        self.target_label = str(self.get_parameter("target_label").value).strip().upper()
        self.gesture_cooldown = float(self.get_parameter("gesture_cooldown_sec").value)

        try:
            self.label_map: Dict[str, str] = {
                str(k): str(v).strip().upper()
                for k, v in json.loads(self.get_parameter("label_map_json").value).items()
            }
        except Exception as e:
            self.get_logger().warn(f"Invalid label_map_json, using empty map: {e}")
            self.label_map = {}

        try:
            gesture_map_raw = json.loads(self.get_parameter("gesture_map_json").value)
            if not isinstance(gesture_map_raw, dict):
                raise ValueError("gesture_map_json must decode to a JSON object")
            self.gesture_map: Dict[str, str] = {
                str(k).strip().upper(): str(v).strip().upper()
                for k, v in gesture_map_raw.items()
            }
        except Exception as e:
            self.get_logger().warn(f"Invalid gesture_map_json, using empty map: {e}")
            self.gesture_map = {}

        # Ordered list of labels by priority (earlier entry wins)
        self.gesture_priority_labels: List[str] = list(self.gesture_map.keys())

        try:
            speech_map_raw = json.loads(self.get_parameter("speech_map_json").value)
            if not isinstance(speech_map_raw, dict):
                raise ValueError("speech_map_json must decode to a JSON object")
            self.speech_map: Dict[str, str] = {
                str(k).strip().upper(): str(v)
                for k, v in speech_map_raw.items()
            }
        except Exception as e:
            self.get_logger().warn(f"Invalid speech_map_json, using empty map: {e}")
            self.speech_map = {}

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
            f"Perception Adapter ready. topic={self.detection_topic}, "
            f"target_label={self.target_label}, gesture_map={self.gesture_map}"
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _normalize_label(self, raw_label: str) -> str:
        raw = str(raw_label).strip()
        mapped = self.label_map.get(raw, raw)
        return mapped.upper()

    def _map_label_to_gesture(self, label: str) -> str:
        return self.gesture_map.get(label, label)

    def _say_something(self, text: str):
        if text and self.last_said != text:
            subprocess.Popen(["flite", "-t", text])
            self.last_said = text

    def _spoken_text_for_gesture(self, gesture: str) -> str:
        return self.speech_map.get(gesture, gesture.replace("_", " ").title())

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

    def _pick_best_gesture_label(
        self,
        detected_non_target_labels: Dict[str, float],
    ) -> Optional[str]:
        """
        Return the winning detected label before mapping to a gesture name.

        Priority rules:
          1. If any detected label appears in gesture_map_json, the earliest such label
             in gesture_map_json wins, regardless of confidence.
          2. Otherwise, pick the highest-confidence unmapped label.
        """
        for priority_label in self.gesture_priority_labels:
            if priority_label in detected_non_target_labels:
                return priority_label

        if not detected_non_target_labels:
            return None

        best_label, _ = max(
            detected_non_target_labels.items(),
            key=lambda item: item[1],
        )
        return best_label

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    def _on_detection(self, msg: Detection2DArray):
        """
        Consume generic Detection2DArray.

        Strategy:
          * Find the best detection whose label matches target_label
          * Consider all other labels as gesture-like semantic commands
          * gesture_map_json labels have explicit priority by order
          * If no mapped label is present, pass through the best other label
          * Publish one coherent BT update per callback
        """
        best_target_x: Optional[float] = None
        best_target_conf = -1.0

        # label -> best confidence seen this callback
        detected_non_target_labels: Dict[str, float] = {}

        for detection in msg.detections:
            for hypothesis_with_pose in detection.results:
                hypothesis = hypothesis_with_pose.hypothesis
                raw_label = hypothesis.class_id
                label = self._normalize_label(raw_label)
                confidence = float(hypothesis.score)

                if confidence < self.min_conf:
                    continue

                if label == self.target_label:
                    obj_x = float(detection.bbox.center.position.x)
                    if confidence > best_target_conf:
                        best_target_conf = confidence
                        best_target_x = obj_x
                    continue

                prev_conf = detected_non_target_labels.get(label, -1.0)
                if confidence > prev_conf:
                    detected_non_target_labels[label] = confidence

        winning_label = self._pick_best_gesture_label(detected_non_target_labels)
        if winning_label is not None:
            gesture = self._map_label_to_gesture(winning_label)
            self._handle_gesture(gesture)

        if best_target_x is not None:
            self._handle_face(best_target_x)

    def _ticker_callback(self):
        """
        Periodic check: if target not detected for face_cooldown_sec, reset to idle.
        """
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
        now = time.time()

        if gesture == self.last_gesture and (now - self.last_gesture_time) < self.gesture_cooldown:
            return

        self.last_gesture = gesture
        self.last_gesture_time = now

        self.get_logger().info(f"Gesture/command: {gesture}")
        self.gesture_pub.publish(String(data=gesture))

        spoken = self._spoken_text_for_gesture(gesture)
        self._say_something(spoken)

    def destroy_node(self):
        self.ticker_timer.cancel()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionAdapter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from math import pi
import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Bool, String, Float32
from sensor_msgs.msg import Illuminance

import subprocess
import time

"""
see: https://chatgpt.com/s/t_693f44d1eca08191b70128efa562496e
     https://chatgpt.com/s/t_693f472c11488191adc7db1457b82db5

Design overview (intentional & Nav2-friendly)

  What this node does
    - Listens to perception output
    - Publishes semantic events for a Behavior Tree
    - Executes aplay for face detection
    - Publishes intent messages for motion (not raw velocity)

  What this node does NOT do
    - Does not publish /cmd_vel
    - Does not fight Nav2 controllers
    - Does not contain navigation logic
"""

class PerceptionAdapter(Node):
    """
    Perception → Behavior adapter for Face & Gesture sensor.

    Responsibilities:
      * Listen to Detection2DArray messages from face_gesture_sensor
      * Detect semantic events (FACE, LIKE, OK, STOP, YES, SIX)
      * Publish BT-friendly signals
      * Trigger sound playback for face detection
    """

    def __init__(self):
        super().__init__('perception_adapter')

        # ---- parameters ----
        self.declare_parameter('face_detected_sound', '')
        self.declare_parameter('face_detected_text', '')
        self.declare_parameter('min_confidence', 0.6)
        self.declare_parameter('face_cooldown_sec', 3.0)
        self.declare_parameter('camera_center_x', 320.0)  # Assuming 640px width sensor camera
        self.declare_parameter('ticker_interval_sec', 0.1)  # New: Ticker interval (defines rate or publishing all messages)

        self.face_detected_sound = self.get_parameter('face_detected_sound').value
        self.face_detected_text = self.get_parameter('face_detected_text').value
        self.min_conf = self.get_parameter('min_confidence').value
        self.face_cooldown = self.get_parameter('face_cooldown_sec').value
        self.camera_center_x = self.get_parameter('camera_center_x').value
        self.ticker_interval = self.get_parameter('ticker_interval_sec').value

        # ---- state machine ----
        self.state = 'idle'  # 'idle' or 'tracking'
        self.last_face_time = 0.0
        self.last_gesture = ''
        self.last_gesture_time = 0.0
        self.last_said = ''

        # ---- publishers (Behavior Tree inputs) ----
        self.face_gesture_detected_pub = self.create_publisher(
            Illuminance, '/bt/face_gesture_detect', 10  # hack: using Illuminance for boolean, string, float32 and timestamp
        )

        # ---- publishers (anybody's inputs) ----
        self.face_detected_pub = self.create_publisher(
            Bool, '/fgs/face_detected', 10
        )

        self.face_yaw_err_pub = self.create_publisher(
            Float32, '/fgs/face_yaw_error', 10
        )

        self.gesture_pub = self.create_publisher(
            String, '/fgs/gesture_command', 10
        )

        # ---- subscription ----
        self.sub = self.create_subscription(
            Detection2DArray,
            'face_gesture_detections',  # Match the fgs_node published topic
            self._on_detection,
            10
        )

        # ---- ticker timer ----
        self.ticker_timer = self.create_timer(self.ticker_interval, self._ticker_callback)

        self.get_logger().info("Perception Adapter ready")

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    def _on_detection(self, msg: Detection2DArray):
        """
        Handle detections from Detection2DArray (of ObjectHypothesisWithPose messages).
        """
        for detection in msg.detections:
            for hypothesis_with_pose in detection.results:
                hypothesis = hypothesis_with_pose.hypothesis
                label = hypothesis.class_id.upper()
                confidence = hypothesis.score

                if confidence < self.min_conf:
                    continue

                if label == 'FACE':
                    # Extract face position from bbox
                    face_x = detection.bbox.center.position.x
                    self._handle_face(face_x)
                elif 'LIKE' in label: # "1 = LIKE (blue)"
                    self._handle_like()
                elif 'OK' in label:
                    self._handle_ok()
                elif 'STOP' in label:
                    self._handle_stop()
                elif 'YES' in label:
                    self._handle_yes()
                elif 'SIX' in label:
                    self._handle_six()

    def _ticker_callback(self):
        """
        Periodic check: If face not detected for face_cooldown_sec, reset to idle.
        """
        now = time.time()
        if self.state == 'tracking' and (now - self.last_face_time) > self.face_cooldown:
            self.get_logger().info("Face disappeared, resetting state to idle")
            self.state = 'idle'

    def _say_something(self, text):
        """
        Use flite to say something.
        """
        if self.last_said != text:
            subprocess.Popen(['flite', '-t', text])
        self.last_said = text

    def _pub_to_bt(self, face_detected: bool, gesture: str, angle_error: float):
        """
        Hack: using Illuminance message to send combo info, because we need a time-stamped message in BT plugins.
        Publish to Behavior Tree topic.
        """
        combo_msg = Illuminance()
        combo_msg.header.stamp = self.get_clock().now().to_msg()
        combo_msg.header.frame_id = gesture  # Use frame_id to send gesture as string
        combo_msg.illuminance = 1.0 if face_detected else 0.0
        combo_msg.variance = angle_error  # Use variance field to send angle error

        self.face_gesture_detected_pub.publish(combo_msg)


    # --------------------------------------------------
    # Semantic handlers
    #
    # Each handler processes a specific gesture or face detection event.
    # They abstract away low-level details and publish high-level intent messages.
    # --------------------------------------------------

    def _handle_face(self, face_x):
        now = time.time()
        self.last_face_time = now  # Update last seen time

        # expect face_x in range 0...640, camera center at 320
        distance_px = face_x - self.camera_center_x # positive turn when you are on robot's left

        angle_factor = pi / (6 * 320)   # Factor to convert pixel error to angle error (assume FOV +-30 degrees = +-320 pixels)
        angle_error = distance_px * angle_factor

        self.face_detected_pub.publish(Bool(data=True)) # Face detected event, publish continuously while face is in view

        self.face_yaw_err_pub.publish(Float32(data=float(angle_error))) # Where to turn, publish continuously while face is in view

        # Face gesture detected event for BT, publish continuously while face is in view
        # Hack: using Illuminance message to send combo info
        bt_gesture = self.last_gesture if (now - self.last_gesture_time) < 1.0 else ''
        self._pub_to_bt(face_detected=True, gesture=bt_gesture, angle_error=float(angle_error))

        # State Machine:
        #    'idle': No face. On detection, greet and switch to 'tracking'.
        #    'tracking': Face in view. Trace only on significant x-change.
        # State resets to 'idle' in "_ticker_callback()" if no detection for face_cooldown_sec.

        if self.state == 'idle':
            # First face detection: greet and start tracking
            # expect face_x in range 0...640
            self.get_logger().info(f"Face detected (first time) at x={face_x}, distance_px: {distance_px}  angle_error: {angle_error}")

            if(self.face_detected_sound != ''):
                # Play greeting sound, once per appearance
                subprocess.Popen(['aplay', self.face_detected_sound])

            if(self.face_detected_text != ''):
                # Or, use "flite": sudo apt install flite
                self._say_something(self.face_detected_text)

            self.state = 'tracking'

        elif self.state == 'tracking':
            # continuously publish face position deviation from the center of view, in pixels
            self.get_logger().info(f"Face at x={face_x}, distance_px: {distance_px}  angle_error: {angle_error}")


    def _handle_like(self):
        self.get_logger().info("Gesture: LIKE")
        # Notify BT
        self.last_gesture = 'LIKE'
        now = time.time()
        self.last_gesture_time = now  # Update last gesture time
        # Notify others
        self.gesture_pub.publish(String(data='LIKE'))
        self._say_something("Like")
        # Add custom action if needed (e.g., play sound or trigger motion)

    def _handle_ok(self):
        self.get_logger().info("Gesture: OK")
        self.last_gesture = 'OK'
        now = time.time()
        self.last_gesture_time = now  # Update last gesture time
        self.gesture_pub.publish(String(data='OK'))
        self._say_something("Okay")

    def _handle_stop(self):
        self.get_logger().info("Gesture: STOP")
        self.last_gesture = 'STOP'
        now = time.time()
        self.last_gesture_time = now  # Update last gesture time
        self.gesture_pub.publish(String(data='STOP'))
        self._say_something("Stop")

    def _handle_yes(self):
        self.get_logger().info("Gesture: YES")
        self.last_gesture = 'YES'
        now = time.time()
        self.last_gesture_time = now  # Update last gesture time
        self.gesture_pub.publish(String(data='YES'))
        self._say_something("Yes")

    def _handle_six(self):
        self.get_logger().info("Gesture: SIX")
        self.last_gesture = 'SIX'
        now = time.time()
        self.last_gesture_time = now  # Update last gesture time
        self.gesture_pub.publish(String(data='SIX'))
        self._say_something("Six")


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


if __name__ == '__main__':
    main()

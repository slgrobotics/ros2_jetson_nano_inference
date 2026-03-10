import json
import socket
import struct
import time
from datetime import datetime
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import (
    Point2D,
    Pose2D,
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
    ObjectHypothesis,
)

from .inference_response_parser import parse_inference_response, InferenceResult


class ImageInferenceNode(Node):
    def __init__(self):
        super().__init__("image_inference_node")

        self.declare_parameter("ticker_interval_sec", 0.1)
        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 5001)
        self.declare_parameter("startup_delay_sec", 5.0)
        self.declare_parameter("image_topic", "/camera/image_raw/compressed")
        self.declare_parameter("frame_id_out", "camera")
        self.declare_parameter("min_confidence", 0.6)
        self.declare_parameter("objects_allowed", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("stats_period_sec", 5.0)
        self.declare_parameter("use_server_cam", False)  # do not send images from ROS, the server's camera feeds ingerence engine directly

        self.ticker_interval_sec = self.get_parameter("ticker_interval_sec").value
        self.server_host = self.get_parameter("server_host").value
        self.server_port = self.get_parameter("server_port").value
        self.startup_delay_sec = self.get_parameter("startup_delay_sec").value
        self.image_topic = self.get_parameter("image_topic").value
        self.frame_id_out = self.get_parameter("frame_id_out").value
        self.min_confidence = self.get_parameter("min_confidence").value
        self.objects_allowed = { s.strip() for s in self.get_parameter("objects_allowed").value if s.strip() }
        self.stats_period_sec = self.get_parameter("stats_period_sec").value
        self.use_server_cam = self.get_parameter("use_server_cam").value

        self.get_logger().info("Image Inference node started")

        if self.objects_allowed:
            self.get_logger().info(
                f"Allowed to detect: {sorted(self.objects_allowed)}"
            )
        else:
            self.get_logger().info("No 'objects_allowed' set; allowing all detected objects")

        self.detection_pub = self.create_publisher(
            Detection2DArray, "image_inference_detections", 10
        )

        if self.use_server_cam:
            server_cam_image = "server_cam_image"
            self.get_logger().info(f"Using server camera feed directly for inference, publishing '{server_cam_image}' topic for visualization")
            self.image_pub = self.create_publisher(Image, "server_cam_image", 10)
        else:
            self.get_logger().info(f"Subscribing to ROS CompressedImage topic '{self.image_topic}' for inference")
            self.image_sub = self.create_subscription(
                CompressedImage,
                self.image_topic,
                self.image_callback,
                10,
            )

        self.setup_timer = self.create_timer(self.startup_delay_sec, self.setup)
        self.loop_timer = self.create_timer(self.ticker_interval_sec, self.loop_callback)

        self.server_ready = False
        self.sock: Optional[socket.socket] = None

        self.latest_jpg: Optional[bytes] = None
        self.latest_image_stamp = None
        self.latest_source_frame_id = ""
        self.latest_image_seq = 0
        self.last_sent_seq = -1

        self.start_time = datetime.now()
        self.stats_window_start = time.monotonic()
        self.stats_last_print = self.stats_window_start
        self.server_calls_in_window = 0
        self.total_server_calls = 0

    def image_callback(self, msg: CompressedImage) -> None:
        # Only used when "use_server_cam" is False
        # Keep latest only
        self.latest_jpg = bytes(msg.data)
        self.latest_image_stamp = msg.header.stamp
        self.latest_source_frame_id = msg.header.frame_id
        self.latest_image_seq += 1

    def recv_exact(self, sock: socket.socket, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError("socket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def send_request(self, sock: socket.socket, frame_id: int, jpg_bytes: bytes) -> None:
        header = {
            "frame_id": frame_id,
            "timestamp_ns": time.time_ns(),
            "encoding": "jpeg",
            "payload_size": len(jpg_bytes),
        }
        hdr = json.dumps(header).encode("utf-8")
        sock.sendall(struct.pack(">I", len(hdr)))
        sock.sendall(hdr)
        if not self.use_server_cam:
            # when server is using its own camera feed directly, we don't send images from ROS at all, so skip sending empty payload
            sock.sendall(jpg_bytes)

    def recv_response(self, sock: socket.socket) -> dict:
        n = struct.unpack(">I", self.recv_exact(sock, 4))[0]
        data = self.recv_exact(sock, n)
        if self.use_server_cam:
            # Expect a JPEG image response from the server camera feed,
            # publish it for visualization and inference result
            pass
        return json.loads(data.decode("utf-8"))

    def connect_server(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

        self.get_logger().info(
            f"Connecting to inference server at {self.server_host}:{self.server_port}..."
        )
        self.sock = socket.create_connection((self.server_host, self.server_port), timeout=10)
        self.sock.settimeout(10.0)
        self.server_ready = True
        self.get_logger().info("Connected to inference server")

    def setup(self) -> None:
        try:
            self.connect_server()
        except Exception as e:
            self.server_ready = False
            self.get_logger().error(f"Failed to connect to inference server: {e}")
            return

        self.setup_timer.cancel()

    def build_detection_array_msg(self, result: InferenceResult) -> Optional[Detection2DArray]:
        msg = Detection2DArray()

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.latest_source_frame_id or self.frame_id_out

        for d in result.detections:

            confidence = d.confidence

            if confidence < self.min_confidence:
                continue

            if self.objects_allowed and d.label not in self.objects_allowed:
                continue

            x1, y1, x2, y2 = d.bbox_xyxy
            cx, cy, w, h = d.bbox_xywh

            self.get_logger().info(
                f"Publishing detection: label={d.label}, confidence={confidence:.3f}, "
                f"bbox_xyxy=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), "
                f"bbox_xywh=({cx:.0f}, {cy:.0f}, {w:.0f}, {h:.0f})"
            )

            detection = Detection2D()
            detection.header = header

            center = Pose2D()
            center.position = Point2D(x=float(cx), y=float(cy))
            center.theta = 0.0

            bbox = BoundingBox2D()
            bbox.center = center
            bbox.size_x = float(w)
            bbox.size_y = float(h)
            detection.bbox = bbox

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis = ObjectHypothesis()
            hypothesis.hypothesis.class_id = d.label  # str(d.class_id)   "label" is easier for downstream use than "class_id", and we have it available
            hypothesis.hypothesis.score = float(d.confidence)

            detection.results.append(hypothesis)

            msg.detections.append(detection)

        if len(msg.detections) == 0:
            return None  # no detections above confidence threshold, return None to indicate empty result

        msg.header = header

        return msg

    def loop_callback(self) -> None:
        if not self.server_ready or self.sock is None:
            return

        if self.latest_jpg is None:
            return

        # Avoid resending the same frame over and over
        if self.latest_image_seq == self.last_sent_seq:
            return

        frame_seq = self.latest_image_seq
        jpg = self.latest_jpg

        try:
            self.send_request(self.sock, frame_seq, jpg)
            sock_response = self.recv_response(self.sock)
            result = parse_inference_response(json.dumps(sock_response))
            self.last_sent_seq = frame_seq
            self.server_calls_in_window += 1
            self.total_server_calls += 1
        except Exception as e:
            self.get_logger().error(f"Inference request failed: {e}")
            self.server_ready = False
            try:
                if self.sock is not None:
                    self.sock.close()
            except Exception:
                pass
            self.sock = None

            try:
                self.connect_server()
            except Exception as reconnect_error:
                self.get_logger().error(f"Reconnect failed: {reconnect_error}")
            return

        self.print_stats()

        self.get_logger().debug(
            f"frame_id={result.frame_id} infer_ms={result.infer_ms:.1f} "
            f"queue_delay_ms={result.queue_delay_ms:.1f} detections={len(result.detections)}"
        )

        """
        for d in result.detections:
            self.get_logger().info(
                f"Server detected: label={d.label}, class_id={d.class_id}, "
                f"confidence={d.confidence:.3f}, bbox={d.bbox_xyxy}"
            )
        """

        detection_array_msg = self.build_detection_array_msg(result)

        if detection_array_msg is not None:
            # Only publish if there are detections above confidence threshold
            self.detection_pub.publish(detection_array_msg)

    def print_stats(self) -> None:
        now_monotonic = time.monotonic()
        elapsed_window = now_monotonic - self.stats_window_start

        if elapsed_window < self.stats_period_sec:
            return

        server_calls_per_second = self.server_calls_in_window / elapsed_window if elapsed_window > 0.0 else 0.0

        now_dt = datetime.now()
        current_time = now_dt.strftime("%H:%M:%S")
        elapsed_total = now_dt - self.start_time
        elapsed_total_str = str(elapsed_total).split(".")[0]

        self.get_logger().info(
            f"Current Time: {current_time} | "
            f"Elapsed: {elapsed_total_str} | "
            f"Total calls: {self.total_server_calls} | "
            f"Calls: {self.server_calls_in_window} in {elapsed_window:.1f}s | "
            f"Server calls per second: {server_calls_per_second:.2f}"
        )

        self.stats_window_start = now_monotonic
        self.server_calls_in_window = 0


    def destroy_node(self):
        self.loop_timer.cancel()
        self.setup_timer.cancel()
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageInferenceNode()
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
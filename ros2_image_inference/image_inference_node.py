import json
import socket
import struct
import time
from datetime import datetime
from typing import Optional

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import (
    Point2D,
    Pose2D,
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
    ObjectHypothesis,
)

from inference_response_parser import parse_inference_response, InferenceResult


class ImageInferenceNode(Node):
    def __init__(self):
        super().__init__("image_inference_node")

        self.declare_parameter("ticker_interval_sec", 0.1)
        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 5001)
        self.declare_parameter("startup_delay_sec", 5.0)
        self.declare_parameter("image_topic", "/camera/image_raw/compressed")
        self.declare_parameter("frame_id_out", "camera")

        self.ticker_interval_sec = self.get_parameter("ticker_interval_sec").value
        self.server_host = self.get_parameter("server_host").value
        self.server_port = self.get_parameter("server_port").value
        self.startup_delay_sec = self.get_parameter("startup_delay_sec").value
        self.image_topic = self.get_parameter("image_topic").value
        self.frame_id_out = self.get_parameter("frame_id_out").value

        self.get_logger().info("Image Inference node started")

        self.detection_pub = self.create_publisher(
            Detection2DArray, "image_inference_detections", 10
        )

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

        self.print_time_counter = 0
        self.start_time = datetime.now()

    def image_callback(self, msg: CompressedImage) -> None:
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
        sock.sendall(jpg_bytes)

    def recv_response(self, sock: socket.socket) -> dict:
        n = struct.unpack(">I", self.recv_exact(sock, 4))[0]
        data = self.recv_exact(sock, n)
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

    def build_detection_array_msg(self, result: InferenceResult) -> Detection2DArray:
        msg = Detection2DArray()

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.latest_source_frame_id or self.frame_id_out
        msg.header = header

        for d in result.detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            cx, cy, w, h = d.bbox_xywh

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

        self.print_time()

        self.get_logger().debug(
            f"frame_id={result.frame_id} infer_ms={result.infer_ms:.1f} "
            f"queue_delay_ms={result.queue_delay_ms:.1f} detections={len(result.detections)}"
        )

        for d in result.detections:
            self.get_logger().info(
                f"Detection: label={d.label}, class_id={d.class_id}, "
                f"confidence={d.confidence:.3f}, bbox={d.bbox_xyxy}"
            )

        detection_array_msg = self.build_detection_array_msg(result)
        self.detection_pub.publish(detection_array_msg)

    def print_time(self) -> None:
        self.print_time_counter += 1
        if self.print_time_counter % 10 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            elapsed = now - self.start_time
            elapsed_str = str(elapsed).split(".")[0]
            self.get_logger().info(f"Current Time: {current_time} | Elapsed: {elapsed_str}")

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
#!/usr/bin/env python3

# ========================================================
# ROS 2 stereo client node for sparse point cloud and preview image visualization.
#
# This node receives sparse 3D point data from the Jetson Nano (or other streamer) over UDP,
# reconstructs a PointCloud2 message, and publishes it for downstream use
# (e.g., Nav2 obstacle processing and RViz2 visualization).
#
# In parallel, it connects to the Nano’s TCP image server to request the latest
# rectified left preview image as JPEG, decodes it, publishes it as a raw ROS
# image, and derives approximate per-point RGB values from the corresponding
# grid cells.
#
# The resulting PointCloud2 contains:
# - geometry (x, y, z)
# - packed RGB color for visualization
# - confidence
# - grid row/column metadata
#
# Key features:
# - UDP reception of sparse stereo point data
# - TCP on-demand retrieval of preview JPEG frames
# - Raw image publication for RViz2 and RQt
# - Approximate point-cloud colorization from image grid cells
# - Single PointCloud2 topic usable for both debugging and navigation
#
# Intended use:
# - Visualizing stereo-derived sparse clouds in RViz2
# - Supplying point clouds to Nav2 / local costmap obstacle layers
# - Debugging perception alignment between cloud and camera image
#
# See https://github.com/slgrobotics/jetson_nano_b01/blob/main/src/stereo/disparity_server.py
#
# Expected packet format must match the sender exactly:
#
# Header: <4sBBBBIQHH
#   magic       4s   "SPC2"
#   version     B
#   rows        B
#   cols        B
#   reserved    B
#   seq         I
#   stamp_ns    Q
#   point_count H
#   reserved2   H
#
# Point record: <ffffHH
#   x           f    meters
#   y           f    meters
#   z           f    meters
#   confidence  f    0..1
#   row         H
#   col         H
# ========================================================

import socket
import struct
from typing import List, Tuple
import threading

import json
import numpy as np

import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2


HEADER_STRUCT = struct.Struct("<4sBBBBIQHH")
POINT_STRUCT = struct.Struct("<ffffHH")

HEADER_MAGIC = b"SPC2"
HEADER_VERSION = 1


class UdpSparseCloudReceiver(Node):
    def __init__(self) -> None:
        super().__init__("disparity_client_node")

        self.declare_parameter("verbose", False)
        self.declare_parameter("bind_ip", "0.0.0.0")
        self.declare_parameter("port", 5005)
        self.declare_parameter("topic", "stereo/sparse_cloud")
        self.declare_parameter("frame_id", "stereo_camera")
        self.declare_parameter("color_patch_fraction", 0.5)   # center patch size relative to cell
        self.declare_parameter("use_mean_color", True)
        self.declare_parameter("ticker_interval_sec", 0.1)  # 10 Hz UDP socket poll timer
        self.declare_parameter("socket_timeout_sec", 0.0)   # non-blocking
        self.declare_parameter("log_every_n_packets", 10)

        self.declare_parameter("image_topic", "camera/image_raw")
        self.declare_parameter("tcp_host", "jetson.local")
        self.declare_parameter("tcp_port", 5006)
        self.declare_parameter("request_image_every_sec", 0.5)
        self.declare_parameter("jpeg_max_width", 320)
        self.declare_parameter("jpeg_max_height", 180)
        self.declare_parameter("jpeg_quality", 60)
        self.declare_parameter("tcp_timeout_sec", 5.0)



        self.verbose = bool(self.get_parameter("verbose").value)
        bind_ip = str(self.get_parameter("bind_ip").value)
        port = int(self.get_parameter("port").value)
        topic = str(self.get_parameter("topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.color_patch_fraction = float(self.get_parameter("color_patch_fraction").value)
        self.use_mean_color = bool(self.get_parameter("use_mean_color").value)
        ticker_interval_sec = float(self.get_parameter("ticker_interval_sec").value)
        socket_timeout_sec = float(self.get_parameter("socket_timeout_sec").value)
        self.log_every_n_packets = int(self.get_parameter("log_every_n_packets").value)

        image_topic = str(self.get_parameter("image_topic").value)
        self.tcp_host = str(self.get_parameter("tcp_host").value)
        self.tcp_port = int(self.get_parameter("tcp_port").value)
        self.request_image_every_sec = float(self.get_parameter("request_image_every_sec").value)
        self.jpeg_max_width = int(self.get_parameter("jpeg_max_width").value)
        self.jpeg_max_height = int(self.get_parameter("jpeg_max_height").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.tcp_timeout_sec = float(self.get_parameter("tcp_timeout_sec").value)

        self.br = CvBridge()
        self.tcp_sock = None

        self.image_pub = self.create_publisher(Image, image_topic, 10)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub_cloud = self.create_publisher(PointCloud2, topic, qos)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_ip, port))
        self.sock.settimeout(socket_timeout_sec)

        self.last_seq = -1
        self.packet_counter = 0

        self.fields = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="confidence", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="row", offset=20, datatype=PointField.UINT16, count=1),
            PointField(name="col", offset=22, datatype=PointField.UINT16, count=1),
        ]

        self.timer = self.create_timer(ticker_interval_sec, self.poll_socket)
        self.image_timer = self.create_timer(
            self.request_image_every_sec,
            self.image_request_callback,
        )

        self.latest_image = None
        self.latest_image_stamp_ns = 0
        self.latest_image_lock = threading.Lock()

        self.get_logger().info(
            f"Listening for UDP sparse cloud packets on {bind_ip}:{port}, "
            f"publishing PointCloud2 on {topic}"
        )

    def destroy_node(self):
        try:
            self.timer.cancel()
        except Exception:
            pass

        try:
            self.image_timer.cancel()
        except Exception:
            pass

        try:
            self.sock.close()
        except Exception:
            pass

        self.disconnect_image_server()
        super().destroy_node()


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

    def send_json_request(self, sock: socket.socket, obj: dict) -> None:
        data = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        sock.sendall(struct.pack(">I", len(data)))
        sock.sendall(data)

    def recv_image_response(self, sock: socket.socket):
        n = struct.unpack(">I", self.recv_exact(sock, 4))[0]
        data = self.recv_exact(sock, n)
        response = json.loads(data.decode("utf-8"))

        jpg_bytes = None
        if bool(response.get("has_jpeg", False)):
            jpeg_len = struct.unpack(">I", self.recv_exact(sock, 4))[0]
            jpg_bytes = self.recv_exact(sock, jpeg_len)

        return response, jpg_bytes

    def connect_image_server(self) -> None:
        self.disconnect_image_server()
        self.get_logger().info(
            f"Connecting to image server at {self.tcp_host}:{self.tcp_port}..."
        )
        self.tcp_sock = socket.create_connection(
            (self.tcp_host, self.tcp_port),
            timeout=self.tcp_timeout_sec,
        )
        self.tcp_sock.settimeout(self.tcp_timeout_sec)
        self.get_logger().info("Connected to image server")

    def disconnect_image_server(self) -> None:
        try:
            if self.tcp_sock is not None:
                self.tcp_sock.close()
        except Exception:
            pass
        self.tcp_sock = None


    def image_request_callback(self):
        if self.tcp_sock is None:
            try:
                self.connect_image_server()
            except Exception as exc:
                self.get_logger().warning(f"Image server connect failed: {exc}")
                return

        try:
            req = {
                "request_jpeg": True,
                "max_width": self.jpeg_max_width,
                "max_height": self.jpeg_max_height,
                "jpeg_quality": self.jpeg_quality,
                "payload_size": 0,
            }

            self.send_json_request(self.tcp_sock, req)
            response, jpg_bytes = self.recv_image_response(self.tcp_sock)

            if not response.get("ok", False):
                self.get_logger().warning(
                    f"Image server error: {response.get('error', 'unknown')}"
                )
                return

            if not response.get("has_jpeg", False) or not jpg_bytes:
                return

            np_buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warning("Failed to decode JPEG from image server")
                return

            msg = self.br.cv2_to_imgmsg(frame, encoding="bgr8")

            stamp_ns = int(response.get("timestamp_ns", 0))
            if stamp_ns > 0:
                msg.header.stamp.sec = stamp_ns // 1_000_000_000
                msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
            else:
                msg.header.stamp = self.get_clock().now().to_msg()

            msg.header.frame_id = self.frame_id
            self.image_pub.publish(msg)

            with self.latest_image_lock:
                self.latest_image = frame.copy()
                self.latest_image_stamp_ns = stamp_ns

            if self.verbose:
                seq = response.get("seq", -1)
                self.get_logger().info(
                    f"Published raw image from TCP server: seq={seq}, shape={frame.shape[1]}x{frame.shape[0]}"
                )

        except Exception as exc:
            self.get_logger().warning(f"TCP image request failed: {exc}")
            self.disconnect_image_server()


    def poll_socket(self) -> None:
        """
        Drain all currently available UDP packets and publish only the latest valid one.
        This avoids building up latency if packets arrive faster than we publish.
        """
        latest_msg = None

        while True:
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                break
            except BlockingIOError:
                break
            except Exception as exc:
                self.get_logger().error(f"Socket receive error: {exc}")
                return

            parsed = self.parse_packet(data)
            if parsed is None:
                continue

            seq, stamp_ns, rows, cols, points = parsed

            # Drop old/out-of-order packets.
            if seq <= self.last_seq:
                continue

            self.last_seq = seq
            latest_msg = self.build_pointcloud2(seq, stamp_ns, rows, cols, points)

            self.packet_counter += 1
            if self.log_every_n_packets > 0 and (self.packet_counter % self.log_every_n_packets == 0):
                self.get_logger().info(
                    f"seq={seq} points={len(points)} grid={rows}x{cols} from {addr[0]}:{addr[1]}"
                )

        if latest_msg is not None:
            self.pub_cloud.publish(latest_msg)

    def parse_packet(
        self, data: bytes
    ) -> Tuple[int, int, int, int, List[Tuple[float, float, float, float, int, int]]] | None:
        if len(data) < HEADER_STRUCT.size:
            self.get_logger().warning("Received packet smaller than header; ignoring")
            return None

        try:
            magic, version, rows, cols, _reserved, seq, stamp_ns, point_count, _reserved2 = \
                HEADER_STRUCT.unpack_from(data, 0)
        except struct.error as exc:
            self.get_logger().warning(f"Failed to unpack header: {exc}")
            return None

        if magic != HEADER_MAGIC:
            self.get_logger().warning(f"Bad magic {magic!r}; ignoring packet")
            return None

        if version != HEADER_VERSION:
            self.get_logger().warning(f"Unsupported packet version {version}; ignoring packet")
            return None

        expected_size = HEADER_STRUCT.size + point_count * POINT_STRUCT.size
        if len(data) != expected_size:
            self.get_logger().warning(
                f"Packet size mismatch: got {len(data)}, expected {expected_size}; ignoring packet"
            )
            return None

        points: List[Tuple[float, float, float, float, int, int]] = []
        offset = HEADER_STRUCT.size

        try:
            for _ in range(point_count):
                x, y, z, confidence, row, col = POINT_STRUCT.unpack_from(data, offset)
                points.append((x, y, z, confidence, row, col))
                offset += POINT_STRUCT.size
        except struct.error as exc:
            self.get_logger().warning(f"Failed to unpack point records: {exc}")
            return None

        return seq, stamp_ns, rows, cols, points

    # RGB component helpers:

    def pack_rgb_float(self, r: int, g: int, b: int) -> float:
        rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]

    def get_latest_image_copy(self):
        with self.latest_image_lock:
            if self.latest_image is None:
                return None
            return self.latest_image.copy()

    def sample_cell_rgb(self, img: np.ndarray, row: int, col: int, rows: int, cols: int) -> float:
        h, w = img.shape[:2]

        x0 = int(col * w / cols)
        x1 = int((col + 1) * w / cols)
        y0 = int(row * h / rows)
        y1 = int((row + 1) * h / rows)

        cell_w = max(1, x1 - x0)
        cell_h = max(1, y1 - y0)

        frac = max(0.05, min(1.0, self.color_patch_fraction))
        patch_w = max(1, int(round(cell_w * frac)))
        patch_h = max(1, int(round(cell_h * frac)))

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2

        px0 = max(x0, cx - patch_w // 2)
        px1 = min(x1, px0 + patch_w)
        py0 = max(y0, cy - patch_h // 2)
        py1 = min(y1, py0 + patch_h)

        patch = img[py0:py1, px0:px1]
        if patch.size == 0:
            return self.pack_rgb_float(255, 0, 255)  # magenta for unavailable color

        if self.use_mean_color:
            mean_bgr = patch.reshape(-1, 3).mean(axis=0)
            b, g, r = [int(round(v)) for v in mean_bgr]
        else:
            b, g, r = [int(v) for v in patch[patch.shape[0] // 2, patch.shape[1] // 2]]

        return self.pack_rgb_float(r, g, b)


    def build_pointcloud2(
        self,
        seq: int,
        stamp_ns: int,
        rows: int,
        cols: int,
        points: List[Tuple[float, float, float, float, int, int]],
    ) -> PointCloud2:
        header = Header()
        header.frame_id = self.frame_id
        header.stamp.sec = int(stamp_ns // 1_000_000_000)
        header.stamp.nanosec = int(stamp_ns % 1_000_000_000)

        img = self.get_latest_image_copy()

        colored_points = []
        for x, y, z, confidence, row, col in points:
            if img is not None:
                rgb = self.sample_cell_rgb(img, row, col, rows, cols)
            else:
                rgb = self.pack_rgb_float(255, 255, 255)

            colored_points.append((x, y, z, rgb, confidence, row, col))

        msg = point_cloud2.create_cloud(header, self.fields, colored_points)
        msg.is_dense = False
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = UdpSparseCloudReceiver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()

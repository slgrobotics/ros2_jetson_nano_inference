#!/usr/bin/env python3

"""
@brief
ROS 2 UDP receiver for sparse stereo point cloud packets.

This node listens for UDP packets produced by the Jetson stereo disparity
server, decodes the packet header and sparse XYZ point records, and publishes
the result as a sensor_msgs/PointCloud2 message.

See https://github.com/slgrobotics/jetson_nano_b01/blob/main/src/stereo/disparity_server.py

Expected packet format must match the sender exactly:

Header: <4sBBBBIQHH
  magic       4s   "SPC2"
  version     B
  rows        B
  cols        B
  reserved    B
  seq         I
  stamp_ns    Q
  point_count H
  reserved2   H

Point record: <ffffHH
  x           f    meters
  y           f    meters
  z           f    meters
  confidence  f    0..1
  row         H
  col         H
"""

import socket
import struct
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Header
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
        self.declare_parameter("topic", "/stereo/sparse_cloud")
        self.declare_parameter("frame_id", "stereo_camera")
        self.declare_parameter("ticker_interval_sec", 0.1)  # 10 Hz UDP socket poll timer
        self.declare_parameter("socket_timeout_sec", 0.0)   # non-blocking
        self.declare_parameter("log_every_n_packets", 10)

        self.verbose = bool(self.get_parameter("verbose").value)
        bind_ip = str(self.get_parameter("bind_ip").value)
        port = int(self.get_parameter("port").value)
        topic = str(self.get_parameter("topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        ticker_interval_sec = float(self.get_parameter("ticker_interval_sec").value)
        socket_timeout_sec = float(self.get_parameter("socket_timeout_sec").value)
        self.log_every_n_packets = int(self.get_parameter("log_every_n_packets").value)

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
            PointField(name="confidence", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="row", offset=16, datatype=PointField.UINT16, count=1),
            PointField(name="col", offset=18, datatype=PointField.UINT16, count=1),
        ]

        self.timer = self.create_timer(ticker_interval_sec, self.poll_socket)

        self.get_logger().info(
            f"Listening for UDP sparse cloud packets on {bind_ip}:{port}, "
            f"publishing PointCloud2 on {topic}"
        )

    def destroy_node(self):
        try:
            self.sock.close()
        except Exception:
            pass
        super().destroy_node()

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

        # Use sender timestamp if possible.
        header.stamp.sec = int(stamp_ns // 1_000_000_000)
        header.stamp.nanosec = int(stamp_ns % 1_000_000_000)

        msg = point_cloud2.create_cloud(header, self.fields, points)
        msg.is_dense = False
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UdpSparseCloudReceiver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

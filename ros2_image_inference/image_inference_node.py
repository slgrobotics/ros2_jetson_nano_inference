import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from datetime import datetime

from std_msgs.msg import Header
from vision_msgs.msg import Point2D, Pose2D, BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose, ObjectHypothesis

import json
import socket
import struct
import time
import cv2

from inference_response_parser import parse_inference_response, InferenceResult, Detection

#
# ros2 run ros2_image_inference image_inference_node
#

class ImageInferenceNode(Node):

    def __init__(self):
        super().__init__('image_inference_node')  # Initialize the Node with a unique name

        self.declare_parameter('ticker_interval_sec', 0.1)     # Ticker interval (defines rate or publishing all messages)
        self.declare_parameter('server_host', '127.0.0.1')
        self.declare_parameter('server_port', 5001)
        self.declare_parameter('startup_delay_sec', 5.0)  # Delay before starting the main loop

        self.ticker_interval_sec = self.get_parameter('ticker_interval_sec').get_parameter_value().double_value
        self.server_host = self.get_parameter('server_host').get_parameter_value().string_value
        self.server_port = self.get_parameter('server_port').get_parameter_value().integer_value
        self.startup_delay_sec = self.get_parameter('startup_delay_sec').get_parameter_value().double_value

        self.get_logger().info('OK: Image Inference node has been started!')

        self.setup_timer = self.create_timer(self.startup_delay_sec, self.setup)  # Call setup after startup delay
        self.loop_timer = self.create_timer(self.ticker_interval_sec, self.loop_callback)  # Call often

        self.detection_pub = self.create_publisher(Detection2DArray, 'image_inference_detections', 10)  # Add publisher

        self.server_ready = False  # Flag to indicate sensor is initialized
        self.print_time_counter = 0  # Add a counter for print_time()
        self.start_time = datetime.now()  # Store program start time
        

    def recv_exact(self, sock, n):
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError("socket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


    def send_request(self, sock, frame_id, jpg_bytes):
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


    def recv_response(self, sock):
        n = struct.unpack(">I", self.recv_exact(sock, 4))[0]
        data = self.recv_exact(sock, n)
        return json.loads(data.decode("utf-8"))


    def setup(self):
        """
        @brief Setup function for initializing sensor thresholds.
        
        This function gets the thresholds for face detection and gesture detection.
        """
        # This will be called after 5 seconds - let the sensor to start.

        self.get_logger().info('IP: establishing connection to the server...')

        IMAGE_PATH = "../media/duckies_2_480x480.jpg"

        self.get_logger().info(f"Loading image from {IMAGE_PATH}...")

        img = cv2.imread(IMAGE_PATH)
        ok, enc = cv2.imencode(".jpg", img)
        jpg = enc.tobytes()

        with socket.create_connection((self.server_host, self.server_port), timeout=10) as sock:
            self.sock = sock
            self.send_request(sock, 1, jpg)
            sock_response = self.recv_response(sock)
            self.get_logger().info(json.dumps(sock_response, indent=2))

        self.server_ready = True

        self.get_logger().info('OK: Connection to the server established, server is ready')
        self.setup_timer.cancel()  # Cancel the setup timer


    def loop_callback(self):
        # This method will be called every 500 ms
        #self.get_logger().info('Periodic callback triggered')

        if not self.server_ready:
            return

        self.send_request(self.sock, 1, jpg)
        sock_response = self.recv_response(self.sock)

        self.print_time()

        json_str = sock_response.decode("utf-8")   # example

        result = parse_inference_response(json_str)

        self.get_logger().info(f"Frame ID: {result.frame_id}")
        self.get_logger().info(f"Inference Time: {result.infer_ms} ms")

        num_detections = len(result.detections)

        # Check if any detections are present:
        if num_detections > 0:
            self.get_logger().info("Number of detections: {}".format(num_detections))

        for d in result.detections:
            self.get_logger().info(f"Detection: {d.label}, Confidence: {d.confidence}, BBox: {d.bbox_xyxy}")

            # Get object score and position coordinates
            object_score = d.confidence
            object_x = d.bbox_xyxy[0]  # Assuming bbox_xyxy is a list [x1, y1, x2, y2]
            object_y = d.bbox_xyxy[1]

            self.get_logger().info("Detect object at (x = {}, y = {}, score = {})".format(object_x, object_y, object_score))
            
            detection_array_msg = Detection2DArray()
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "object_gesture_sensor"  # or another appropriate frame
            detection_array_msg.header = header

            center = Pose2D()
            center.position = Point2D(x=float(object_x), y=float(object_y))
            center.theta = 0.0  # Assuming theta is not used for 2D detection
            object_bbox = BoundingBox2D(center=center, size_x=10.0, size_y=10.0)
            detection = Detection2D(bbox=object_bbox)

            # Fill object_score in results
            hypothesis_object = ObjectHypothesisWithPose()
            hypothesis_object.hypothesis = ObjectHypothesis()
            hypothesis_object.hypothesis.score = float(object_score)
            hypothesis_object.hypothesis.class_id = "object"
            detection.results.append(hypothesis_object)

            detection_array_msg.detections.append(detection)
            self.detection_pub.publish(detection_array_msg)  # Publish the message


    def print_time(self):
        self.print_time_counter += 1
        if self.print_time_counter % 10 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            elapsed = now - self.start_time
            elapsed_str = str(elapsed).split('.')[0]  # Format as H:M:S
            self.get_logger().info(f"Current Time: {current_time} | Elapsed: {elapsed_str}")
            

    def print_gesture(self, gesture_type, gesture_score):
        if gesture_type > 0 and gesture_score > 0:
            gesture_str = self.gesture_names_long.get(gesture_type, "Unknown")
            self.get_logger().info(f"Gesture: {gesture_str}, Score: {gesture_score}")
        

    def destroy_node(self):
        self.loop_timer.cancel()  # Cancel the loop timer
        #gfd.destroy()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageInferenceNode()
    try:
        rclpy.spin(node)  # Keep the node alive and processing callbacks
    except KeyboardInterrupt:
        # can't log here, rclpy is already shutting down
        pass  # Handle keyboard interrupt gracefully
    #except Exception:
    #     traceback.print_exc()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()


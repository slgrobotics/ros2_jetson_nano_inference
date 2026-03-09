## A ROS2 package to delegate image inference to Jetson Nano

This package uses a Jetson Nano for image recognition (inference) and passes detections to Behavior Trees.

There are two components:
- A node that calls a remote inference server (for example, running on an NVIDIA Jetson Nano) and publishes detections.
- A "perception" node that converts the detections array into a format suitable for Behavior Trees.

It is based on the following repositories:
- https://github.com/slgrobotics/jetson_nano_b01 — code for the Nano, including the TCP/IP server and test clients
- https://github.com/slgrobotics/face_gesture_sensor — a similar publisher and perception nodes

The inference server running on the Nano is called from a ROS2 node running on a Raspberry Pi.

See this [AI Chat](https://chatgpt.com/s/t_69ab6eac6b0081919c64ed4045987d0f) for general architecture.

### Prerequisites

You need a Jetson Nano (or another machine), running [Inference TCP/IP Server](https://github.com/slgrobotics/jetson_nano_b01/blob/main/README.md#inference-tcpip-server).
Note that machine's TCP/IP (IPV4) address.

I run the server with the following command (in the container; assuming that you've already built the `yolo11n.engine`):
```
root@jetson:/code/src/dt-duckpack-yolo/shared/src# python3 yolo_tcp_server.py --model yolo11n.engine --imgsz 480 --warmup 3 --host 0.0.0.0 --port 5001
```

Make sure that the server responds, using Python scripts in the "test" directory.

### Build instructions:

You need a camera grabber or other publisher of `/camera/image_raw/compressed` topic.

```
sudo apt install flite aplay    # optional: install text-to speech and sound player

mkdir -p ~/grabber_ws/src
cd ~/grabber_ws/src/
git clone https://github.com/slgrobotics/ros2_jetson_nano_inference.git
git clone https://github.com/slgrobotics/camera_publisher.git   # works with webcams

# edit `~/grabber_ws/src/ros2_jetson_nano_inference/launch/ros2_image_inference.launch.py`
#       - point 'server_host' to your Jetson Nano, running Inference TCP/IP Server

cd ~/grabber_ws
colcon build
```

Run camera grabber in the first terminal:
```
source cd ~/grabber_ws/install/setup.bash
ros2 run cv_basics img_publisher
```

Launch the two nodes in the second terminal:
```
source cd ~/grabber_ws/install/setup.bash
ros2 launch ros2_image_inference ros2_image_inference.launch.py
```

Use RQT and RViz2 to observe the published messages:
- `/camera/image_raw`  - can be viewed in RViz2
- `/camera/image_raw/compressed`  - passed to "image_inference_node"
- `/image_inference_detections` - passed to "perception_adapter" node
- `/fgs/face_detected`, `/fgs/face_yaw_error` and `/fgs/gesture_command` - can be consumed by Behavior Trees custom plugins

See https://github.com/slgrobotics/slg_bt_plugins for more information.

-------------------------

Back to [Main Project Home](https://github.com/slgrobotics/articubot_one/wiki)

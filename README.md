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

-------------------------

Back to [Main Project Home](https://github.com/slgrobotics/articubot_one/wiki)

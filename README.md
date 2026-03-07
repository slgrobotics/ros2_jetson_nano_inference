## A ROS2 node to delegate image inference to Jetson Nano

A ROS2 node to call a remote inference server, running, for example, on an NVIDIA Jetson Nano

It is based on the following repositories:
- https://github.com/slgrobotics/jetson_nano_b01  - code for the Nano, TCP/IP Server and test clients
- https://github.com/slgrobotics/face_gesture_sensor - a similar publisher node



The inference server running on Nano is called from the ROS2 node running on Raspberry Pi.

See this [AI Chat](https://chatgpt.com/s/t_69ab6eac6b0081919c64ed4045987d0f) for general architecture.

-------------------------

Back to [Main Project Home](https://github.com/slgrobotics/articubot_one/wiki)

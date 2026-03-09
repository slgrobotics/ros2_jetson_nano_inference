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

On a development machine:
- You need a [camera publisher](https://github.com/slgrobotics/camera_publisher) or other publisher of `/camera/image_raw/compressed` topic.
- We also add "[detection_visualizer](https://github.com/ros2/detection_visualizer)" package to see bounding boxes overlaid over the image.

```
sudo apt install flite aplay    # optional: install text-to speech and sound player

mkdir -p ~/grabber_ws/src
cd ~/grabber_ws/src/
git clone https://github.com/slgrobotics/ros2_jetson_nano_inference.git
git clone https://github.com/slgrobotics/camera_publisher.git   # works with webcams
git clone https://github.com/ros2/detection_visualizer.git

# edit `~/grabber_ws/src/ros2_jetson_nano_inference/launch/ros2_image_inference.launch.py`
#       - point 'server_host' to your Jetson Nano, running Inference TCP/IP Server

cd ~/grabber_ws
colcon build
```

Run camera publisher in the first terminal:
```
source cd ~/grabber_ws/install/setup.bash
ros2 run cv_basics img_publisher
```

Launch the two nodes and visualizer in the second terminal:
```
source cd ~/grabber_ws/install/setup.bash
ros2 launch ros2_image_inference ros2_image_inference.launch.py
```

Use RQT and RViz2 to observe the published messages:
- `/camera/image_raw`  - can be viewed in RViz2
- `/camera/image_raw/compressed`  - passed to "image_inference_node"
- `/image_inference_detections` - passed to "perception_adapter" node
- `/fgs/face_detected`, `/fgs/face_yaw_error` and `/fgs/gesture_command` - can be consumed by Behavior Trees custom plugins

You can open RQT or RViz2 to see the recognized objects and bounding boxes:

(**RQT:** *Plugins->Visualization->Image View*;  topic: `/image_inference_overlay`)
<img width="990" height="747" alt="Screenshot from 2026-03-09 14-33-39" src="https://github.com/user-attachments/assets/b19ab059-5110-43fe-9cd6-af06b75662dd" />

The *image_inference_node* will print statistics, with objects in view server is called ~6 times per second, a blank image is processed at the rate of ~9+ per second:
```
[image_inference_node]: Current Time: 15:28:37 | Elapsed: 0:00:30 | Total calls: 161 | Calls: 33 in 5.2s | Server calls per second: 6.39
[image_inference_node]: Publishing detection: label=person, confidence=0.930, bbox_xyxy=(100, 146, 552, 474), bbox_xywh=(326, 310, 452, 328)
...
[image_inference_node]: Publishing detection: label=person, confidence=0.884, bbox_xyxy=(8, 289, 638, 473), bbox_xywh=(323, 381, 630, 185)
[perception_adapter-2] [INFO] [1773088122.395230070] [perception_adapter]: Target disappeared, resetting state to idle
[image_inference_node]: Current Time: 15:28:42 | Elapsed: 0:00:35 | Total calls: 195 | Calls: 34 in 5.1s | Server calls per second: 6.62
[image_inference_node]: Current Time: 15:28:47 | Elapsed: 0:00:40 | Total calls: 241 | Calls: 46 in 5.1s | Server calls per second: 9.08
[image_inference_node]: Current Time: 15:28:52 | Elapsed: 0:00:45 | Total calls: 290 | Calls: 49 in 5.1s | Server calls per second: 9.67
```

See https://github.com/slgrobotics/slg_bt_plugins for more information.

-------------------------

Back to [Main Project Home](https://github.com/slgrobotics/articubot_one/wiki)

-------------------------

### List of objects recognized by *yolo11n* model

```
Model classes:
0: person
1: bicycle
2: car
3: motorcycle
4: airplane
5: bus
6: train
7: truck
8: boat
9: traffic light
10: fire hydrant
11: stop sign
12: parking meter
13: bench
14: bird
15: cat
16: dog
17: horse
18: sheep
19: cow
20: elephant
21: bear
22: zebra
23: giraffe
24: backpack
25: umbrella
26: handbag
27: tie
28: suitcase
29: frisbee
30: skis
31: snowboard
32: sports ball
33: kite
34: baseball bat
35: baseball glove
36: skateboard
37: surfboard
38: tennis racket
39: bottle
40: wine glass
41: cup
42: fork
43: knife
44: spoon
45: bowl
46: banana
47: apple
48: sandwich
49: orange
50: broccoli
51: carrot
52: hot dog
53: pizza
54: donut
55: cake
56: chair
57: couch
58: potted plant
59: bed
60: dining table
61: toilet
62: tv
63: laptop
64: mouse
65: remote
66: keyboard
67: cell phone
68: microwave
69: oven
70: toaster
71: sink
72: refrigerator
73: book
74: clock
75: vase
76: scissors
77: teddy bear
78: hair drier
79: toothbrush
```

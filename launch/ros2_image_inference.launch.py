import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('ros2_image_inference')

    # Path to params YAML file (optional)
    params_file = os.path.join(package_dir, 'config', 'params.yaml')

    # Image inference node
    image_inference_node = Node(
        package='ros2_image_inference',
        executable='image_inference_node',
        name='image_inference_node',
        output='screen',
        parameters=[{
            'ticker_interval_sec': 0.1,
            'server_host': 'jetson.local',  # Jetson Nano host IP (not container)
            'server_port': 5001,
            'startup_delay_sec': 5.0,
            'image_topic': '/camera/image_raw/compressed',
            'frame_id_out': 'camera',
            'min_confidence': 0.6,    # do not publish if below this confidence threshold
            'objects_allowed': ['person', 'cup', 'dog', 'cat'], # Case sensitive. Empty list means allow all detected objects
            'stats_period_sec': 5.0,
            #"use_server_cam": True,  # Default: false. If True - do not send images from ROS, the server's camera feeds inference engine directly
        }]
        # parameters=[params_file]  # Load params from YAML instead
    )

    # Perception adapter node
    perception_adapter_node = Node(
        package='ros2_image_inference',
        executable='perception_adapter',
        name='perception_adapter',
        output='screen',
        parameters=[{
            'ticker_interval_sec': 0.1,
            'detection_topic': '/image_inference_detections',
            # 'face_detected_sound': 'my_face.wav',
            'face_detected_text': 'I see you!',
            'min_confidence': 0.6,
            'face_cooldown_sec': 2.0,
            'camera_center_x': 320.0,
            'target_label': 'person',
        }]
        # parameters=[params_file]  # Load params from YAML instead
    )

    visualization_node = Node(
        package='detection_visualizer',
        executable='detection_visualizer',
        name='detection_visualizer',
        output='screen',
        remappings=[
            ('~/images', '/camera/image_raw'),
            ('~/detections', '/image_inference_detections'),
            ('~/dbg_images', '/image_inference_overlay'),
        ]
    )

    return LaunchDescription([
        image_inference_node,
        perception_adapter_node,
        visualization_node,
    ])

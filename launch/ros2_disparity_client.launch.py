import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

#
# colcon build; source install/setup.bash; ros2 launch ros2_image_inference ros2_image_inference.launch.py
#

def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('ros2_image_inference')

    # Path to params YAML file (optional)
    params_file = os.path.join(package_dir, 'config', 'params.yaml')

    # disparity client node
    disparity_client_node = Node(
        package='ros2_image_inference',
        executable='disparity_client_node',
        name='disparity_client_node',
        output='screen',
        parameters=[{
            'verbose': True,        # If true - print debug info.
            'bind_ip': "0.0.0.0",
            'port': 5005,
            'topic': "/stereo/sparse_cloud",
            'frame_id': "stereo_camera",
            'ticker_interval_sec': 0.1,  # 10 Hz UDP socket poll timer
            'socket_timeout_sec': 0.0,    # non-blocking
            'log_every_n_packets': 10,
        }]
        # parameters=[params_file]  # Load params from YAML instead
    )

    return LaunchDescription([
        disparity_client_node,
    ])

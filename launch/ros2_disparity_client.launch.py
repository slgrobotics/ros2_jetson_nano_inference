import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

#
# See https://github.com/slgrobotics/jetson_nano_b01/blob/main/src/stereo/disparity_server.py
#
# colcon build; source install/setup.bash; ros2 launch ros2_image_inference ros2_disparity_client.launch.py
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

    # Visualize PointCloud2 in RViz2:
    rviz_config = os.path.join(package_dir, 'config', 'config.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    # static transform publisher for RViz2:
    tf_to_map = Node(package = "tf2_ros", 
                    executable = "static_transform_publisher",
                    arguments=[
                        '--x', '0.0',     # X translation in meters
                        '--y', '0.0',     # Y translation in meters
                        '--z', '1.0',     # Z translation in meters
                        '--roll', '0.0',  # Roll in radians
                        '--pitch', '0.0', # Pitch in radians
                        '--yaw', '0.0',   # Yaw in radians (e.g., 90 degrees)
                        '--frame-id', 'map', # Parent frame ID
                        '--child-frame-id', 'stereo_camera' # Child frame ID
                    ]
   )

    return LaunchDescription([
        disparity_client_node,
        tf_to_map,
        rviz,
    ])

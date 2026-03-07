import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('ros2_image_inference')
    
    # Path to params YAML file
    params_file = os.path.join(package_dir, 'config', 'params.yaml')
    
    # Define nodes
    image_inference_node = Node(
        package='ros2_image_inference',
        executable='image_inference_node',
        name='image_inference_node',
        output='screen'
    )
    
    perception_adapter_node = Node(
        package='ros2_image_inference',
        executable='perception_adapter',
        name='perception_adapter',
        output='screen',
        parameters=[params_file]  # Load params from YAML
    )
    
    # Launch description
    return LaunchDescription([
        image_inference_node,
        perception_adapter_node
    ])

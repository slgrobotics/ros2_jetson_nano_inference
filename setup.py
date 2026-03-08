# This is not used until you change package.xml to include the following:
#   <export>
#     <build_type>ament_python</build_type>
#   </export>

from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'ros2_image_inference'

setup(
    name=package_name,
    version='0.1.0',
    #packages=find_packages(),
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # config files
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        # media files
        (os.path.join('share', package_name, 'media'), glob(os.path.join('media', '*.*'))),
    ],
    install_requires=[
        'setuptools',
        ],
    zip_safe=True,
    maintainer='Sergei Grichine',
    maintainer_email='slg@quakemap.com',
    description='Face and Gesture detection node',
    license='MIT',
     entry_points={
         'console_scripts': [
            f'image_inference_node = {package_name}.image_inference_node:main',
            #f'perception_adapter = {package_name}.perception_adapter:main',
         ],
     },
)

import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'air_io'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kenders',
    maintainer_email='kendrick.ursua@gmail.com',
    description='Air-IO ROS2 inference node',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'imu_inference_node = air_io.imu_inference_node_blackbird:main',
            'air_imu_node = air_io.air_imu_network:main',
            'air_io_ekf_node = air_io.air_io_ekf:main',
        ],
    },
)

from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Construct the correct absolute path to virtual environment Python
    home_dir = os.path.expanduser('~')
    venv_python = os.path.join(home_dir, 'ros2_ardupilot_SITL/Air-IO/airvenv/bin/python3')

    air_io_node = Node(
        package='air_io',
        executable='imu_inference_node',
        name='air_io_inference_node',
        prefix=f'{venv_python}',
        parameters=[{
            'CONFIG_PATH': f'{home_dir}/ros2_ardupilot_SITL/Air-IO/configs/blackbird/motion_body_rot.conf',
            'CKPT_PATH':   f'{home_dir}/ros2_ardupilot_SITL/Air-IO/experiments/blackbird/motion_body_rot/ckpt/best_model.ckpt',
            'WINDOW_SIZE': 200,
        }],
        output='screen'
    )
    air_imu_node = Node(
        package='air_io',
        executable='air_imu_node',
        name='air_imu_node',
        prefix=f'{venv_python}',
        output='screen'
    )
    air_io_ekf_node = Node(
        package='air_io',
        executable='air_io_ekf_node',
        name='air_io_ekf_node',
        prefix=f'{venv_python}',
        output='screen'
    )

    return LaunchDescription([air_io_node, air_imu_node])
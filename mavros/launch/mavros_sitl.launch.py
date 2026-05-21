"""MAVROS launch file for ArduPilot SITL container."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for MAVROS with ArduPilot SITL."""

    # ── Arguments ────────────────────────────────────────────────────────────
    fcu_url_arg = DeclareLaunchArgument(
        'fcu_url',
        default_value='udp://:14550@127.0.0.1:14555',
        description='FCU connection URL (MAVLink endpoint exposed by ArduPilot SITL)',
    )

    gcs_url_arg = DeclareLaunchArgument(
        'gcs_url',
        default_value='',
        description='GCS proxy URL (leave empty to disable)',
    )

    tgt_system_arg = DeclareLaunchArgument(
        'tgt_system',
        default_value='1',
        description='MAVLink target system ID',
    )

    tgt_component_arg = DeclareLaunchArgument(
        'tgt_component',
        default_value='1',
        description='MAVLink target component ID',
    )

    # ── MAVROS node ───────────────────────────────────────────────────────────
    mavros_node = Node(
        package='mavros',
        executable='mavros_node',
        name='mavros',
        namespace='mavros',
        output='screen',
        parameters=[
            {
                'fcu_url': LaunchConfiguration('fcu_url'),
                'gcs_url': LaunchConfiguration('gcs_url'),
                'target_system_id': LaunchConfiguration('tgt_system'),
                'target_component_id': LaunchConfiguration('tgt_component'),
                # SITL-friendly plugin whitelist — extend as needed
                'plugin_allowlist': [
                    'command',
                    'globalposition',
                    'imu',
                    'local_position',
                    'mission',
                    'param',
                    'rc_io',
                    'setpoint_position',
                    'setpoint_velocity',
                    'state',
                    'sys_status',
                    'waypoint',
                ],
            }
        ],
    )

    return LaunchDescription([
        fcu_url_arg,
        gcs_url_arg,
        tgt_system_arg,
        tgt_component_arg,
        mavros_node,
    ])

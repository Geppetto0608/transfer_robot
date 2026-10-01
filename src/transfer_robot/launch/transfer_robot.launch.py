"""ROS 2 Humble bringup for the real 4WS patient-transfer platform.

This launch starts perception, the high-level docking controller, and the small
mode adapter required by motor_node. Motion control always starts disabled.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("transfer_robot")
    detector_config = os.path.join(share, "config", "bed_detector.yaml")
    controller_config = os.path.join(share, "config", "docking_controller.yaml")
    pointcloud_topic = LaunchConfiguration("pointcloud_topic")
    base_frame = LaunchConfiguration("base_frame")
    bed_axis_yaw = ParameterValue(
        LaunchConfiguration("bed_axis_yaw"), value_type=float
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "pointcloud_topic",
                default_value="/livox_merge/merged_pointcloud",
                description="Three-LiDAR merged sensor_msgs/msg/PointCloud2 topic",
            ),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument(
                "bed_axis_yaw",
                default_value="-1.57079632679",
                description="Bed insertion-axis yaw in base_link (-pi/2 = robot right)",
            ),
            # Select the raised top-LiDAR docking TF after the external
            # bringup/top_tf_switcher has had time to create its subscription.
            TimerAction(
                period=1.0,
                actions=[
                    ExecuteProcess(
                        cmd=[
                            "ros2",
                            "topic",
                            "pub",
                            "--once",
                            "/docking",
                            "std_msgs/msg/Bool",
                            "{data: true}",
                        ],
                        output="screen",
                    )
                ],
            ),
            Node(
                package="transfer_robot",
                executable="bed_detector_node",
                name="bed_detector",
                output="screen",
                parameters=[
                    detector_config,
                    {
                        "input_topic": pointcloud_topic,
                        "base_frame": base_frame,
                        "bed_axis_yaw": bed_axis_yaw,
                    },
                ],
            ),
            Node(
                package="transfer_robot",
                executable="wall_detector_node",
                name="wall_detector",
                output="screen",
                # NumPy may otherwise create several BLAS workers for this one
                # process and delay the independent bed detector callback.
                additional_env={
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                parameters=[
                    detector_config,
                    {
                        "base_frame": base_frame,
                        "bed_axis_yaw": bed_axis_yaw,
                    },
                ],
            ),
            Node(
                package="transfer_robot",
                executable="docking_controller_node",
                name="docking_controller",
                output="screen",
                parameters=[
                    controller_config,
                    {
                        "pose_frame": base_frame,
                        "target_bed_yaw": bed_axis_yaw,
                    },
                ],
            ),
            Node(
                package="transfer_robot",
                executable="platform_mode_bridge",
                name="platform_mode_bridge",
                output="screen",
                parameters=[controller_config],
            ),
        ]
    )

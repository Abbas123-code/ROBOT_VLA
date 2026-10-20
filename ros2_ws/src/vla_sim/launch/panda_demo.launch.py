from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("enable_motion", default_value="false"),
        SetEnvironmentVariable("ROS_DOMAIN_ID", "42"),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        Node(package="vla_sim", executable="joint_monitor", namespace="vla_sim", output="screen",
             parameters=[{"use_sim_time": True}]),
        Node(package="vla_sim", executable="joint_commander", namespace="vla_sim", output="screen",
             parameters=[{"use_sim_time": True, "enable_motion": ParameterValue(
                 LaunchConfiguration("enable_motion"), value_type=bool)}]),
    ])
from setuptools import find_packages, setup

setup(
    name="vla_sim",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/vla_sim"]),
        ("share/vla_sim", ["package.xml"]),
        ("share/vla_sim/launch", ["launch/panda_demo.launch.py"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="ROBOT_VLA maintainers",
    maintainer_email="maintainer@example.com",
    description="Simulation-only Panda ROS 2 nodes",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "joint_commander = vla_sim.joint_commander:main",
        "joint_monitor = vla_sim.joint_monitor:main",
        "position_report = vla_sim.position_report:main",
    ]},
)
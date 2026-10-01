from glob import glob
from setuptools import find_packages, setup


package_name = "transfer_robot"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="hyeongjun",
    maintainer_email="phj990608@hanyang.ac.kr",
    description="ROS 2 Humble docking stack for the real patient transfer robot.",
    license="Research-Only Non-Commercial (see LICENSE)",
    entry_points={
        "console_scripts": [
            "bed_detector_node = transfer_robot.node:main",
            "docking_console = transfer_robot.docking_console:main",
            "docking_controller_node = transfer_robot.docking_controller_node:main",
            "platform_mode_bridge = transfer_robot.platform_mode_bridge:main",
            "wall_detector_node = transfer_robot.wall_detector_node:main",
        ]
    },
)

from setuptools import find_packages, setup

package_name = 'grasp_node_py'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros2-grasp-latency',
    maintainer_email='kenny09077@gmail.com',
    description='rclpy grasp node wrapping the Python pipeline at python/grasp_core.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'grasp_node = grasp_node_py.grasp_node:main',
        ],
    },
)

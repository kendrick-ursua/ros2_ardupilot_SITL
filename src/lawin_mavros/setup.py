from setuptools import find_packages, setup

package_name = 'lawin_mavros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cjay',
    maintainer_email='cjay.m.mendoza@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lawin_tracker     = lawin_mavros.lawin_tracker_node:main',
            'flight_controller = lawin_mavros.flight_controller:main',
            'rc_controller     = lawin_mavros.rc_controller:main',
            'test              = lawin_mavros.test:main',
            'zigzag_mission    = lawin_mavros.zigzag_mission:main',
            'oak1_tracker      = lawin_mavros.oak1_tracker_node:main',
            'hover_test        = lawin_mavros.hover_test:main',
            'orig_hover_test = lawin_mavros.orig_hover_test:main',
            'mock_detection = lawin_mavros.mock_detection:main',
            'test_gazebo = lawin_mavros.test_gazebo:main',
        ],
    },
)

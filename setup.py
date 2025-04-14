from setuptools import find_packages, setup

package_name = 'state_machine'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/state_machine.launch.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mcalec',
    maintainer_email='mcalec9999@gmail.com',
    description='Video Board Game State Machine',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_machine = state_machine.state_machine:main'
        ],
    },
)

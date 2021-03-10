#!/usr/bin/env python

from distutils.core import setup

package_name='baldor'
setup(
        name=package_name,
        version='0.1.3',
        description='The baldor package',
        author='Francisco Suarez-Ruiz',
        author_email='fsuarez6@gmail.com',
        url='http://wiki.ros.org/baldor',
        packages=['src/' + package_name],
        data_files=[
            ('share/' + package_name, ['package.xml']),
            ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
            ],
        )

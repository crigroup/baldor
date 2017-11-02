# baldor
[![Build Status](https://travis-ci.org/crigroup/baldor.svg?branch=master)](https://travis-ci.org/crigroup/baldor) [![Coverage Status](https://coveralls.io/repos/github/crigroup/baldor/badge.svg)](https://coveralls.io/github/crigroup/baldor)

ROS (Python) package to work with Homogeneous Transformation Matrices, Quaternions, Euler angles, axis-angle rotations.

This package has been developed by [CRI Group](http://www.ntu.edu.sg/home/cuong/),
[Nanyang Technological University, Singapore](http://www.ntu.edu.sg).

This package is based on:
- [transformations.py](http://www.lfd.uci.edu/~gohlke/code/transformations.py.html) by Christoph Gohlke
- [transforms3d](http://matthew-brett.github.io/transforms3d) by Matthew Brett.

### Maintainer
* [Francisco Suárez Ruiz](http://fsuarez6.github.io)

### Documentation
* Throughout the various files in this repository.
* Website: http://wiki.ros.org/baldor

## ROS Buildfarm

ROS Distro | Source | Debian | Release Status
---------- | ------ | ------ | --------------
indigo | [![Build Status](http://build.ros.org/buildStatus/icon?job=Isrc_uT__baldor__ubuntu_trusty__source)](http://build.ros.org/job/Isrc_uT__baldor__ubuntu_trusty__source) | [![Build Status](http://build.ros.org/buildStatus/icon?job=Ibin_uT64__baldor__ubuntu_trusty_amd64__binary)](http://build.ros.org/job/Ibin_uT64__baldor__ubuntu_trusty_amd64__binary) | <a href="http://repositories.ros.org/status_page/ros_indigo_default.html?q=baldor">Status</a>
kinetic | [![Build Status](http://build.ros.org/buildStatus/icon?job=Ksrc_uX__baldor__ubuntu_xenial__source)](http://build.ros.org/job/Ksrc_uX__baldor__ubuntu_xenial__source/) | [![Build Status](http://build.ros.org/buildStatus/icon?job=Kbin_uX64__baldor__ubuntu_xenial_amd64__binary)](http://build.ros.org/job/Kbin_uX64__baldor__ubuntu_xenial_amd64__binary/) | <a href="http://repositories.ros.org/status_page/ros_kinetic_default.html?q=baldor">Status</a>

Check in the *release status* which versions of the package are in **building**, **ros-shadow-fixed**
(tagged as 'testing'), and **ros** (tagged as 'main').

Approximately every two weeks, the ROS platform manager manually synchronizes
the contents of **ros-shadow-fixed** into **ros** (the public repository).

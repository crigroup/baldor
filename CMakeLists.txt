cmake_minimum_required(VERSION 3.5)
project(baldor)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

# Tests
if (CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(tests)
endif()

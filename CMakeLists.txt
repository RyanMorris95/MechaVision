cmake_minimum_required(VERSION 2.8.4)
PROJECT(Detect)

include(~/dlib-19.2/dlib/cmake)

ADD_EXECUTABLE(Detect detect_obstacles.cpp)
TARGET_LINK_LIBRARIES(Detect dlib)


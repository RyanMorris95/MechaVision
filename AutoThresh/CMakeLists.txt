cmake_minimum_required(VERSION 2.8.4)
PROJECT(AutoThreshold)

include(~/dlib-19.2/dlib/cmake)

ADD_EXECUTABLE(AutoThreshold main.cpp)
TARGET_LINK_LIBRARIES(AutoThreshold dlib)



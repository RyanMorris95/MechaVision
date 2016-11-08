cmake_minimum_required(VERSION 2.8.4)
PROJECT(DetectWithHog)

include(~/dlib-19.2/dlib/cmake)

ADD_EXECUTABLE(DetectWithHog main.cpp)
TARGET_LINK_LIBRARIES(DetectWithHog dlib)


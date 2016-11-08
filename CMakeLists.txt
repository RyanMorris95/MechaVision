cmake_minimum_required(VERSION 2.8.4)
PROJECT(Detect_v2)

include(~/dlib-19.2/dlib/cmake)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


ADD_EXECUTABLE(Detect_v2 detect_obstacles_v2.cpp)
TARGET_LINK_LIBRARIES(Detect_v2 dlib ${OpenCV_LIBS})


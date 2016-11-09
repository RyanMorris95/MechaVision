cmake_minimum_required(VERSION 2.8.4)
PROJECT(Detect)

include(~/dlib-19.2/dlib/cmake)
find_package(OpenCV REQUIRED)

set(YOUR_DIRECTORY ~/Workspace/MechaVision/)
include_directories(${OpenCV_INCLUDE_DIRS})
#set(SOURCES ${YOUR_DIRECTORY}/TestGrabCut.cpp ${YOUR_DIRECTOR}/GrabCut.cpp ${YOUR_DIRECTORY}/GrabCut.h)

ADD_EXECUTABLE(Detect detect_obstacles.cpp)
TARGET_LINK_LIBRARIES(Detect dlib)

ADD_EXECUTABLE(TestGrabCut TestGrabCut.cpp)
TARGET_LINK_LIBRARIES(TestGrabCut ${OpenCV_LIBS})

#ADD_EXECUTABLE(Detect_v2 detect_obstacles_v2.cpp)
#TARGET_LINK_LIBRARIES(Detect_v2 dlib ${OpenCV_LIBS})


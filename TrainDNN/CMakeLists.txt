cmake_minimum_required(VERSION 2.8.4)
PROJECT(DetectObstacles)

include(~/dlib-19.2/dlib/cmake)

ADD_EXECUTABLE(DetectObstacles train_dnn_mmod_obstacles.cpp)
TARGET_LINK_LIBRARIES(DetectObstacles dlib)


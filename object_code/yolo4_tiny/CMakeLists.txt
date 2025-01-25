cmake_minimum_required(VERSION 3.10)
project(YOLOv4TinyObjectDetection)

set(CMAKE_CXX_STANDARD 14)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Source files
set(SOURCES
    src/main.cpp
    src/yolo_utils.cpp
)

# Executable
add_executable(yolov4_tiny ${SOURCES})
target_link_libraries(yolov4_tiny ${OpenCV_LIBS})

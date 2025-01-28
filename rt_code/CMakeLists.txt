cmake_minimum_required(VERSION 3.10)
project(Xenomai_RT_Tasks)

# Specify C++ version
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(include)

# Source files
file(GLOB SOURCES src/*.cpp)

# Link Xenomai libraries
find_library(EVL_LIB evl REQUIRED)

# Add executable
add_executable(rt_tasks ${SOURCES})

# Link the executable with Xenomai EVL library
target_link_libraries(rt_tasks ${EVL_LIB})

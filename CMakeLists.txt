cmake_minimum_required(VERSION 3.18)
project(trtinfer VERSION 0.1.0 LANGUAGES C CXX)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# For finding TensorRT
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(TensorRT_ROOT /opt/TensorRT-8.6.1.6/)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(OpenCV_DIR /usr/local)


# require CUDA, OpenCV, fmt
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(xtensor REQUIRED)

# include headers
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${xtensor_INCLUDE_DIRS} /usr/include include)

file(GLOB SOURCES "src/*.cpp")

add_executable(trtinfer ${SOURCES})

target_link_libraries(trtinfer ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} fmt::fmt xtensor)

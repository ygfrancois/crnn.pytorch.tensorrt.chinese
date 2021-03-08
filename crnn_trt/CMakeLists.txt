cmake_minimum_required(VERSION 2.6)

project(crnn)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# tensorrt include/lib
include_directories(/home/ps/Downloads/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6/TensorRT-7.0.0.11/include)
link_directories(/home/ps/Downloads/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6/TensorRT-7.0.0.11/lib)

add_executable(crnn ${PROJECT_SOURCE_DIR}/crnn.cpp)
# num
# add_executable(crnn ${PROJECT_SOURCE_DIR}/crnn_num.cpp)
# chinese
#add_executable(crnn ${PROJECT_SOURCE_DIR}/crnn_cn.cpp)
target_link_libraries(crnn nvinfer)
target_link_libraries(crnn cudart)
target_link_libraries(crnn ${OpenCV_LIBS})

add_definitions(-O2 -pthread)


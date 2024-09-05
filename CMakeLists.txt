cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Examples VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /root/clh/software/onnxruntime-linux-x64-gpu-1.12.1/include/) 

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /root/clh/software/onnxruntime-linux-x64-gpu-1.12.1/lib/) 
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

add_subdirectory(src)

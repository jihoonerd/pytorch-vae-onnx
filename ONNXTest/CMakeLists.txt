cmake_minimum_required(VERSION 3.29)
project(ONNXTest)

set(CMAKE_CXX_STANDARD 20)

set(ONNX_RUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_pkg/onnxruntime-osx-arm64-1.19.2")

add_executable(ONNXTest main.cpp)

target_include_directories(ONNXTest PRIVATE "${ONNX_RUNTIME_ROOT}/include")
target_link_directories(ONNXTest PRIVATE "${ONNX_RUNTIME_ROOT}/lib")
target_link_libraries(ONNXTest PRIVATE onnxruntime)
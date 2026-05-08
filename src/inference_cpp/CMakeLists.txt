cmake_minimum_required(VERSION 3.15)
project(gesture_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime SDK paths - use absolute path
set(ORT_ROOT "C:/Users/varun/Projects/radar-gesture/radar_gesture_pipeline/onnxruntime_cpp/onnxruntime-win-x64-1.25.1")
set(ORT_INCLUDE "${ORT_ROOT}/include")
set(ORT_LIB "${ORT_ROOT}/lib")

add_executable(gesture_inference
    main.cpp
    inference.cpp
    preprocessor.cpp
)

add_executable(streaming_inference
    streaming_main.cpp
    streaming_inference.cpp
    preprocessor.cpp
)

target_include_directories(gesture_inference PRIVATE ${ORT_INCLUDE})
target_include_directories(streaming_inference PRIVATE ${ORT_INCLUDE})

target_link_libraries(gesture_inference
    ${ORT_LIB}/onnxruntime.lib
)
target_link_libraries(streaming_inference
    ${ORT_LIB}/onnxruntime.lib
)

# Copy DLL to output directory after build
add_custom_command(TARGET gesture_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_LIB}/onnxruntime.dll"
        $<TARGET_FILE_DIR:gesture_inference>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_LIB}/onnxruntime_providers_shared.dll"
        $<TARGET_FILE_DIR:gesture_inference>
)

add_custom_command(TARGET streaming_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_LIB}/onnxruntime.dll"
        $<TARGET_FILE_DIR:streaming_inference>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_LIB}/onnxruntime_providers_shared.dll"
        $<TARGET_FILE_DIR:streaming_inference>
)
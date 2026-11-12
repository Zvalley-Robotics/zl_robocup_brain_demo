# Only include CUDA/TensorRT when NO_CUDA is OFF.
# Non-default installations can override these cache variables, e.g.
#   colcon build --cmake-args -DNO_CUDA=OFF -DCUDA_ROOT=/usr/local/cuda \
#                -DTENSORRT_ROOT=/opt/TensorRT
set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit root")
set(TENSORRT_ROOT "/usr/local/TensorRT-8.6.1.6" CACHE PATH "TensorRT root (x86_64)")
if(NO_CUDA)
  # Build with ONNX
  add_subdirectory(onnx)
else()
  # Build with CUDA/TensorRT
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    include_directories("${CUDA_ROOT}/targets/aarch64-linux/include")
    link_directories("${CUDA_ROOT}/targets/aarch64-linux/lib")
  else()
    # cuda
    include_directories("${CUDA_ROOT}/include")
    link_directories("${CUDA_ROOT}/lib64")

    # tensorrt
    include_directories("${TENSORRT_ROOT}/include")
    link_directories("${TENSORRT_ROOT}/lib")
  endif()

  add_subdirectory(trt)
endif()

add_library(yolov8_model SHARED detector.cc
                                segmentor.cc)

if(NO_CUDA)
  # Link only ONNX when NO_CUDA is ON
  target_link_libraries(yolov8_model PUBLIC yolov8_onnx
                                          ${YAML_CPP_LIBRARIES})
else()
  # Link TensorRT when CUDA is available
  target_link_libraries(yolov8_model PUBLIC yolov8_trt
                                          ${YAML_CPP_LIBRARIES})
endif()
target_compile_options(yolov8_model PRIVATE -w)

install(TARGETS yolov8_model DESTINATION lib)
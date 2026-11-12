# add_definitions(-DAPI_EXPORTS)

if(NOT DEFINED CUDA_ROOT)
  set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit root")
endif()
set(CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc")
enable_language(CUDA)

set(SRCS block.cpp
         calibrator.cpp
         impl.cpp
         model.cpp
         postprocess.cpp
         postprocess.cu
         preprocess.cu
         yololayer.cu
         )

add_library(yolov8_trt SHARED ${SRCS})
target_link_libraries(yolov8_trt PRIVATE nvinfer
                                         cudart
                                 PUBLIC ${OpenCV_LIBS})
target_compile_options(yolov8_trt PRIVATE -w -Wno-dev)
install(TARGETS yolov8_trt DESTINATION lib)
                              
add_executable(yolov8_det yolov8_det.cpp)
target_link_libraries(yolov8_det yolov8_trt nvinfer cudart)
target_compile_options(yolov8_det PRIVATE -w)

add_executable(yolov8_seg yolov8_seg.cpp)
target_link_libraries(yolov8_seg yolov8_trt nvinfer cudart)
target_compile_options(yolov8_seg PRIVATE -w)

install(TARGETS yolov8_det
                yolov8_seg
        DESTINATION lib/${PROJECT_NAME})
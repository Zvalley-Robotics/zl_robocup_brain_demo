add_library(robot_vision_base SHARED intrin.cpp
                                       pose.cpp
                                       pointcloud_process.cpp
                                       data_syncer.cpp
                                       )
target_include_directories(robot_vision_base PUBLIC ${geometry_msgs_INCLUDE_DIRS}
                                                      ${tf2_INCLUDE_DIRS})

ament_target_dependencies(robot_vision_base PUBLIC  builtin_interfaces
                                                      geometry_msgs
                                                      tf2)

target_link_libraries(robot_vision_base PUBLIC ${OpenCV_LIBRARIES}
                                                 ${YAML_CPP_LIBRARIES}
                                                 ${PCL_LIBRARIES}
                                                #  builtin_interfaces
                                                #  geometry_msgs
                                                 )

target_compile_options(robot_vision_base PRIVATE -Wno-dev)

install(TARGETS robot_vision_base DESTINATION lib)
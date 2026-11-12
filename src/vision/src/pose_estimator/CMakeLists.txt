add_library(robot_vision_pose_estimator SHARED pose_estimator.cpp)
target_link_libraries(robot_vision_pose_estimator PUBLIC ${OpenCV_LIBRARIES}
                                                           ${YAML_CPP_LIBRARIES}
                                                           robot_vision_base
                                                           yolov8_model)
install(TARGETS 
  robot_vision_pose_estimator
  # DESTINATION lib/${PROJECT_NAME})
  DESTINATION lib)

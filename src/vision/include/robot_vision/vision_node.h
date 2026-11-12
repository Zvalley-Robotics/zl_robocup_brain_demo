#pragma once

#include <memory>
#include <map>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <image_transport/image_transport.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "vision_interface/msg/detections.hpp"

#include "vision_interface/msg/line_segments.hpp"
#include "vision_interface/msg/ball.hpp"

#include <yaml-cpp/yaml.h>

#include "robot_vision/base/intrin.h"
#include "robot_vision/base/pose.h"

#include "robot_vision/color_classifier.hpp"

namespace robot_vision {

class DataLogger;
class DataSyncer;
class PoseEstimator;
class YoloV8Detector;
class YoloV8Segmentor;
class SyncedDataBlock;

class VisionNode : public rclcpp::Node {
public:
    VisionNode(const std::string &node_name, const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
    ~VisionNode() = default;

    // cfg_paths 按顺序逐层覆盖 cfg_template_path；不存在的层会被跳过。
    void Init(const std::string &cfg_template_path, const std::vector<std::string> &cfg_paths);
    void ColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
    void CompressedColorCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    void SegmentationCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
    void CompressedSegmentationCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    void DepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
    void CompressedDepthCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    void PoseTFCallBack(const geometry_msgs::msg::TransformStamped::SharedPtr msg);
    // /robot/head_to_base 自 2026-11-11 起为 geometry_msgs/PoseStamped。
    void PoseCallBack(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void CameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void ProcessData(SyncedDataBlock &synced_data, vision_interface::msg::Detections &detections);
    void ProcessSegmentationData(SyncedDataBlock &synced_data, vision_interface::msg::LineSegments &field_line_segs_msg);

private:
    bool use_depth_ = false;
    bool show_det_ = false;
    bool show_seg_ = false;
    bool publish_det_img_ = false;
    int publish_det_img_fps_ = 15;
    int publish_det_img_quality_ = 70;
    double publish_det_img_scale_ = 1.0;
    double det_img_last_ts_ = -1.0;
    bool save_data_ = false;
    bool save_depth_ = false;
    bool offline_mode_ = false;
    std::string detection_model_path;
    std::string segmentation_model_path;

    int save_cnt_ = 0;
    int save_every_n_frame_ = 0;

    std::string color_topic_;
    std::string depth_topic_;
    std::string intrin_topic_;
    std::string img_log_path_;

    Intrinsics intr_;
    // Positive height of the waist/base origin above the foot origin. Points
    // expressed in the foot frame are converted to base with z_base = z_foot - height.
    float base_height_above_foot_ = 0.83F;
    // 场地线分割类别过滤：>=0 只取该 class_id 的轮廓做拟合；-1 = 不过滤（单类模型）。
    // 缺省 1：joint_test/仿真用多类模型（sim_data_seg_0126）时必须按类过滤。
    int field_line_class_id_ = 1;
    int line_segment_area_threshold_ = 10; // threshold for line segment detection

    // post processing
    bool enable_post_process_ = false;
    bool single_ball_assumption_ = false;
    std::vector<std::string> classnames_;
    // std::vector<float> duration_list_;
    // std::vector<std::string> duration_name_list_;
    std::map<std::string, float> confidence_map_;
    // std::string profiler_log_path_;
    // bool first_write_profiler_log_ = true;


    std::shared_ptr<rclcpp::Node> nh_;
    rclcpp::Publisher<vision_interface::msg::Detections>::SharedPtr detection_pub_;
    rclcpp::Publisher<vision_interface::msg::LineSegments>::SharedPtr field_line_pub_;
    rclcpp::Publisher<vision_interface::msg::Ball>::SharedPtr ball_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr det_img_pub_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detection_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr segmentation_img_pub_;

    rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr pose_tf_pub_;
    rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr pose_tf_sub_;

    std::shared_ptr<image_transport::ImageTransport> it_;
    image_transport::Subscriber color_sub_;
    image_transport::Subscriber depth_sub_;
    image_transport::Subscriber color_seg_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_color_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_color_seg_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

    rclcpp::CallbackGroup::SharedPtr callback_group_sub_1_;
    rclcpp::CallbackGroup::SharedPtr callback_group_sub_2_;
    rclcpp::CallbackGroup::SharedPtr callback_group_sub_3_;
    rclcpp::CallbackGroup::SharedPtr callback_group_sub_4_;

    std::shared_ptr<ColorClassifier> color_classifier_;

    std::shared_ptr<DataLogger> data_logger_;
    std::shared_ptr<DataSyncer> data_syncer_;
    std::shared_ptr<DataSyncer> seg_data_syncer_;
    std::shared_ptr<YoloV8Detector> detector_;
    std::shared_ptr<YoloV8Segmentor> segmentor_;
    std::shared_ptr<PoseEstimator> pose_estimator_;
    std::map<std::string, std::shared_ptr<PoseEstimator>> pose_estimator_map_;
    
    YAML::Node config_node_;  // Store config for recreating pose estimators
};

} // namespace robot_vision

#include "robot_vision/vision_node.h"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <functional>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <fstream>

#include <yaml-cpp/yaml.h>
#include "ament_index_cpp/get_package_share_directory.hpp"

#include "vision_interface/msg/detected_object.hpp"
#include "vision_interface/msg/detections.hpp"

#include "robot_vision/base/data_syncer.hpp"
#include "robot_vision/base/data_logger.hpp"
#include "robot_vision/base/misc_utils.hpp"
#include "robot_vision/model/detector.h"
#include "robot_vision/model/segmentor.h"
#include "robot_vision/pose_estimator/pose_estimator.h"
#include "robot_vision/img_bridge.h"

namespace robot_vision {

namespace {

bool RejectProjectedBallSize(const DetectionRes &detection, const Pose &eye_to_foot,
                             const Pose &projected_ball_foot, const Intrinsics &intr,
                             float expected_radius, float radius_tolerance) {
    const auto xyz_foot = projected_ball_foot.getTranslationVec();
    const cv::Point3f point_camera = eye_to_foot.inverse() *
        cv::Point3f(xyz_foot[0], xyz_foot[1], xyz_foot[2]);
    const float z = point_camera.z;
    if (!std::isfinite(z) || z <= 0.0F || !std::isfinite(intr.fx) || intr.fx <= 0.0F ||
        !std::isfinite(intr.fy) || intr.fy <= 0.0F ||
        detection.bbox.width <= 0 || detection.bbox.height <= 0) {
        return false;
    }

    // Use the longer observed side for both edge and occluded detections.
    const float estimated_radius = 0.5F * z * std::max(
        detection.bbox.width / intr.fx, detection.bbox.height / intr.fy);
    if (!std::isfinite(estimated_radius)) return false;
    return std::abs(estimated_radius - expected_radius) > radius_tolerance;
}

void AdjustEdgeBallCenter(const DetectionRes &detection, const cv::Size &image_size,
                          const Intrinsics &intr, float radius, float radius_tolerance,
                          cv::Point3f &center_camera) {
    const cv::Rect bbox = detection.bbox & cv::Rect(0, 0, image_size.width, image_size.height);
    if (bbox.empty() || !std::isfinite(center_camera.x) || !std::isfinite(center_camera.y) ||
        !std::isfinite(center_camera.z) || center_camera.z <= 0.0F ||
        !std::isfinite(intr.fx) || intr.fx <= 0.0F ||
        !std::isfinite(intr.fy) || intr.fy <= 0.0F) return;

    const bool left = bbox.x <= 0;
    const bool right = bbox.x + bbox.width >= image_size.width - 1;
    const bool top = bbox.y <= 0;
    const bool bottom = bbox.y + bbox.height >= image_size.height - 1;
    const bool horizontal_edge = left != right;
    const bool vertical_edge = top != bottom;
    if (!horizontal_edge && !vertical_edge) return;

    const float visible_width = bbox.width * center_camera.z / intr.fx;
    const float visible_height = bbox.height * center_camera.z / intr.fy;
    // The unclipped (long) side must still resemble a full ball diameter.
    const bool horizontal_short_side = visible_width <= visible_height;
    if ((horizontal_short_side && !horizontal_edge) ||
        (!horizontal_short_side && !vertical_edge) ||
        std::abs(0.5F * std::max(visible_width, visible_height) - radius) > radius_tolerance) return;

    const cv::Point2f center_uv = intr.Project(center_camera);
    if (!std::isfinite(center_uv.x) || !std::isfinite(center_uv.y)) return;

    if (horizontal_short_side) {
        const float inner_u = left ? static_cast<float>(bbox.x + bbox.width)
                                   : static_cast<float>(bbox.x);
        const float inner_x = intr.BackProject(cv::Point2f(inner_u, center_uv.y), center_camera.z).x;
        if (!std::isfinite(inner_x)) return;
        const float distance = left ? inner_x - center_camera.x : center_camera.x - inner_x;
        if (distance < radius) center_camera.x = left ? inner_x - radius : inner_x + radius;
    } else {
        const float inner_v = top ? static_cast<float>(bbox.y + bbox.height)
                                  : static_cast<float>(bbox.y);
        const float inner_y = intr.BackProject(cv::Point2f(center_uv.x, inner_v), center_camera.z).y;
        if (!std::isfinite(inner_y)) return;
        const float distance = top ? inner_y - center_camera.y : center_camera.y - inner_y;
        if (distance < radius) center_camera.y = top ? inner_y - radius : inner_y + radius;
    }
}

bool EstimateDepthAtPixel(const cv::Mat &depth, const cv::Point2f &uv, const Intrinsics &intr,
                          cv::Point3f &position_camera) {
    if (depth.empty() || depth.type() != CV_32FC1 ||
        !std::isfinite(uv.x) || !std::isfinite(uv.y)) return false;
    if (uv.x < 0 || uv.y < 0 || uv.x > depth.cols || uv.y > depth.rows) return false;

    const int u = std::clamp(cvRound(uv.x), 0, depth.cols - 1);
    const int v = std::clamp(cvRound(uv.y), 0, depth.rows - 1);
    const cv::Rect sample = cv::Rect(u - 2, v - 2, 5, 5) &
                            cv::Rect(0, 0, depth.cols, depth.rows);
    std::vector<float> depths;
    for (int y = sample.y; y < sample.y + sample.height; ++y) {
        for (int x = sample.x; x < sample.x + sample.width; ++x) {
            const float value = depth.at<float>(y, x);
            if (std::isfinite(value) && value > 0.05F && value < 20.0F) depths.push_back(value);
        }
    }
    if (depths.empty()) return false;
    auto middle = depths.begin() + depths.size() / 2;
    std::nth_element(depths.begin(), middle, depths.end());
    position_camera = intr.BackProject(uv, *middle);
    return true;
}

} // namespace

VisionNode::VisionNode(const std::string &node_name, const rclcpp::NodeOptions &options) :
    rclcpp::Node(node_name, options) {
    this->declare_parameter<bool>("offline_mode", false);
    this->declare_parameter<bool>("show_det", false);
    this->declare_parameter<bool>("show_seg", false);
    this->declare_parameter<bool>("publish_det_img", false);
    this->declare_parameter<int>("publish_det_img_fps", 15);
    this->declare_parameter<int>("publish_det_img_quality", 70);
    this->declare_parameter<double>("publish_det_img_scale", 1.0);
    this->declare_parameter<bool>("save_data", true);
    this->declare_parameter<bool>("save_depth", true);
    this->declare_parameter<int>("save_fps", 3);
    this->declare_parameter<std::string>("robot_name", "");
    this->declare_parameter<std::string>("detection_model_path", "");
    this->declare_parameter<std::string>("segmentation_model_path", "");
    this->declare_parameter<std::string>("color_topic", "");
    this->declare_parameter<std::string>("depth_topic", "");
    this->declare_parameter<std::string>("intrin_topic", "");
}

// TODO(GW): oneline offline
void VisionNode::Init(const std::string &cfg_template_path,
                      const std::vector<std::string> &cfg_paths) {
    if (!std::filesystem::exists(cfg_template_path)) {
        // TODO(SS): throw exception here
        std::cerr << "Error: Configuration template file '" << cfg_template_path << "' does not exist." << std::endl;
        return;
    }

    YAML::Node node = YAML::LoadFile(cfg_template_path);
    for (const auto &cfg_path : cfg_paths) {
        if (cfg_path.empty() || !std::filesystem::exists(cfg_path)) {
            std::cout << "Optional config layer not found, skipped: " << cfg_path << std::endl;
            continue;
        }
        YAML::Node cfg_node = YAML::LoadFile(cfg_path);
        // merge input cfg to template cfg
        MergeYAML(node, cfg_node);
        std::cout << "merged config layer: " << cfg_path << std::endl;
    }

    // 模型路径允许写成相对路径：先按当前工作目录找，找不到再到 vision 包
    // share 目录下找。这样从任意目录 ros2 launch 都能加载随包安装的模型。
    auto resolve_model_path = [](YAML::Node model) {
        if (!model || !model["model_path"]) return;
        const std::string path = model["model_path"].as<std::string>();
        if (path.empty() || path.front() == '/' || std::filesystem::exists(path)) return;
        const auto share = ament_index_cpp::get_package_share_directory("vision");
        const auto candidate = std::filesystem::path(share) / path;
        if (std::filesystem::exists(candidate)) {
            model["model_path"] = candidate.string();
        }
    };
    resolve_model_path(node["detection_model"]);
    resolve_model_path(node["segmentation_model"]);

    std::cout << "loaded file: " << std::endl
              << node << std::endl;

    // Store config node for later use (e.g., when updating intrinsics)
    config_node_ = node;

    this->get_parameter<bool>("show_det", show_det_);
    this->get_parameter<bool>("show_seg", show_seg_);
    this->get_parameter<bool>("publish_det_img", publish_det_img_);
    this->get_parameter<int>("publish_det_img_fps", publish_det_img_fps_);
    this->get_parameter<int>("publish_det_img_quality", publish_det_img_quality_);
    this->get_parameter<double>("publish_det_img_scale", publish_det_img_scale_);
    publish_det_img_fps_ = std::max(1, publish_det_img_fps_);
    publish_det_img_quality_ = std::clamp(publish_det_img_quality_, 1, 100);
    publish_det_img_scale_ = std::clamp(publish_det_img_scale_, 0.2, 1.0);
    this->get_parameter<bool>("save_data", save_data_);
    this->get_parameter<bool>("save_depth", save_depth_);
    this->get_parameter<bool>("offline_mode", offline_mode_);
    this->get_parameter<std::string>("color_topic", color_topic_);
    this->get_parameter<std::string>("depth_topic", depth_topic_);
    this->get_parameter<std::string>("intrin_topic", intrin_topic_);
    this->get_parameter<std::string>("detection_model_path", detection_model_path);
    std::cout << "detection_model_path origin: " << detection_model_path << std::endl;

    if(!detection_model_path.empty()){
        if(detection_model_path[0] == '/') {
            // absolute path, do nothing
        } else {
            std::string package_path = ament_index_cpp::get_package_share_directory("vision");
            detection_model_path = (std::filesystem::path(package_path) / detection_model_path).string();
        }
    }


    this->get_parameter<std::string>("segmentation_model_path", segmentation_model_path);
    std::cout << "segmentation_model_path origin: " << segmentation_model_path << std::endl;

    if(!segmentation_model_path.empty()){
        if(segmentation_model_path[0] == '/') {
            // absolute path, do nothing
        } else {
            std::string package_path = ament_index_cpp::get_package_share_directory("vision");
            segmentation_model_path = (std::filesystem::path(package_path) / segmentation_model_path).string();
        }
    }
    
    int save_fps = 0;
    this->get_parameter<int>("save_fps", save_fps);
    save_depth_ = save_depth_ && save_data_;
    std::cout << "offline_mode: " << offline_mode_ << std::endl;
    std::cout << "show_det: " << show_det_ << std::endl;
    std::cout << "show_seg: " << show_seg_ << std::endl;
    std::cout << "save_data: " << save_data_ << std::endl;
    std::cout << "save_depth: " << save_depth_ << std::endl;
    std::cout << "save_fps: " << save_fps << std::endl;
    std::cout << "color_topic: " << color_topic_ << std::endl;
    std::cout << "depth_topic: " << depth_topic_ << std::endl;
    std::cout << "intrin_topic: " << intrin_topic_ << std::endl;
    std::cout << "detection_model_path: " << detection_model_path << std::endl;
    std::cout << "segmentation_model_path: " << segmentation_model_path << std::endl;
    save_every_n_frame_ = std::max(1, save_fps > 0 ? 30 / save_fps : 1);
    std::cout << "save_every_n_frame: " << save_every_n_frame_ << std::endl;

    // read camera param
    if (!node["camera"]) {
        throw std::runtime_error("no camera param found here");
    } else {
        if (color_topic_.empty())
        {
            color_topic_ = node["camera"]["color_topic"].as<std::string>();
        }
        if (intrin_topic_.empty())
        {
            intrin_topic_ = node["camera"]["intrin_topic"].as<std::string>();
        }
        if (depth_topic_.empty())
        {
            depth_topic_ = node["camera"]["depth_topic"].as<std::string>();
        }
        intr_ = Intrinsics(node["camera"]["intrin"]);
        base_height_above_foot_ = as_or<float>(node["camera"]["base_height_above_foot"], 0.83F);
        std::cout << "base height above foot: " << base_height_above_foot_ << " m" << std::endl;
    }

    // init detector
    if (!node["detection_model"]) {
        std::cerr << "no detection model param here" << std::endl;
        return;
    } else {
        detector_ = YoloV8Detector::CreateYoloV8Detector(node["detection_model"], detection_model_path);
        classnames_ = node["detection_model"]["classnames"].as<std::vector<std::string>>();
        // detector post processing
        float default_threshold = as_or<float>(node["detection_model"]["confidence_threshold"], 0.2);
        if (node["detection_model"]["post_process"]) {
            enable_post_process_ = true;
            single_ball_assumption_ = as_or<bool>(node["detection_model"]["post_process"]["single_ball_assumption"], false);
            if (node["detection_model"]["post_process"]["confidence_thresholds"]) {
                for (const auto &item : node["detection_model"]["post_process"]["confidence_thresholds"]) {
                    confidence_map_[item.first.as<std::string>()] = item.second.as<float>();
                }
                // set default confidence for other classes
                for (const auto &classname : classnames_) {
                    if (confidence_map_.find(classname) == confidence_map_.end()) {
                        confidence_map_[classname] = default_threshold;
                    }
                }
            } else {
                std::cout << "all class apply same default threshold: " << default_threshold << std::endl;
            }
        }
    }

    if (!node["segmentation_model"]) {
        std::cerr << "no segmentation model param found" << std::endl;
    } else {
        segmentor_ = YoloV8Segmentor::CreateYoloV8Segmentor(node["segmentation_model"], segmentation_model_path);
        // -1 = 不按类别过滤（seg_261112 单类模型）；缺省 1 兼容 joint_test/仿真的多类 ONNX 模型。
        field_line_class_id_ = as_or<int>(node["segmentation_model"]["field_line_class_id"], 1);
        if (field_line_class_id_ < -1) {
            throw std::runtime_error("segmentation_model.field_line_class_id must be >= -1 (-1 = accept all)");
        }
    }

    // add detector_ warmup

    // init data_syncer
    use_depth_ = as_or<bool>(node["use_depth"], false);
    data_syncer_ = std::make_shared<DataSyncer>(use_depth_);
    bool save_data_nonstationary = as_or<bool>(node["misc"]["save_data_nonstationary"], true);
    std::string log_root = std::string(std::getenv("HOME")) + "/Workspace/vision_log/" + getTimeString();
    data_logger_ = save_data_ ? std::make_shared<DataLogger>(log_root, save_data_nonstationary) : nullptr;
    if (data_logger_) {
        data_logger_->LogYAML(node, "vision_local.yaml");
    }
    seg_data_syncer_ = std::make_shared<DataSyncer>(false);


    std::string robot_name = as_or<std::string>(node["robot_name"], "");
    std::string topic_suffix = "";
    if (robot_name.empty()) {
        robot_name = "robot0";
        topic_suffix = "";
    } else {
        topic_suffix = "/" + robot_name;
    }

    // init robot color classifier
    if (node["robot_color_classifier"]) {
        color_classifier_ = std::make_shared<ColorClassifier>();
        color_classifier_->Init(node["robot_color_classifier"]);
    }

    // init pose estimator
    pose_estimator_ = std::make_shared<PoseEstimator>(intr_);
    pose_estimator_->Init(YAML::Node());
    pose_estimator_map_["default"] = pose_estimator_;

    if (node["ball_pose_estimator"]) {
        pose_estimator_map_["ball"] = std::make_shared<BallPoseEstimator>(intr_);
        pose_estimator_map_["ball"]->Init(node["ball_pose_estimator"]);
    }

    if (node["human_like_pose_estimator"]) {
        pose_estimator_map_["human_like"] = std::make_shared<HumanLikePoseEstimator>(intr_);
        pose_estimator_map_["human_like"]->Init(node["human_like_pose_estimator"]);
    }

    if (node["field_marker_pose_estimator"]) {
        pose_estimator_map_["field_marker"] = std::make_shared<FieldMarkerPoseEstimator>(intr_);
        pose_estimator_map_["field_marker"]->Init(node["field_marker_pose_estimator"]);

        line_segment_area_threshold_ = as_or<int>(node["field_marker_pose_estimator"]["line_segment_area_threshold"], 75);
    }

    // init ros related

    if (color_topic_.find("robot0_rgbd_camera") != std::string::npos && !robot_name.empty()) { // sim
        color_topic_ = color_topic_.replace(color_topic_.find("robot0_rgbd_camera"), std::string("robot0_rgbd_camera").length(), robot_name + "_rgbd_camera");
    }
    if (depth_topic_.find("robot0_rgbd_camera") != std::string::npos && !robot_name.empty()) { // sim
        depth_topic_ = depth_topic_.replace(depth_topic_.find("robot0_rgbd_camera"), std::string("robot0_rgbd_camera").length(), robot_name + "_rgbd_camera");
    }
    if (intrin_topic_.find("robot0_rgbd_camera") != std::string::npos && !robot_name.empty()) { // sim
        intrin_topic_ = intrin_topic_.replace(intrin_topic_.find("robot0_rgbd_camera"), std::string("robot0_rgbd_camera").length(), robot_name + "_rgbd_camera");
    }
    

    // Subscribe to camera info to update intrinsics dynamically
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        intrin_topic_, rclcpp::QoS(1).best_effort(),
        std::bind(&VisionNode::CameraInfoCallback, this, std::placeholders::_1));
    std::cout << "Subscribing to camera info topic: " << intrin_topic_ << std::endl;

    callback_group_sub_1_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_sub_2_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_sub_3_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_sub_4_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    auto sub_opt_1 = rclcpp::SubscriptionOptions();
    sub_opt_1.callback_group = callback_group_sub_1_;
    auto sub_opt_2 = rclcpp::SubscriptionOptions();
    sub_opt_2.callback_group = callback_group_sub_2_;
    auto sub_opt_3 = rclcpp::SubscriptionOptions();
    sub_opt_3.callback_group = callback_group_sub_3_;
    auto sub_opt_4 = rclcpp::SubscriptionOptions();
    sub_opt_4.callback_group = callback_group_sub_4_;

    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    image_transport::TransportHints hints(this, "compressed");
    
    // Subscribe to color image (raw or compressed)
    if (color_topic_.find("compressed") != std::string::npos) {
        std::cout << "Subscribing to compressed color topic: " << color_topic_ << std::endl;
        compressed_color_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            color_topic_, 1, std::bind(&VisionNode::CompressedColorCallback, this, std::placeholders::_1), sub_opt_1);
    } else {
        std::cout << "Subscribing to raw color topic: " << color_topic_ << std::endl;
        color_sub_ = it_->subscribe(color_topic_, 1, &VisionNode::ColorCallback, this, nullptr, sub_opt_1);
    }
    
    // Subscribe to depth image (raw or compressed)
    if (use_depth_ && depth_topic_.find("compressed") != std::string::npos) {
        std::cout << "Subscribing to compressed depth topic: " << depth_topic_ << std::endl;
        compressed_depth_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            depth_topic_, 1, std::bind(&VisionNode::CompressedDepthCallback, this, std::placeholders::_1), sub_opt_3);
    } else if (use_depth_) {
        std::cout << "Subscribing to raw depth topic: " << depth_topic_ << std::endl;
        depth_sub_ = it_->subscribe(depth_topic_, 1, &VisionNode::DepthCallback, this, nullptr, sub_opt_3);
    }

    detection_pub_ = this->create_publisher<vision_interface::msg::Detections>("/soccer/detection" + topic_suffix, rclcpp::QoS(1));

    if (node["segmentation_model"]) {
        std::cout << "create sub for segmentor" << std::endl;
        if (color_topic_.find("compressed") != std::string::npos) {
            compressed_color_seg_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
                color_topic_, 1, std::bind(&VisionNode::CompressedSegmentationCallback, this, std::placeholders::_1), sub_opt_2);
        } else {
            color_seg_sub_ = it_->subscribe(color_topic_, 1, &VisionNode::SegmentationCallback, this, nullptr, sub_opt_2);
        }
        field_line_pub_ = this->create_publisher<vision_interface::msg::LineSegments>("/soccer/line_segments" + topic_suffix, rclcpp::QoS(1));
    }
    ball_pub_ = this->create_publisher<vision_interface::msg::Ball>("/soccer/ball" + topic_suffix, rclcpp::QoS(1));
    if (publish_det_img_) {
        det_img_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
            "/soccer/debug/detection_img" + topic_suffix, rclcpp::QoS(1));
    }

    if (offline_mode_) {
        pose_tf_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>("/soccer/t_eye2foot" + topic_suffix, 10, std::bind(&VisionNode::PoseTFCallBack, this, std::placeholders::_1));
    } else {
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("/robot/head_to_base" + topic_suffix, 10, std::bind(&VisionNode::PoseCallBack, this, std::placeholders::_1), sub_opt_4);
        pose_tf_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/soccer/t_eye2foot" + topic_suffix, rclcpp::QoS(10));
    }
}

void VisionNode::ProcessData(SyncedDataBlock &synced_data, vision_interface::msg::Detections &detection_msg) {
    static uint64_t detection_frame_count = 0;
    ++detection_frame_count;
    const bool print_camera_positions = (detection_frame_count % 10 == 0);

    double timestamp = synced_data.color_data.timestamp;
    double depth_time_diff = (timestamp - synced_data.depth_data.timestamp) * 1000;
    double pose_time_diff = (timestamp - synced_data.pose_data.timestamp) * 1000;
    if (use_depth_ && depth_time_diff > 40) {
        std::cerr << "color depth time diff: " << depth_time_diff << "ms" << std::endl;
    }
    if (pose_time_diff > 40) {
        std::cerr << "color pose time diff: " << pose_time_diff << " ms" << std::endl;
    }
    cv::Mat color = synced_data.color_data.data;
    cv::Mat depth = synced_data.depth_data.data;

    cv::Mat depth_float;
    if (!depth.empty() && depth.depth() == CV_16U) {
        depth.convertTo(depth_float, CV_32F, 0.001, 0);
    } else {
        depth_float = depth;
    }

    // The received transform maps camera optical coordinates directly to base.
    // Restore camera -> foot for intersections with the ground plane (z = 0).
    Pose p_eye2base = synced_data.pose_data.data;
    Pose p_base2foot(0.0F, 0.0F, base_height_above_foot_, 0.0F, 0.0F, 0.0F);
    Pose p_eye2foot = p_base2foot * p_eye2base;
    Pose p_foot2eye = p_eye2foot.inverse();
    Pose p_foot2base(0.0F, 0.0F, -base_height_above_foot_, 0.0F, 0.0F, 0.0F);
    std::cout << "det: p_eye2foot: \n"
              << p_eye2foot.toCVMat() << std::endl;
    std::cout << "det: p_eye2base: \n"
              << p_eye2base.toCVMat() << std::endl;

    // inference
    auto detections = detector_->Inference(color);
    std::cout << detections.size() << " objects detected." << std::endl;

    auto get_estimator = [&](const std::string &class_name) {
        if (class_name == "Ball") {
            return pose_estimator_map_.find("ball") != pose_estimator_map_.end() ? pose_estimator_map_["ball"] : pose_estimator_map_["default"];
        } else if (class_name == "Person" || class_name == "Opponent" || class_name == "Goalpost") {
            return pose_estimator_map_.find("human_like") != pose_estimator_map_.end() ? pose_estimator_map_["human_like"] : pose_estimator_map_["default"];
        } else if (class_name.find("Cross") != std::string::npos || class_name == "PenaltyPoint") {
            return pose_estimator_map_.find("field_marker") != pose_estimator_map_.end() ? pose_estimator_map_["field_marker"] : pose_estimator_map_["default"];
        } else {
            return pose_estimator_map_["default"];
        }
    };

    std::vector<robot_vision::DetectionRes> filtered_detections;
    if (enable_post_process_ && !detections.empty()) {
        // filter detections with different confidence
        if (!confidence_map_.empty()) {
            for (auto &detection : detections) {
                auto classname = classnames_[detection.class_id];
                if (detection.confidence < confidence_map_[classname]) {
                    continue;
                }
                filtered_detections.push_back(detection);
            }
        } else {
            filtered_detections = detections;
        }

        // keep the highest ball detections
        if (single_ball_assumption_) {
            std::vector<robot_vision::DetectionRes> ball_detections;
            std::vector<robot_vision::DetectionRes> filtered_detections_bk = filtered_detections;
            filtered_detections.clear();

            for (const auto &detection : filtered_detections_bk) {
                if (classnames_[detection.class_id] == "Ball") {
                    ball_detections.push_back(detection);
                } else {
                    filtered_detections.push_back(detection);
                }
            }

            if (ball_detections.size() > 1) {
                std::cout << "Multiple ball detections found, keeping the one with highest confidence." << std::endl;
                auto max_ball_detection = *std::max_element(ball_detections.begin(), ball_detections.end(),
                                                            [](const robot_vision::DetectionRes &a, const robot_vision::DetectionRes &b) {
                                                                return a.confidence < b.confidence;
                                                            });
                filtered_detections.push_back(max_ball_detection);
            } else {
                filtered_detections.insert(filtered_detections.end(), ball_detections.begin(), ball_detections.end());
            }
        }
    } else {
        filtered_detections = detections;
    }

    // Publish one Ball message for every detection frame. A zero-confidence
    // message represents "no ball in this frame". Use the same
    // ground-plane projection for distant balls and a depth-derived ball center
    // in base coordinates for nearby balls.
    vision_interface::msg::Ball ball_msg;
    ball_msg.header = detection_msg.header;
    ball_msg.x = 0.0;
    ball_msg.y = 0.0;
    ball_msg.confidence = 0.0;
    constexpr double kBallMinX = -0.5;
    constexpr double kBallMaxX = 15.0;
    constexpr double kBallConfMargin = 15.0;
    constexpr double kBallRangeHyst = 1.0;
    bool has_ball_pick = false;
    double best_ball_conf = 0.0;
    double best_ball_range = 0.0;

    std::vector<robot_vision::DetectionRes> detections_for_display;
    for (auto &detection : filtered_detections) {
        vision_interface::msg::DetectedObject detection_obj;

        detection.class_name = detector_->kClassLabels[detection.class_id];

        // Estimate a target position directly in camera_optical_frame. The
        // detector runs on the color image, so this requires a depth image that
        // is registered/aligned to that image. A median over the central half
        // of the bounding box is more stable than one centre pixel.
        if (print_camera_positions && detection.class_name == "Ball") {
            cv::Point3f point_camera;
            bool has_camera_position = false;

            if (!depth_float.empty() &&
                depth_float.type() == CV_32FC1 &&
                depth_float.size() == color.size()) {
                cv::Rect image_bounds(0, 0, depth_float.cols, depth_float.rows);
                cv::Rect bbox = detection.bbox & image_bounds;

                if (bbox.width > 0 && bbox.height > 0) {
                    const int margin_x = bbox.width / 4;
                    const int margin_y = bbox.height / 4;
                    cv::Rect sample_roi(
                        bbox.x + margin_x,
                        bbox.y + margin_y,
                        std::max(1, bbox.width - 2 * margin_x),
                        std::max(1, bbox.height - 2 * margin_y));
                    sample_roi &= image_bounds;

                    std::vector<float> valid_depths;
                    valid_depths.reserve(sample_roi.area());
                    for (int v = sample_roi.y; v < sample_roi.y + sample_roi.height; ++v) {
                        for (int u = sample_roi.x; u < sample_roi.x + sample_roi.width; ++u) {
                            const float depth_m = depth_float.at<float>(v, u);
                            if (std::isfinite(depth_m) && depth_m > 0.05F && depth_m < 20.0F) {
                                valid_depths.push_back(depth_m);
                            }
                        }
                    }

                    if (!valid_depths.empty()) {
                        const size_t median_index = valid_depths.size() / 2;
                        std::nth_element(valid_depths.begin(),
                                         valid_depths.begin() + median_index,
                                         valid_depths.end());
                        const float median_depth = valid_depths[median_index];
                        const cv::Point2f target_uv(
                            bbox.x + bbox.width * 0.5F,
                            bbox.y + bbox.height * 0.5F);
                        point_camera = intr_.BackProject(target_uv, median_depth);
                        has_camera_position = true;
                    }
                }
            }

            if (has_camera_position) {
                std::clog << "[camera_optical_frame] class=Ball"
                          << " x=" << point_camera.x << " m"
                          << " y=" << point_camera.y << " m"
                          << " z=" << point_camera.z << " m"
                          << std::endl;
            }
        }
        auto pose_estimator = get_estimator(detection.class_name);
        cv::Point2f selected_uv;
        Pose pose_obj_by_color_foot = pose_estimator->EstimateByColor(p_eye2foot, detection, color, &selected_uv);
        detection_obj.target_uv = {selected_uv.x, selected_uv.y};
        const bool valid_depth_image = !depth_float.empty() && depth_float.type() == CV_32FC1 &&
                                       depth_float.size() == color.size();
        const auto ball_estimator = std::dynamic_pointer_cast<BallPoseEstimator>(pose_estimator);
        const float ball_x_foot = pose_obj_by_color_foot.getTranslationVec()[0];
        if (detection.class_name == "Ball" && ball_estimator &&
            std::isfinite(ball_x_foot) && ball_x_foot >= ball_estimator->filter_distance() &&
            RejectProjectedBallSize(detection, p_eye2foot, pose_obj_by_color_foot,
                                    intr_, ball_estimator->radius(),
                                    ball_estimator->projected_radius_tolerance())) {
            continue;
        }
        const bool near_ball = detection.class_name == "Ball" && ball_estimator &&
                               std::isfinite(ball_x_foot) &&
                               ball_x_foot < ball_estimator->filter_distance();
        cv::Point3f center_camera;
        auto ball_fit_status = BallPoseEstimator::CenterFitStatus::Unavailable;
        if (near_ball && valid_depth_image && synced_data.depth_data.timestamp > 0.0 &&
            std::abs(timestamp - synced_data.depth_data.timestamp) <= 0.1) {
            ball_fit_status = ball_estimator->EstimateCenterInCamera(
                detection, color, depth_float, center_camera);
            if (ball_fit_status == BallPoseEstimator::CenterFitStatus::Valid) {
                AdjustEdgeBallCenter(detection, color.size(), intr_, ball_estimator->radius(),
                                     ball_estimator->projected_radius_tolerance(), center_camera);
            }
        }
        // RadiusMismatch 只说明“深度球心不可信”（贴脚时球与脚一起被框进去、bbox 长边偏大），
        // 不代表检测无效：直接丢整帧会让推球阶段成批断流，球龄超 300 ms 后触发 Kick abort
        // （40 号机实测 8/12 次 abort 由此引起）。保留检测，近距球位回退 position_projection。
        if (near_ball &&
            ball_fit_status != BallPoseEstimator::CenterFitStatus::Valid) {
            static uint64_t near_depth_fallback_count = 0;
            if (++near_depth_fallback_count % 30 == 1) {
                std::clog << "[ball] near-range depth center unavailable, fallback to color projection (n="
                          << near_depth_fallback_count << ")" << std::endl;
            }
        }
        cv::Point3f near_ball_center_base;
        bool has_near_ball_center_base = false;
        if (near_ball && ball_fit_status == BallPoseEstimator::CenterFitStatus::Valid &&
            std::isfinite(center_camera.x) && std::isfinite(center_camera.y) &&
            std::isfinite(center_camera.z)) {
            if (synced_data.pose_data.timestamp > 0.0 &&
                std::abs(timestamp - synced_data.pose_data.timestamp) <= 0.1) {
                near_ball_center_base = p_eye2base * center_camera;
                has_near_ball_center_base = std::isfinite(near_ball_center_base.x) &&
                                            std::isfinite(near_ball_center_base.y) &&
                                            std::isfinite(near_ball_center_base.z);
            }
        }
        Pose pose_obj_by_depth_foot;
        if (!near_ball && (detection.class_name != "Ball" || valid_depth_image)) {
            pose_obj_by_depth_foot = pose_estimator->EstimateByDepth(p_eye2foot, detection, color, depth_float);
        }
        Pose pose_obj_by_color_base = p_foot2base * pose_obj_by_color_foot;
        detection_obj.position_projection = pose_obj_by_color_base.getTranslationVec();

        if (detection.class_name == "Ball") {
            cv::Point3f position_base;
            cv::Point3f position_camera;
            bool has_depth_position = false;
            if (near_ball) {
                // Use the same adjusted sphere center and head_to_base transform as /ball.
                has_depth_position = has_near_ball_center_base;
                if (has_depth_position) {
                    position_base = near_ball_center_base;
                    position_camera = center_camera;
                }
            } else {
                cv::Point3f depth_position_camera;
                has_depth_position = valid_depth_image && pose_obj_by_depth_foot != Pose();
                if (has_depth_position) {
                    const auto xyz_foot = pose_obj_by_depth_foot.getTranslationVec();
                    has_depth_position = std::all_of(xyz_foot.begin(), xyz_foot.end(),
                                                     [](float value) { return std::isfinite(value); });
                    if (has_depth_position) {
                        depth_position_camera = p_foot2eye *
                            cv::Point3f(xyz_foot[0], xyz_foot[1], xyz_foot[2]);
                    }
                }
                if (!has_depth_position && valid_depth_image) {
                    has_depth_position = EstimateDepthAtPixel(depth_float, selected_uv, intr_,
                                                              depth_position_camera);
                }
                if (has_depth_position) {
                    position_base = p_eye2base * depth_position_camera;
                    position_camera = depth_position_camera;
                }
            }
            if (has_depth_position) {
                detection_obj.position = {position_base.x, position_base.y, position_base.z};
                detection_obj.position_cam = {position_camera.x, position_camera.y, position_camera.z};
                detection_obj.position_confidence = 1;
            } else {
                const float unavailable = std::numeric_limits<float>::quiet_NaN();
                detection_obj.position = {unavailable, unavailable, unavailable};
                detection_obj.position_cam = {unavailable, unavailable, unavailable};
                detection_obj.position_confidence = 0;
            }
        } else if (pose_obj_by_depth_foot == Pose()) {
            // Pose() is the invalid/no-depth sentinel for other classes.
            detection_obj.position = pose_obj_by_depth_foot.getTranslationVec();
        } else {
            const auto xyz_foot = pose_obj_by_depth_foot.getTranslationVec();
            const cv::Point3f point_camera = p_foot2eye *
                cv::Point3f(xyz_foot[0], xyz_foot[1], xyz_foot[2]);
            const cv::Point3f point_base = p_eye2base * point_camera;
            detection_obj.position = {point_base.x, point_base.y, point_base.z};
        }

        auto xyz = p_eye2base.getTranslationVec();
        auto rpy = p_eye2base.getEulerAnglesVec();
        detection_obj.received_pos = {xyz[0], xyz[1], xyz[2],
                                      static_cast<float>(rpy[0] / CV_PI * 180), static_cast<float>(rpy[1] / CV_PI * 180), static_cast<float>(rpy[2] / CV_PI * 180)};

        detection_obj.confidence = detection.confidence * 100;
        detection_obj.xmin = detection.bbox.x;
        detection_obj.ymin = detection.bbox.y;
        detection_obj.xmax = detection.bbox.x + detection.bbox.width;
        detection_obj.ymax = detection.bbox.y + detection.bbox.height;
        detection_obj.label = detection.class_name;

        // 球位口径：近距且深度球心可信 -> 球心；否则一律回退颜色地面投影。
        // 不能因为近距拟合失败就不发 /ball：RL 侧把 confidence=0 当作“本拍看不见球”。
        const bool use_near_center = near_ball && has_near_ball_center_base;
        if (detection_obj.label == "Ball" &&
            detection_obj.position_projection.size() >= 2) {
            const double bx = use_near_center ? near_ball_center_base.x : detection_obj.position_projection[0];
            const double by = use_near_center ? near_ball_center_base.y : detection_obj.position_projection[1];
            const double ball_range = std::hypot(bx, by);
            const double conf = detection_obj.confidence;
            if (std::isfinite(bx) && std::isfinite(by) &&
                bx >= kBallMinX && bx <= kBallMaxX &&
                (!has_ball_pick || conf > best_ball_conf + kBallConfMargin ||
                 (conf >= best_ball_conf - kBallConfMargin &&
                  ball_range < best_ball_range - kBallRangeHyst))) {
                has_ball_pick = true;
                best_ball_conf = conf;
                best_ball_range = ball_range;
                ball_msg.x = bx;
                ball_msg.y = by;
                ball_msg.confidence = conf;
            }
        }

        if ((color_classifier_ != nullptr) && (detection.class_name == "Opponent")) {
            // get a crop of the image given detection.bbox
            cv::Mat crop = color(detection.bbox);
            std::string robot_color_str = color_classifier_->Classify(crop);
            // add robot color to detection_obj
            detection_obj.color = robot_color_str;
        }

        // publish detection
        detection_msg.detected_objects.push_back(detection_obj);
        detections_for_display.push_back(detection);
    }

    // compute corner points positision
    std::vector<cv::Point2f> corner_uvs = {cv::Point2f(0, 0), cv::Point2f(color.cols - 1, 0),
                                           cv::Point2f(color.cols - 1, color.rows - 1), cv::Point2f(0, color.rows - 1),
                                           cv::Point2f(color.cols / 2.0, color.rows / 2.0)};
    for (auto &uv : corner_uvs) {
        auto corner_pos = CalculatePositionByIntersection(p_eye2foot, uv, intr_);
        detection_msg.corner_pos.push_back(corner_pos.x);
        detection_msg.corner_pos.push_back(corner_pos.y);
    }

    // sync-radar measurements

    // publish msg
    detection_pub_->publish(detection_msg);
    ball_pub_->publish(ball_msg);
    std::cout << std::endl;

    {
        static double last_pub_time = -1.0;
        static uint64_t count = 0;
        double pub_ts = detection_msg.header.stamp.sec +
                        static_cast<double>(detection_msg.header.stamp.nanosec) * 1e-9;
        if (last_pub_time >= 0.0) {
            double diff_ms = (pub_ts - last_pub_time) * 1000.0;
            std::cout << "[Detections Pub Interval] #" << (count) 
                      << " -> #" << (count + 1) << ": " << diff_ms << " ms" << std::endl;
        }
        last_pub_time = pub_ts;
        count++;
    }

    // Draw only when a local window or a debug image topic is requested.
    if (show_det_ || publish_det_img_) {
        const double img_ts = detection_msg.header.stamp.sec +
                              static_cast<double>(detection_msg.header.stamp.nanosec) * 1e-9;
        const double min_interval = 1.0 / static_cast<double>(publish_det_img_fps_);
        const bool publish_this_frame = publish_det_img_ && det_img_pub_ &&
            (det_img_last_ts_ < 0.0 || img_ts - det_img_last_ts_ >= min_interval);

        // Keep the local window at the input resolution. The published image
        // can use a smaller canvas while keeping drawn labels legible.
        cv::Mat window_img;
        if (show_det_) {
            window_img = YoloV8Detector::DrawDetection(color, detections_for_display);
            cv::imshow("Detection", window_img);
            cv::waitKey(1);
        }

        if (publish_this_frame) {
            cv::Mat img_out;
            if (publish_det_img_scale_ < 0.999) {
                cv::Mat scaled;
                cv::resize(color, scaled, cv::Size(), publish_det_img_scale_,
                           publish_det_img_scale_, cv::INTER_AREA);
                auto draw_dets = detections_for_display;
                for (auto &d : draw_dets) {
                    d.bbox.x = static_cast<int>(std::lround(d.bbox.x * publish_det_img_scale_));
                    d.bbox.y = static_cast<int>(std::lround(d.bbox.y * publish_det_img_scale_));
                    d.bbox.width = std::max(1, static_cast<int>(std::lround(d.bbox.width * publish_det_img_scale_)));
                    d.bbox.height = std::max(1, static_cast<int>(std::lround(d.bbox.height * publish_det_img_scale_)));
                }
                img_out = YoloV8Detector::DrawDetection(scaled, draw_dets);
            } else {
                img_out = show_det_ ? window_img :
                    YoloV8Detector::DrawDetection(color, detections_for_display);
            }
            std::vector<uchar> jpeg;
            const std::vector<int> encode_params{cv::IMWRITE_JPEG_QUALITY, publish_det_img_quality_};
            if (cv::imencode(".jpg", img_out, jpeg, encode_params)) {
                sensor_msgs::msg::CompressedImage img_msg;
                img_msg.header = detection_msg.header;
                img_msg.format = "jpeg";
                img_msg.data.assign(jpeg.begin(), jpeg.end());
                det_img_pub_->publish(img_msg);
                det_img_last_ts_ = img_ts;
            }
        }
    }

    if (save_data_) {
        save_cnt_++;
        if (save_cnt_ % save_every_n_frame_ != 0) {
            return;
        } else {
            save_cnt_ = 0;
        }
        // Keep recorded pose files in the historical camera -> foot format.
        SyncedDataBlock log_data = synced_data;
        log_data.pose_data.data = p_eye2foot;
        data_logger_->LogDataBlock(log_data);
    }
}

void VisionNode::ColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    std::cout << "new color for det received" << std::endl;
    auto start = std::chrono::system_clock::now();
    if (!msg) {
        std::cerr << "empty image message." << std::endl;
        return;
    }

    cv::Mat img;
    try {
        img = toCVMat(*msg);
    } catch (std::exception &e) {
        std::cerr << "cv_bridge exception: " << e.what() << std::endl;
        return;
    }

    if (msg->encoding == "rgb8") {
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    }

    vision_interface::msg::Detections detection_msg;
    detection_msg.header = msg->header;
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    // get synced data
    SyncedDataBlock synced_data = data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));
    
    ProcessData(synced_data, detection_msg);
    auto end = std::chrono::system_clock::now();
    std::cout << "color callback takes: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;
}

void VisionNode::CompressedColorCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    std::cout << "new compressed color for det received" << std::endl;
    auto start = std::chrono::system_clock::now();
    if (!msg) {
        std::cerr << "empty compressed image message." << std::endl;
        return;
    }

    cv::Mat img;
    try {
        // Decode compressed image
        img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Failed to decode compressed image" << std::endl;
            return;
        }
    } catch (std::exception &e) {
        std::cerr << "decode exception: " << e.what() << std::endl;
        return;
    }

    vision_interface::msg::Detections detection_msg;
    detection_msg.header = msg->header;
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    // get synced data
    SyncedDataBlock synced_data = data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));
    
    ProcessData(synced_data, detection_msg);
    auto end = std::chrono::system_clock::now();
    std::cout << "compressed color callback takes: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;
}

void VisionNode::ProcessSegmentationData(SyncedDataBlock &synced_data, vision_interface::msg::LineSegments &field_line_segs_msg) {
    double timestamp = synced_data.color_data.timestamp;
    cv::Mat color = synced_data.color_data.data;
    // Field-line fitting intersects rays with the foot-frame ground plane.
    Pose p_base2foot(0.0F, 0.0F, base_height_above_foot_, 0.0F, 0.0F, 0.0F);
    Pose p_eye2foot = p_base2foot * synced_data.pose_data.data;

    double time_diff = (timestamp - synced_data.pose_data.timestamp) * 1000;
    if (time_diff > 40) {
        std::cerr << "seg: color pose time diff: " << time_diff << " ms" << std::endl;
    }
    std::cout << "seg: p_eye2foot: \n"
              << p_eye2foot.toCVMat() << std::endl;

    // inference
    auto segmentations = segmentor_->Inference(color);
    std::vector<SegmentationRes> field_line_segmentations;
    std::vector<FieldLineSegment> field_line_segs;
    for (auto &seg : segmentations) {
        // 单类模型（seg_261112）配 -1 不过滤；多类模型由 field_line_class_id 指定类别。
        if (field_line_class_id_ >= 0 && seg.class_id != field_line_class_id_) {
            continue;
        }
        field_line_segmentations.push_back(seg);
        auto line_segs = FitFieldLineSegments(p_eye2foot, intr_, seg.contour, line_segment_area_threshold_);
        for (auto line_seg : line_segs) {
            float inlier_precentage = static_cast<float>(line_seg.inlier_count) / line_seg.contour_2d_points.size();
            if (inlier_precentage < 0.25) {
                continue;
            }
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[0].x);
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[0].y);
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[1].x);
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[1].y);

            field_line_segs_msg.coordinates_uv.push_back(line_seg.end_points_2d[0].x);
            field_line_segs_msg.coordinates_uv.push_back(line_seg.end_points_2d[0].y);
            field_line_segs_msg.coordinates_uv.push_back(line_seg.end_points_2d[1].x);
            field_line_segs_msg.coordinates_uv.push_back(line_seg.end_points_2d[1].y);

            field_line_segs.push_back(line_seg);
        }
    }
    std::cout << segmentations.size() << " objects segmented." << std::endl;

    field_line_pub_->publish(field_line_segs_msg);
    if (show_seg_) {
        cv::Mat img_out = YoloV8Segmentor::DrawSegmentation(color, field_line_segmentations);
        img_out = DrawFieldLineSegments(img_out, field_line_segs);
        cv::imshow("Segmentation", img_out);
        cv::waitKey(1);
    }
}

void VisionNode::SegmentationCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (!segmentor_) {
        std::cerr << "no segmentor loaded." << std::endl;
        return;
    }
    std::cout << "new color for seg received" << std::endl;
    if (!msg) {
        std::cerr << "empty image message." << std::endl;
        return;
    }

    // cv_bridge::CvImagePtr cv_ptr; // make use of cv_bridge to convert ROS image message to OpenCV cv::Mat format
    cv::Mat img;
    try {
        // cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
        img = toCVMat(*msg).clone();
    } catch (std::exception &e) {
        std::cerr << "cv_bridge exception: " << e.what() << std::endl;
        return;
    }

    if (msg->encoding == "rgb8") {
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    }

    vision_interface::msg::LineSegments field_line_segs_msg;
    field_line_segs_msg.header = msg->header;
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    // get synced data
    SyncedDataBlock synced_data = seg_data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));
    ProcessSegmentationData(synced_data, field_line_segs_msg);
}

void VisionNode::CompressedSegmentationCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    if (!segmentor_) {
        std::cerr << "no segmentor loaded." << std::endl;
        return;
    }
    std::cout << "new compressed color for seg received" << std::endl;
    if (!msg) {
        std::cerr << "empty compressed image message." << std::endl;
        return;
    }

    cv::Mat img;
    try {
        img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Failed to decode compressed image" << std::endl;
            return;
        }
    } catch (std::exception &e) {
        std::cerr << "decode exception: " << e.what() << std::endl;
        return;
    }

    vision_interface::msg::LineSegments field_line_segs_msg;
    field_line_segs_msg.header = msg->header;
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    // get synced data
    SyncedDataBlock synced_data = seg_data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));
    ProcessSegmentationData(synced_data, field_line_segs_msg);
}

void VisionNode::DepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    std::cout << "new depth received" << std::endl;
    // cv_bridge::CvImagePtr cv_ptr;
    cv::Mat img;
    try {
        // TODO(SS): check if the image is 16-bit for zed camera
        // cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        img = toCVMat(*msg).clone();
    } catch (std::exception &e) {
        std::cerr << "cv_bridge exception " << e.what() << std::endl;
        return;
    }

    if (img.empty()) {
        std::cerr << "empty image recevied." << std::endl;
        return;
    }

    // Check if the image is indeed 16-bit
    if (img.depth() != CV_16U && img.depth() != CV_32F) {
        std::cerr << "image is either 16u or 32f." << std::endl;
        return;
    }

    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    data_syncer_->AddDepth(DepthDataBlock(img, timestamp));
    // seg_data_syncer_->AddDepth(DepthDataBlock(img, timestamp));
}

void VisionNode::CompressedDepthCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    std::cout << "new compressed depth received" << std::endl;
    cv::Mat img;
    try {
        cv::Mat compressed_data = cv::Mat(msg->data);
        img = cv::imdecode(compressed_data, cv::IMREAD_ANYDEPTH);
        
        if (img.empty()) {
            std::cerr << "Failed to decode compressed depth image" << std::endl;
            return;
        }
    } catch (std::exception &e) {
        std::cerr << "decode exception " << e.what() << std::endl;
        return;
    }

    if (img.empty()) {
        std::cerr << "empty image received." << std::endl;
        return;
    }

    // Check if the image is indeed 16-bit or 32-bit float
    if (img.depth() != CV_16U && img.depth() != CV_32F) {
        std::cerr << "image must be either 16u or 32f." << std::endl;
        return;
    }

    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    data_syncer_->AddDepth(DepthDataBlock(img, timestamp));
    // seg_data_syncer_->AddDepth(DepthDataBlock(img, timestamp));
}

void VisionNode::PoseTFCallBack(const geometry_msgs::msg::TransformStamped::SharedPtr msg) {
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    Pose p_foot2base(0.0F, 0.0F, -base_height_above_foot_, 0.0F, 0.0F, 0.0F);
    Pose p_eye2base = p_foot2base * Pose(*msg);
    data_syncer_->AddPose(PoseDataBlock(p_eye2base, timestamp));
    seg_data_syncer_->AddPose(PoseDataBlock(p_eye2base, timestamp));
}

void VisionNode::PoseCallBack(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    if (!msg) return;
    // 时间戳：仍用本机 now()，暂不用 msg->header.stamp。zl 侧 stamp 是板端时钟，
    // 板子与 NX 各自 NTP 同步但不同源（2026-11-11 实测偏差约 40~200 ms 量级），
    // 而图像时间戳是本机 now()，直接改用板端 stamp 会让位姿/图像同步整体平移。
    // 时钟同源（同一主机或启用 chrony/PTP 对齐）后再切成 header.stamp。
    auto current_time = this->get_clock()->now();
    double timestamp = static_cast<double>(current_time.nanoseconds()) * 1e-9;

    float x = msg->pose.position.x;
    float y = msg->pose.position.y;
    float z = msg->pose.position.z;
    float qx = msg->pose.orientation.x;
    float qy = msg->pose.orientation.y;
    float qz = msg->pose.orientation.z;
    float qw = msg->pose.orientation.w;
    Pose p_eye2base(x, y, z, qx, qy, qz, qw);
    data_syncer_->AddPose(PoseDataBlock(p_eye2base, timestamp));
    seg_data_syncer_->AddPose(PoseDataBlock(p_eye2base, timestamp));

    if (!offline_mode_) {
        Pose p_base2foot(0.0F, 0.0F, base_height_above_foot_, 0.0F, 0.0F, 0.0F);
        Pose p_eye2foot = p_base2foot * p_eye2base;
        auto tf_msg = p_eye2foot.toRosTFMsg();
        tf_msg.header.stamp = builtin_interfaces::msg::Time(current_time);
        tf_msg.header.frame_id = "foot";
        tf_msg.child_frame_id = "camera_optical_frame";

        pose_tf_pub_->publish(tf_msg);
    }
}

void VisionNode::CameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    if (!msg) {
        std::cerr << "empty camera info message." << std::endl;
        return;
    }

    // Track if this is the first update
    static bool first_update = true;
    
    float fx = msg->k[0];
    float fy = msg->k[4];
    float cx = msg->k[2];
    float cy = msg->k[5];

    std::vector<float> distortion_coeffs(msg->d.begin(), msg->d.end());

    // Determine distortion model based on camera type
    Intrinsics::DistortionModel distortion_model;

    if (msg->d.empty())
    {
        distortion_model = Intrinsics::DistortionModel::kNone;
    } else {
        // Heuristic: if distortion coefficients are all zero, treat as no distortion
        float distortion_sum = 0.0;
        for (size_t i = 0; i < msg->d.size(); ++i) {
            distortion_sum += std::abs(msg->d[i]);
        }
        if (distortion_sum < 1e-6) {
            distortion_model = Intrinsics::DistortionModel::kNone;
        } else {
            if (msg->distortion_model == "plumb_bob") {
                distortion_model = Intrinsics::DistortionModel::kInverseBrownConrady;
            } else  {
                distortion_model = Intrinsics::DistortionModel::kBrownConrady;
            }
        }
    }
    // Check if distortion coefficients are effectively zero
    float distortion_sum = 0.0;
    for (auto coeff : distortion_coeffs) {
        distortion_sum += std::abs(coeff);
    }
    
    Intrinsics new_intr;
    if (distortion_sum < 1e-6 || distortion_coeffs.empty()) {
        new_intr = Intrinsics(fx, fy, cx, cy);
        std::cout << "Camera has no distortion" << std::endl;
    } else {
        new_intr = Intrinsics(fx, fy, cx, cy, distortion_coeffs, distortion_model);
        std::cout << "Camera has distortion model: " << static_cast<int>(distortion_model) << std::endl;
    }
    
    // Check if values have changed significantly (or if this is the first update)
    bool should_update = first_update || 
                        std::abs(intr_.fx - fx) > 0.1 || 
                        std::abs(intr_.fy - fy) > 0.1 || 
                        std::abs(intr_.cx - cx) > 0.1 || 
                        std::abs(intr_.cy - cy) > 0.1;
    
    if (should_update) {
        intr_ = new_intr;
        std::cout << "Camera intrinsics updated from topic:" << std::endl;
        std::cout << "  fx=" << fx << ", fy=" << fy << std::endl;
        std::cout << "  cx=" << cx << ", cy=" << cy << std::endl;
        std::cout << "  width=" << msg->width << ", height=" << msg->height << std::endl;
        std::cout << "  distortion_model=" << msg->distortion_model << std::endl;
        if (!distortion_coeffs.empty()) {
            std::cout << "  distortion_coeffs: [";
            for (size_t i = 0; i < distortion_coeffs.size(); ++i) {
                std::cout << distortion_coeffs[i];
                if (i < distortion_coeffs.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Recreate all pose estimators with new intrinsics
        pose_estimator_map_.clear();
        
        pose_estimator_ = std::make_shared<PoseEstimator>(intr_);
        pose_estimator_->Init(YAML::Node());
        pose_estimator_map_["default"] = pose_estimator_;
        std::cout << "Created default pose estimator with updated intrinsics" << std::endl;
        
        if (config_node_["ball_pose_estimator"]) {
            pose_estimator_map_["ball"] = std::make_shared<BallPoseEstimator>(intr_);
            pose_estimator_map_["ball"]->Init(config_node_["ball_pose_estimator"]);
            std::cout << "Created ball pose estimator with updated intrinsics" << std::endl;
        }
        
        if (config_node_["human_like_pose_estimator"]) {
            pose_estimator_map_["human_like"] = std::make_shared<HumanLikePoseEstimator>(intr_);
            pose_estimator_map_["human_like"]->Init(config_node_["human_like_pose_estimator"]);
            std::cout << "Created human_like pose estimator with updated intrinsics" << std::endl;
        }
        
        if (config_node_["field_marker_pose_estimator"]) {
            pose_estimator_map_["field_marker"] = std::make_shared<FieldMarkerPoseEstimator>(intr_);
            pose_estimator_map_["field_marker"]->Init(config_node_["field_marker_pose_estimator"]);
            std::cout << "Created field_marker pose estimator with updated intrinsics" << std::endl;
        }
        
        first_update = false;
    }
    
    // After receiving several stable updates, unsubscribe to save resources
    static int received_count = 0;
    received_count++;
    if (received_count >= 5) {
        camera_info_sub_.reset();
        std::cout << "Camera info callback disabled after " << received_count << " messages" << std::endl;
        std::cout << "Final camera intrinsics:" << std::endl << intr_ << std::endl;
    }
}

} // namespace robot_vision

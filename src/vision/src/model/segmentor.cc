#include "robot_vision/model/segmentor.h"

#ifdef NO_CUDA
#include "robot_vision/model/onnx/segmentation_impl.h"
#else
#include "robot_vision/model/trt/impl.h"
#endif

namespace robot_vision {

const std::vector<std::string> YoloV8Segmentor::kClassLabels = {"CircleLine", "Line"};

std::shared_ptr<YoloV8Segmentor> YoloV8Segmentor::CreateYoloV8Segmentor(const YAML::Node &node, const std::string &model_path_override) {
    try {
        std::string model_path = model_path_override.empty() ? node["model_path"].as<std::string>() : model_path_override;
        float conf_thresh = node["confidence_threshold"].as<float>();

        #ifdef NO_CUDA
        // Use ONNX when CUDA is not available
        return std::shared_ptr<YoloV8Segmentor>(new YoloV8SegmentorONNX(model_path, conf_thresh));
        #else
        // Use TensorRT when CUDA is available
        return std::shared_ptr<YoloV8Segmentor>(new YoloV8SegmentorTRT(model_path, conf_thresh));
        #endif
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return nullptr;
    }
}

cv::Mat YoloV8Segmentor::DrawSegmentation(const cv::Mat &img, const std::vector<SegmentationRes> &segmentations) {
    static std::unordered_map<int, cv::Scalar> class_colors;
    auto get_color_for_class = [&](int class_id) {
        if (class_colors.find(class_id) == class_colors.end()) {
            // Generate a random color for the class
            cv::RNG rng(class_id);
            class_colors[class_id] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 256));
        }
        return class_colors[class_id];
    };

    cv::Mat img_out;
    cv::cvtColor(img, img_out, cv::COLOR_RGB2BGR);
    for (const auto &seg : segmentations) {
        auto display_color = get_color_for_class(seg.class_id);
        for (int y = seg.bbox.y; y < seg.bbox.y + seg.bbox.height; y++) {
            for (int x = seg.bbox.x; x < seg.bbox.x + seg.bbox.width; x++) {
                if (seg.mask.at<uchar>(y, x) > 127) {
                    for (int c = 0; c < 3; c++) {
                        // img_out.at<cv::Vec3b>(y, x)[c] = std::max(255, img_out.at<cv::Vec3b>(y, x)[c] + static_cast<int>(display_color[c]));
                        img_out.at<cv::Vec3b>(y, x)[c] = std::min(255, static_cast<int>(display_color[c]));
                    }
                }
            }
        }
        cv::rectangle(img_out, seg.bbox, display_color, 2);
        const std::string class_label =
            seg.class_id >= 0 && static_cast<size_t>(seg.class_id) < kClassLabels.size()
                ? kClassLabels[seg.class_id]
                : "class_" + std::to_string(seg.class_id);
        cv::putText(img_out, class_label + " " + std::to_string(int(seg.confidence * 100)), cv::Point(seg.bbox.x, seg.bbox.y + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
    }
    return img_out;
}

} // namespace robot_vision
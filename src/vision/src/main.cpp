#include "robot_vision/vision_node.h"

#include <iostream>
#include <streambuf>

namespace {

// Silence the verbose stdout/stderr output produced by the vision pipeline and
// its inference backends. The one intentionally retained runtime message (the
// detected ball position in camera_optical_frame) is written to std::clog.
class NullStreamBuffer : public std::streambuf {
protected:
    int_type overflow(int_type character) override {
        return traits_type::not_eof(character);
    }
};

} // namespace

int main(int argc, char **argv) {
    static NullStreamBuffer null_stream_buffer;
    std::cout.rdbuf(&null_stream_buffer);
    std::cerr.rdbuf(&null_stream_buffer);

    if (argc < 2) {
        std::cerr << "Usage: vision_node <config_template_file> [config_override_file ...]" << std::endl;
        return -1;
    }
    rclcpp::init(argc, argv);

    std::string node_name = "vision_node";
    
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    
    auto node = std::make_shared<robot_vision::VisionNode>(node_name, options);

    std::string config_template_path = argv[1];
    std::vector<std::string> config_paths;
    for (int i = 2; i < argc; ++i) {
        config_paths.emplace_back(argv[i]);
    }
    node->Init(config_template_path, config_paths);

    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();

    return 0;
}

// ball_fwd_node：独立运行球类型转换（vision_interface/Ball -> robot_interfaces/Ball）。
//
// 用途：nx_ctl.sh 的 ball 调试模式（只发球位置、不起 brain，因此没有进程内的 BallFwd）。
// 正常比赛由 brain_node 内的 BallFwd 完成同样的事，不需要本可执行文件。

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "robot_io/ball_fwd.h"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("ball_fwd_node");
  robot_io::BallFwd fwd(node.get());
  fwd.init();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

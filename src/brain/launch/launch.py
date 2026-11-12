# coding: utf8

import launch
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def handle_configuration(context, *args, **kwargs):
    # get launch argument 'vision_config_path' and construct paths to vision.yaml and vision_local.yaml
    vision_config_dir = context.perform_substitution(LaunchConfiguration('vision_config_path'))
    vision_config_file = os.path.join(vision_config_dir, 'vision.yaml')
    vision_config_local_file = os.path.join(vision_config_dir, 'vision_local.yaml')

    config_path = os.path.join(os.path.dirname(__file__), '../config')
    config_file = os.path.join(config_path, 'config.yaml')
    config_local_file = os.path.join(config_path, 'config_local.yaml')
    config_home_file = os.path.join(os.path.expanduser('~'), 'agents/soccer/brain.yaml')

    # 参数文件层：按优先级从低到高排列，存在才加载。
    # config_local.yaml / ~/agents/soccer/brain.yaml 都是可选层，缺失时直接跳过——
    # 旧实现无条件把路径塞进 --params-file，缺文件时该层既不能覆盖也没提示，
    # 且 ~/agents/... 这一层在多数机器上并不存在，属于幽灵层。
    param_files = [config_file]
    for layer in (config_local_file, config_home_file):
        if os.path.exists(layer):
            param_files.append(layer)
        else:
            print(f'[brain launch] 跳过不存在的配置层: {layer}')

    # robocup_sim：可选仿真画像（GT 开关、multi_instance、kick_targets/ball_fwd 关闭等）。
    # 默认空 = 行为与历史完全一致；实机禁止与 real_robot:=true 同时使用。
    sim_profile = context.perform_substitution(LaunchConfiguration('sim_profile')).strip()
    real_robot_mode = context.perform_substitution(
        LaunchConfiguration('real_robot')).strip().lower() in ('true', '1')
    if real_robot_mode and sim_profile:
        raise RuntimeError('real_robot:=true 不能与 sim_profile 同时使用（仿真画像不得带入实机）')
    if sim_profile:
        if not os.path.isfile(sim_profile):
            raise RuntimeError(f'sim_profile 不存在: {sim_profile}')
        param_files.append(sim_profile)

    print('[brain launch] 参数文件层: ' + ' -> '.join(param_files) + ' -> <launch 内联覆盖>')


    behavior_trees_dir = os.path.join(os.path.dirname(__file__), '../behavior_trees')
    def make_tree_path(name):
        if not name.endswith('.xml'):
            name += '.xml'
        return os.path.join(behavior_trees_dir, name)
    tree = context.perform_substitution(LaunchConfiguration('tree'))
    tree_path = make_tree_path(tree)

    # if there are parameters that are commonly changed for different matches, you can add them here to make it more convenient to modify when launching. For example, you can specify player role, team id, player id, etc. when launching, instead of modifying config.yaml every time before launching.
    config = {
            # The behavior tree file to load, should be placed in src/brain/config/behavior_trees
            "tree_file_path": tree_path,
            "vision_config_path": vision_config_file,
            "vision_config_local_path": vision_config_local_file,
    }

    role = context.perform_substitution(LaunchConfiguration('role'))
    if not role == '':
        config['game.player_role'] = role
    
    team_id = context.perform_substitution(LaunchConfiguration('team_id'))
    if not team_id == '':
        config['game.team_id'] = int(team_id)

    player_id = context.perform_substitution(LaunchConfiguration('player_id'))
    if not player_id == '':
        config['game.player_id'] = int(player_id)

    robot_name = context.perform_substitution(LaunchConfiguration('robot_name'))
    if not robot_name == '':
        config['robot.robot_name'] = robot_name
        # 多机（仿真）每台一套 zl 命名空间：/{name}/* + /{name}/cmd_vel。
        # 实机一机一图，robot_name 为空，保持默认的 /robot/* + /cmd_vel。
        config['zl_io.topic_ns'] = '/' + robot_name
        config['zl_io.cmd_vel_topic'] = '/' + robot_name + '/cmd_vel'

    odom_source = context.perform_substitution(LaunchConfiguration('odom_source')).strip()
    if odom_source not in ('legacy', 'autoodom'):
        raise ValueError('odom_source 只接受 legacy 或 autoodom')
    if odom_source == 'autoodom':
        if robot_name:
            raise ValueError('当前 AutoOdom 旁路入口仅支持单机，不可复用到多机命名空间')
        config['zl_io.odom_topic'] = '/autoodom/selected'
        config['zl_io.odom_convention'] = 'consistent'
        config['zl_io.consistent_odom_scale'] = 1.0

    agent_mode = context.perform_substitution(LaunchConfiguration('agent_mode'))
    if agent_mode in ['true', 'True', '1']:
        config['game.agent_mode'] = True

    sim = context.perform_substitution(LaunchConfiguration('sim'))
    if sim in ['true', 'True', '1']:
        # MuJoCo 不发布 /clock，use_sim_time=True 会导致 get_clock()->now() 冻结在 0，
        # RobotClient 的 intent 节流会丢弃所有运动指令，因此仿真模式也必须关闭
        config['use_sim_time'] = False


    disableCom = context.perform_substitution(LaunchConfiguration('disable_com'))
    if disableCom in ['true', 'True', '1']:
        config['enable_com'] = False

    # 实机每队上场人数：留空沿用旧行为 1（单台）；多台同队必须传实际人数（如 3），
    # 否则 brain 的 liveCount / 角色重分配会按单人计算。仅 real_robot:=true 时生效。
    num_players_raw = context.perform_substitution(LaunchConfiguration('num_players')).strip()
    try:
        num_players = int(num_players_raw) if num_players_raw else 1
    except ValueError:
        num_players = 1
    if not 1 <= num_players <= 11:
        num_players = 1

    if real_robot_mode:
        # 单台实机不使用仿真真值。自动视觉踢球改为 launch 参数控制：
        # 门感知（/robot/perception/target_destination）就绪后可传 true 实测，
        # 默认仍为 false，避免在无门/无球感知时误切踢球模型。
        auto_visual_kick = context.perform_substitution(
            LaunchConfiguration('enable_auto_visual_kick')).strip().lower() in ('true', '1')
        config.update({
            'robot.robot_name': '',
            # 实机一机一图，固定使用 zl 原生契约名；即使误传 robot_name 也不留多机命名空间。
            'zl_io.topic_ns': '/robot',
            'zl_io.cmd_vel_topic': '/cmd_vel',
            'game.number_of_players': num_players,
            'ground_truth.enable_pose': False,
            'ground_truth.enable_ball': False,
            'RLVisionKick.enableAutoVisualKick': auto_visual_kick,
            'use_sim_time': False,
        })
        # 真实球门位姿合成（原 goal_pose_synth 节点，2026-11-03 已并入 brain 的 zl_io）。
        # launch 参数只决定开机初值；运行期仍可 ros2 param set /brain_node zl_io.kick_targets_enable false。
        goal_synth = context.perform_substitution(
            LaunchConfiguration('goal_synth')).strip().lower() in ('true', '1')
        config['zl_io.kick_targets_enable'] = goal_synth
        # 可选的触发距离上限覆盖（实机实测时放宽/收窄触发窗；留空沿用 config_local）。
        dist_max_raw = context.perform_substitution(
            LaunchConfiguration('auto_visual_kick_dist_max')).strip()
        if dist_max_raw:
            try:
                config['RLVisionKick.autoVisualKickEnableDistMax'] = float(dist_max_raw)
            except ValueError:
                pass

        # 避障（Chase/Assist 基于深度栅格的绕行）：留空沿用 config.yaml，只有显式
        # true/false 才覆盖。只覆盖 avoid_during_chase（S2 行为），不碰
        # obstacle_avoidance.enable（后者会改变 GoToFreekick 的运动路径）。
        avoidance_raw = context.perform_substitution(
            LaunchConfiguration('enable_avoidance')).strip().lower()
        if avoidance_raw in ('true', '1', 'false', '0'):
            config['obstacle_avoidance.avoid_during_chase'] = avoidance_raw in ('true', '1')

    return [
        Node(
            package ='brain',
            executable='brain_node',
            output='screen',
            parameters=[
                *param_files,
                config
            ]
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('odom_source', default_value='legacy',
                              description='里程计入口：legacy 保留原本体；autoodom 订阅旁路选择器的自洽坐标输出'),
        DeclareLaunchArgument(
            'real_robot', default_value='false',
            description='单台实机安全配置：无话题后缀、关闭真值注入'
        ),
        DeclareLaunchArgument(
            'enable_auto_visual_kick', default_value='false',
            description='实机是否允许自动视觉踢球（RLVisionKick.enableAutoVisualKick）；'
                        '门感知 target_destination 有数据后再打开'
        ),
        DeclareLaunchArgument(
            'enable_avoidance', default_value='',
            description='实机避障开关（obstacle_avoidance.avoid_during_chase，Chase/Assist 绕行）；'
                        'true/false 显式覆盖，留空沿用 config.yaml'
        ),
        DeclareLaunchArgument(
            'auto_visual_kick_dist_max', default_value='',
            description='实机视觉踢触发距离上限（米）；留空沿用 config_local.yaml'
        ),
        DeclareLaunchArgument(
            'num_players', default_value='',
            description='实机每队上场人数（1~11；留空=1）。多台同队时必须传实际人数（如 3），'
                        '否则 liveCount/角色重分配按单人计算。仅 real_robot:=true 时生效。'
        ),
        DeclareLaunchArgument(
            'goal_synth', default_value='true',
            description='实机是否开机即发布真实球门位姿（zl_io.kick_targets_enable）；'
                        'false 时 zl 侧保持最后一帧/假门'
        ),
        DeclareLaunchArgument(
            'vision_config_path',
            default_value=os.path.join(os.path.dirname(__file__), '../../../../vision/share/vision/config'),
            description='Directory containing vision.yaml and vision_local.yaml'
        ),
        # Parameters that can be provided through ros2 launch brain launch.py param:=value need to be declared here using DeclareLaunchArgument, and then handled in handle_configuration
        DeclareLaunchArgument(
            'tree', 
            default_value='game.xml',
            description='Specify behavior tree file name. DO NOT include full path, file should be in src/brain/config/behavior_trees'
        ),
        DeclareLaunchArgument(
            'role', 
            default_value='',
            description='If you need to override game.player_role in config.yaml, you can specify the parameter role:=striker when launching'
        ),
        DeclareLaunchArgument(
            'team_id',
            default_value='',
            description='Set the team ID, for example team_id:=1'
        ),
        DeclareLaunchArgument(
            'player_id',
            default_value='',
            description='Set the player ID, for example player_id:=2'
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value='',
            description='Set the robot name, for example robot_name:=robot0'
        ),
        DeclareLaunchArgument(
            'sim',
            default_value='false',
            description='Whether to run in simulation'

        ),
        DeclareLaunchArgument(
            'sim_profile',
            default_value='',
            description='可选仿真画像 YAML 路径（robocup_sim 使用）；默认空，行为不变；'
                        '不得与 real_robot:=true 同时使用'
        ),
        DeclareLaunchArgument(
            'disable_com', 
            default_value='false',
            description='Force disable communication'
        ),
        DeclareLaunchArgument(
            'agent_mode', 
            default_value='false',
            description='Whether to run in agent mode. In agent mode, the robot is controlled by the APP and does not receive commands from the referee or physical controller'
        ),
        OpaqueFunction(function=handle_configuration) # Continue processing in handle_configuration
    ])

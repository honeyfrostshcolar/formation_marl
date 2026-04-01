import os
import time
import argparse
import numpy as np
import torch

from envs.formation_2d_env import Formation2DMultiAgentEnv
from agents.maddpg_agent import MADDPG_Agent


# ==========================================
# 0. 参数解析（必须与训练时网络结构保持一致）
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="MADDPG + MAGIC Scheduler + CoDe Evaluation"
    )

    # =========================================================
    # [新增] 评估专属参数
    # =========================================================
    parser.add_argument("--model_dir", type=str, required=True, help="要评估的模型文件夹路径 (包含 maddpg_checkpoint.pt 的目录)。")
    parser.add_argument("--eval_episodes", type=int, default=10, help="评估多少局。")
    parser.add_argument("--map_mode", type=str, default="custom", choices=["open", "z_map", "custom"], help="你想在哪个地图上评估模型？")
    parser.add_argument("--custom_map_path", type=str, default="/home/lpp/formation_test/maps/underground_garage5.pgm", help="真实地图路径。")


    # =========================================================
    # 一、环境参数
    # =========================================================
    parser.add_argument("--num_robots", type=int, default=3, help="总机器人数量，包含 1 个 leader 和 N-1 个 follower。")
    parser.add_argument("--max_steps", type=int, default=1000, help="每个评估 episode 的最大步数。")
    parser.add_argument("--sensing_radius", type=float, default=5.0, help="每个 follower 的局部感知半径。")

    # =========================================================
    # 二、MADDPG 基础参数（评估时主要为了初始化 Agent）
    # =========================================================
    parser.add_argument("--lr_actor", type=float, default=5e-5, help="Actor 学习率，占位参数，评估时不更新。")
    parser.add_argument("--lr_critic", type=float, default=3e-4, help="Critic 学习率，占位参数，评估时不更新。")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子，占位参数。")
    parser.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数，占位参数。")

    # =========================================================
    # 三、MAGIC 调度器参数
    # =========================================================
    parser.add_argument("--hid_size", type=int, default=64, help="LSTM 隐状态维度。")
    parser.add_argument("--gat_hid_size", type=int, default=64, help="GAT 中间特征维度。")
    parser.add_argument("--gat_num_heads", type=int, default=4, help="第一轮 GAT 多头数。")
    parser.add_argument("--gat_num_heads_out", type=int, default=1, help="第二轮 GAT 多头数。")
    parser.add_argument("--self_loop_type1", type=int, default=2, help="第一层自环控制方式。")
    parser.add_argument("--self_loop_type2", type=int, default=2, help="第二层自环控制方式。")
    parser.add_argument("--first_gat_normalize", action="store_true", default=False, help="是否对第一层 GAT 注意力归一化。")
    parser.add_argument("--second_gat_normalize", action="store_true", default=False, help="是否对第二层 GAT 注意力归一化。")

    parser.add_argument("--use_gat_encoder", action="store_true", default=False, help="是否启用 scheduler 前的 GAT encoder。")
    parser.add_argument("--first_graph_complete", action="store_true", default=False, help="第一轮通信图是否强制全连接。")

    parser.add_argument("--learn_second_graph", dest="learn_second_graph", action="store_true", help="是否启用第二轮调度器。")
    parser.add_argument("--no_learn_second_graph", dest="learn_second_graph", action="store_false", help="是否关闭第二轮调度器。")
    parser.set_defaults(learn_second_graph=True)

    parser.add_argument("--second_graph_complete", action="store_true", default=False, help="第二轮通信图是否强制全连接。")

    parser.add_argument("--directed", dest="directed", action="store_true", help="使用有向通信图。")
    parser.add_argument("--undirected", dest="directed", action="store_false", help="使用无向通信图。")
    parser.set_defaults(directed=True)

    parser.add_argument("--comm_mask_zero", action="store_true", default=False, help="是否强制关闭通信。")

    parser.add_argument("--gat_encoder_out_size", type=int, default=64, help="scheduler 专属 GAT encoder 输出维度。")
    parser.add_argument("--ge_num_heads", type=int, default=4, help="scheduler 专属 GAT encoder 多头数。")
    parser.add_argument("--gat_encoder_normalize", action="store_true", default=False, help="scheduler 专属 GAT encoder 是否归一化。")

    parser.add_argument("--message_encoder", action="store_true", default=False, help="发送端是否对 hidden state 再做一层消息映射。")
    parser.add_argument("--message_decoder", action="store_true", default=False, help="接收端是否对融合消息再做一层映射。")
    parser.add_argument("--comm_init", type=str, default="zeros", help="通信模块初始化方式。")

    # =========================================================
    # 四、CoDe 参数
    # =========================================================
    parser.add_argument("--intent_dim", type=int, default=32, help="意图向量维度。")
    parser.add_argument("--decoder_hidden_dim", type=int, default=64, help="Intent Decoder 的 GRU hidden 维度。")
    parser.add_argument("--value_dim", type=int, default=64, help="接收端融合消息向量维度。")
    parser.add_argument("--attn_dim", type=int, default=32, help="Q/K 投影维度。")
    parser.add_argument("--gamma_t", type=float, default=0.90, help="消息时间衰减系数。")
    parser.add_argument("--lambda_inf", type=float, default=1.0, help="L_inf 权重，占位参数。")
    parser.add_argument("--lambda_c", type=float, default=0.1, help="L_c 权重，占位参数。")
    parser.add_argument("--lambda_k", type=float, default=1e-3, help="L_k 权重，占位参数。")
    parser.add_argument("--lambda_e", type=float, default=1e-3, help="L_e 权重，占位参数。")
    parser.add_argument("--eps", type=float, default=1e-8, help="数值稳定项。")
    parser.add_argument("--renorm_after_decay", action="store_true", default=False, help="时间衰减后是否重新归一化注意力。")
    parser.add_argument("--pred_horizon", type=int, default=4, help="发送端 decoder 预测未来动作的步数 K。")

    # =========================================================
    # 五、延迟设置
    # =========================================================
    parser.add_argument("--delay_mode", type=str, default="fixed", choices=["none", "fixed", "uniform"], help="通信延迟模式。")
    parser.add_argument("--fixed_delay", type=int, default=2, help="固定延迟模式下的延迟步数。")
    parser.add_argument("--min_delay", type=int, default=1, help="随机延迟模式下的最小延迟。")
    parser.add_argument("--max_delay", type=int, default=3, help="随机延迟模式下的最大延迟。")

    args = parser.parse_args()

    # =========================================================
    # 六、派生参数
    # =========================================================
    args.num_followers = args.num_robots - 1
    args.nagents = args.num_followers
    args.action_dim = 2
    args.obs_size = 35 # ✅ 必须改为 35，适配最新加入的 formation_alpha 指令！

    return args


# ==========================================
# 1. 只加载模型权重
# ==========================================
def load_evaluate_model(agent, path):
    checkpoint_file = os.path.join(path, "maddpg_checkpoint.pt")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"找不到模型文件，请检查路径: {checkpoint_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_file, map_location=device)

    # 评估时主要使用 actor
    agent.actor.load_state_dict(checkpoint["actor"])

    # 同步 target_actor，避免内部状态不一致
    if "target_actor" in checkpoint:
        agent.target_actor.load_state_dict(checkpoint["target_actor"])
    else:
        agent.target_actor.load_state_dict(checkpoint["actor"])

    print(f"✅ 成功加载第 {checkpoint['episode']} 轮的模型。")


# ==========================================
# 2. 主评估流程
# ==========================================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估启动，运行设备: {device}")
    print(f"即将测试的地图模式: {args.map_mode}")

    # ✅ 把地图模式塞进 config 给环境初始化用
    config = {
        "num_robots": args.num_robots,
        "max_steps": args.max_steps,
        "render": True,
        "sensing_radius": args.sensing_radius,
        "map_mode": args.map_mode,
        "custom_map_path": args.custom_map_path,
    }

    env = Formation2DMultiAgentEnv(config)
    agent = MADDPG_Agent(args)
    load_evaluate_model(agent, args.model_dir)

    for episode in range(args.eval_episodes):
        obs_dict, _ = env.reset()
        obs_array = np.array([obs_dict[agent_id] for agent_id in env._agent_ids], dtype=np.float32)

        # 每个 episode 初始化 LSTM 和记忆化延迟缓冲
        h_in, c_in = agent.init_hidden()
        agent.reset_runtime()

        episode_reward = 0.0

        for step in range(args.max_steps):
            # 新版接口返回 7 个量
            action_policy, action_exec, graphs_array, graphs_soft_array, h_out, c_out, comm_snapshot = agent.select_action(
                obs_array,
                h_in,
                c_in,
                add_noise=False, # 评估时严格无噪
            )

            # 评估时执行无噪动作；这里 action_exec == action_policy，因为 add_noise=False
            action_dict = {env._agent_ids[i]: action_exec[i] for i in range(args.num_followers)}
            graph_dict = {env._agent_ids[i]: graphs_array[i] for i in range(args.num_followers)}
            graph_soft_dict = {env._agent_ids[i]: graphs_soft_array[i] for i in range(args.num_followers)}

            next_obs_dict, reward_dict, terminated_dict, truncated_dict, _ = env.step(
                action_dict,
                graph_dict,
                graph_soft_dict
            )

            next_obs_array = np.array([next_obs_dict[agent_id] for agent_id in env._agent_ids], dtype=np.float32)

            team_reward = reward_dict[env._agent_ids[0]]
            team_done = terminated_dict["__all__"] or truncated_dict["__all__"]

            episode_reward += team_reward

            obs_array = next_obs_array
            h_in = h_out
            c_in = c_out

            time.sleep(0.01)

            if team_done:
                break

        print(f"评估 {episode + 1}/{args.eval_episodes} | 步数: {step + 1:3d} | 总得分: {episode_reward:8.2f}")

    print("\n评估结束。")


if __name__ == "__main__":
    main()
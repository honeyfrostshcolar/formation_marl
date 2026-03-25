import os
import time
import argparse
import numpy as np
import torch

from envs.formation_2d_env import Formation2DMultiAgentEnv 
from agents.maddpg_agent import MADDPG_Agent             

# ==========================================
# 0. 统一参数解析 (必须与训练时保持完全一致)
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser("MADDPG + MAGIC (LSTM) Evaluation")
    
    # 环境参数 (默认展示 4 辆车的编队)
    parser.add_argument("--num_robots", type=int, default=3, help="总机器人数量(包含1个领航者)")
    parser.add_argument("--max_steps", type=int, default=1000, help="每回合最大步数")
    parser.add_argument("--sensing_radius", type=float, default=5.0, help="感知半径")
    
    # 强化学习基础参数 (评估时其实用不到，但为了初始化 Agent 不报错必须留着)
    parser.add_argument("--lr_actor", type=float, default=5e-5)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    
    # MAGIC 网络结构参数 (必须和训练时一模一样！)
    parser.add_argument("--hid_size", type=int, default=64)
    parser.add_argument("--gat_hid_size", type=int, default=64)
    parser.add_argument("--gat_num_heads", type=int, default=4)
    parser.add_argument("--gat_num_heads_out", type=int, default=1)
    parser.add_argument("--self_loop_type1", type=int, default=2)
    parser.add_argument("--self_loop_type2", type=int, default=2)
    parser.add_argument("--first_gat_normalize", action="store_true", default=False)
    parser.add_argument("--second_gat_normalize", action="store_true", default=False)
    
    parser.add_argument("--use_gat_encoder", action="store_true", default=False)
    parser.add_argument("--first_graph_complete", action="store_true", default=False)
    parser.add_argument("--learn_second_graph", action="store_true", default=True)
    parser.add_argument("--second_graph_complete", action="store_true", default=False)
    parser.add_argument("--message_encoder", action="store_true", default=False)
    parser.add_argument("--message_decoder", action="store_true", default=False)
    parser.add_argument("--comm_init", type=str, default="zeros")
    parser.add_argument("--directed", action="store_true", default=True)
    parser.add_argument("--comm_mask_zero", action="store_true", default=False)
    parser.add_argument("--gat_encoder_out_size", type=int, default=64)
    parser.add_argument("--ge_num_heads", type=int, default=4)
    parser.add_argument("--gat_encoder_normalize", action="store_true", default=False)

    args = parser.parse_args()
    
    args.num_followers = args.num_robots - 1
    args.nagents = args.num_followers
    args.action_dim = 2 
    
    args.obs_size = 34
    
    return args

# ==========================================
# 辅助函数：只加载模型权重
# ==========================================
def load_evaluate_model(agent, path):
    checkpoint_file = os.path.join(path, "maddpg_checkpoint.pt")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"🚨 找不到模型文件，请检查路径: {checkpoint_file}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # 评估时只需要演员(Actor)的大脑，裁判(Critic)可以休息了
    agent.actor.load_state_dict(checkpoint['actor'])
    print(f"✅ 成功加载第 {checkpoint['episode']} 轮的完美大脑！")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 评估测试启动，运行设备: {device}")

    # ==========================================
    # 🚨 第一步：填写你最终训练的模型路径！(请替换成你真实的 latest_checkpoint 路径)
    # ==========================================
    model_path = "/home/nankai/formation_test/data/MADDPG_Formation_f907bf_2026-03-25_16-56-30/latest_checkpoint"
    
    # 获取参数
    args = parse_args()

    # ==========================================
    # 2. 初始化环境 (开启渲染) 和 Agent
    # ==========================================
    config = {
        "num_robots": args.num_robots, 
        "max_steps": args.max_steps, 
        "render": True,   # ✅ 必须开启渲染！我们要看动画！
        "sensing_radius": args.sensing_radius
    }
    env = Formation2DMultiAgentEnv(config)
    
    # ✅ 核心修改：使用统一的 args 初始化 Agent
    agent = MADDPG_Agent(args)
    
    # 加载你的心血结晶
    load_evaluate_model(agent, model_path)

    # ==========================================
    # 3. 开始观赏表演
    # ==========================================
    eval_episodes = 10  # 跑 10 局看看稳定性
    
    for episode in range(eval_episodes):
        obs_dict, _ = env.reset()
        obs_array = np.array([obs_dict[agent_id] for agent_id in env._agent_ids])
        
        # ✅ LSTM 核心：评估回合开始，初始化全新的空记忆！
        h_in, c_in = agent.init_hidden()
        
        episode_reward = 0.0
        
        for step in range(config["max_steps"]):
            # ✅ 核心：传入 h_in, c_in，并且 add_noise=False 纯靠真实实力！
            actions_array, graphs_array, graphs_soft_array, h_out, c_out = agent.select_action(
                obs_array, h_in, c_in, add_noise=False
            )
            
            action_dict = {env._agent_ids[i]: actions_array[i] for i in range(args.num_followers)}
            graph_dict = {env._agent_ids[i]: graphs_array[i] for i in range(args.num_followers)}
            graph_soft_dict = {env._agent_ids[i]: graphs_soft_array[i] for i in range(args.num_followers)}
            
            # 环境步进
            next_obs_dict, reward_dict, terminated_dict, truncated_dict, _ = env.step(action_dict, graph_dict, graph_soft_dict)
            next_obs_array = np.array([next_obs_dict[agent_id] for agent_id in env._agent_ids])
            
            team_reward = reward_dict[env._agent_ids[0]]
            team_done = terminated_dict["__all__"] or truncated_dict["__all__"]
            
            episode_reward += team_reward
            
            # ✅ 记忆流动：走向下一步
            obs_array = next_obs_array
            h_in = h_out
            c_in = c_out
            
            # ✅ 为了让你肉眼能看清阵型的变化，强制加一点延时，否则画面一闪而过
            time.sleep(0.01) 
            
            if team_done:
                break
                
        print(f"🎬 评估测试 {episode + 1}/{eval_episodes} | 存活步数: {step+1:3d} | 总得分: {episode_reward:8.2f}")

    print("\n🎉 评估结束！！")

if __name__ == "__main__":
    main()
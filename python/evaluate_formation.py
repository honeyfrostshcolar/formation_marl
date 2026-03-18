import os
import time
import numpy as np
import torch

from envs.formation_2d_env import Formation2DMultiAgentEnv 
from agents.maddpg_agent import MADDPG_Agent             

# ==========================================
# 辅助函数：只加载模型权重，不搞经验池
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
    model_path = "/home/nankai/formation_test/data/MADDPG_Formation_560a3e_2026-03-18_17-54-54/latest_checkpoint"
    
    # ==========================================
    # 2. 初始化环境 (开启渲染)
    # ==========================================
    config = {
        "num_robots": 3, 
        "max_steps": 1000, # 可以设长一点，看看能坚持多久
        "render": True,   # ✅ 必须开启渲染！我们要看动画！
        "sensing_radius": 5.0
    }
    env = Formation2DMultiAgentEnv(config)
    num_followers = env.num_followers
    agent_ids = env._agent_ids
    
    obs_dim = env.observation_space.shape[0] 
    action_dim = env.action_space.shape[0]   

    agent = MADDPG_Agent(num_followers=num_followers, obs_dim=obs_dim, action_dim=action_dim)
    
    # 加载你的心血结晶
    load_evaluate_model(agent, model_path)

    # ==========================================
    # 3. 开始观赏表演
    # ==========================================
    eval_episodes = 10  # 跑 10 局看看稳定性
    
    for episode in range(eval_episodes):
        obs_dict, _ = env.reset()
        obs_array = np.array([obs_dict[agent_id] for agent_id in agent_ids])
        
        episode_reward = 0.0
        
        for step in range(config["max_steps"]):
            # ✅ 核心：add_noise=False 绝对不加任何探索噪音！纯靠真实实力！
            actions_array, graphs_array, graphs_soft_array = agent.select_action(obs_array, add_noise=False)
            action_dict = {agent_ids[i]: actions_array[i] for i in range(num_followers)}
            graph_dict = {agent_ids[i]: graphs_array[i] for i in range(num_followers)}
            graph_soft_dict = {agent_ids[i]: graphs_soft_array[i] for i in range(num_followers)}
            
            # 环境步进
            next_obs_dict, reward_dict, terminated_dict, truncated_dict, _ = env.step(action_dict, graph_dict, graph_soft_dict)
            next_obs_array = np.array([next_obs_dict[agent_id] for agent_id in agent_ids])
            
            team_reward = reward_dict[agent_ids[0]]
            team_done = terminated_dict["__all__"] or truncated_dict["__all__"]
            
            episode_reward += team_reward
            obs_array = next_obs_array
            
            # ✅ 为了让你肉眼能看清阵型的变化，强制加一点延时，否则画面一闪而过
            time.sleep(0.01) 
            
            if team_done:
                break
                
        print(f"🎬 评估测试 {episode + 1}/{eval_episodes} | 存活步数: {step+1:3d} | 总得分: {episode_reward:8.2f}")

    print("\n🎉 评估结束！你的算法非常棒！")

if __name__ == "__main__":
    main()
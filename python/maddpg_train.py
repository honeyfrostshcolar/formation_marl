# 文件路径: train_maddpg.py (项目根目录)
import numpy as np
import torch

from envs.formation_2d_env import Formation2DMultiAgentEnv 
from agents.maddpg_agent import MADDPG_Agent             
from utils.replay_buffer import ReplayBuffer             

def main():
    # 检测 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training starting on device: {device}")

    # ==========================================
    # 1. 初始化环境
    # ==========================================
    config = {
        "num_robots": 5, 
        "max_steps": 200, 
        "render": True,          # 刚开始调代码建议开 True 看小车动不动，跑通后关掉提速
        "sensing_radius": 5.0
    }
    env = Formation2DMultiAgentEnv(config)
    
    num_followers = env.num_followers
    agent_ids = env._agent_ids
    
    # 动态获取观测和动作的维度，防止写死
    obs_dim = env.observation_space.shape[0]  # 我们之前算的是 32
    action_dim = env.action_space.shape[0]    # 我们定的是 2 (dx, dy)

    # ==========================================
    # 2. 初始化 MADDPG 智能体和经验池
    # ==========================================
    agent = MADDPG_Agent(num_followers=num_followers, obs_dim=obs_dim, action_dim=action_dim)
    
    # 注意：ReplayBuffer 需要传入维度参数，用来开辟 numpy 内存矩阵
    buffer = ReplayBuffer(
        capacity=100000, 
        num_followers=num_followers, 
        obs_dim=obs_dim, 
        action_dim=action_dim, 
        device=device
    )

    batch_size = 64 # 每次从经验池里抽 64 条数据更新网络

    # ==========================================
    # 3. 开始炼丹大循环
    # ==========================================
    for episode in range(10000):
        # 重置环境，拿到初始字典
        obs_dict, _ = env.reset()
        
        # ✅ 将字典转为 numpy 数组: 形状 (num_followers, obs_dim)
        obs_array = np.array([obs_dict[agent_id] for agent_id in agent_ids])
        
        episode_reward = 0.0 # 记录这一局的总得分
        
        for step in range(config["max_steps"]):
            # 1. 神经网络做决策 (加入 高斯/OU 探索噪声)
            actions_array = agent.select_action(obs_array, add_noise=True)
            
            # ✅ 将输出的数组转回字典，喂给环境
            action_dict = {agent_ids[i]: actions_array[i] for i in range(num_followers)}
            
            # 2. 环境执行物理步进
            next_obs_dict, reward_dict, terminated_dict, truncated_dict, _ = env.step(action_dict)
            
            # 处理下一状态的观测数组
            next_obs_array = np.array([next_obs_dict[agent_id] for agent_id in agent_ids])
            
            # ✅ 提取团队总分和结束标志 (因为是团队大锅饭，随便取一个 follower_0 的分就行)
            team_reward = reward_dict[agent_ids[0]]
            # 只要触发了 __all__，就代表团队全剧终 (比如有人撞墙了)
            team_done = terminated_dict["__all__"] or truncated_dict["__all__"]
            
            # 3. 把这一步的经验存入 Buffer (注意 reward 和 done 传的是标量 float)
            buffer.store(obs_array, actions_array, float(team_reward), next_obs_array, float(team_done))
            
            episode_reward += team_reward
            
            # 4. 如果 Buffer 里的数据攒够了 batch_size，就开始高频更新网络！
            if buffer.size() >= batch_size:
                sample_batch = buffer.sample(batch_size)
                actor_loss, critic_loss = agent.update(sample_batch)
            
            # 状态滚动
            obs_array = next_obs_array
            
            # 如果全剧终，跳出这一局，进入下一局的 reset
            if team_done:
                break
                
        # 每局结束打印一下进度
        print(f"Episode: {episode:5d} | Steps: {step+1:3d} | Reward: {episode_reward:8.2f}")

if __name__ == "__main__":
    main()
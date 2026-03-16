import os
import time
import uuid
import shutil
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter # 原生 TensorBoard

from envs.formation_2d_env import Formation2DMultiAgentEnv 
from agents.maddpg_agent import MADDPG_Agent             
from utils.replay_buffer import ReplayBuffer             

# ==========================================
# 辅助函数：保存和加载 PyTorch 模型状态
# ==========================================
def save_checkpoint(agent, episode, path):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        'episode': episode,
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'target_actor': agent.target_actor.state_dict(),
        'target_critic': agent.target_critic.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'critic_optimizer': agent.critic_optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(path, "maddpg_checkpoint.pt"))

def load_checkpoint(agent, path):
    checkpoint_file = os.path.join(path, "maddpg_checkpoint.pt")
    if os.path.exists(checkpoint_file):
        # 兼容 CPU 和 GPU 加载
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        agent.target_actor.load_state_dict(checkpoint['target_actor'])
        agent.target_critic.load_state_dict(checkpoint['target_critic'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        return checkpoint['episode']
    return 0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training starting on device: {device}")

    # ==========================================
    # 1. 训练参数与工程目录设置
    # ==========================================
    train_iterations = 1000
    base_save_dir = "/home/lpp/formation_test/data" # 你的数据保存目录
    
    # ⚠️ 断点续训设置 
    # 如果想从头训练，保持 None；如果想继续，填入 latest_checkpoint 路径
    # resume_checkpoint = None  
    resume_checkpoint = "/home/lpp/formation_test/data/MADDPG_Formation_94ece9_2026-03-16_16-25-47/latest_checkpoint" 

    # 生成本次运行专属的文件夹名字
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    random_suffix = uuid.uuid4().hex[:6]
    task_dir = f"MADDPG_Formation_{random_suffix}_{timestamp}"
    save_root_dir = os.path.join(base_save_dir, task_dir)
    os.makedirs(save_root_dir, exist_ok=True)
    
    fixed_checkpoint_dir = os.path.join(save_root_dir, "latest_checkpoint")
    print(f"📁 本次训练的根目录：{save_root_dir}")

    # 初始化原生 TensorBoard 记录器
    writer = SummaryWriter(log_dir=save_root_dir)

    # ==========================================
    # 2. 初始化环境、智能体和经验池
    # ==========================================
    config = {
        "num_robots": 3, 
        "max_steps": 200, 
        "render": False,  # ⚠️ 训练时必须关闭渲染以保证速度！
        "sensing_radius": 5.0
    }
    env = Formation2DMultiAgentEnv(config)
    num_followers = env.num_followers
    agent_ids = env._agent_ids
    
    obs_dim = env.observation_space.shape[0] 
    action_dim = env.action_space.shape[0]   

    agent = MADDPG_Agent(num_followers=num_followers, obs_dim=obs_dim, action_dim=action_dim)
    
    buffer = ReplayBuffer(
        capacity=10000, 
        num_followers=num_followers, 
        obs_dim=obs_dim, 
        action_dim=action_dim, 
        device=device
    )
    batch_size = 256 # 每次更新的批量大小

    # ==========================================
    # 3. 执行断点续训加载逻辑
    # ==========================================
    start_episode = 0
    if resume_checkpoint is not None:
        if os.path.exists(resume_checkpoint):
            print(f"\n🔄 [恢复训练] 正在加载检查点：{resume_checkpoint}")
            start_episode = load_checkpoint(agent, resume_checkpoint)
            buffer.load(resume_checkpoint) # 同步加载经验池状态
            print(f"✅ [恢复成功] 将从第 {start_episode} 轮继续训练！\n")
        else:
            print(f"\n⚠️ [警告] 检查点不存在：{resume_checkpoint}，从头开始训练！\n")
    else:
        print("\n✨ [全新训练] 从头开始训练...\n")

    # ==========================================
    # 4. 开始炼丹大循环
    # ==========================================
    for episode in range(start_episode, train_iterations):

        if episode < 2000:
            current_noise = 0.08
        else:
            progress = min(1.0, (episode - 2000) / 1000)
            current_noise = 0.10 - progress * 0.09
            current_noise = max(current_noise, 0.01)
        

        obs_dict, _ = env.reset()
        obs_array = np.array([obs_dict[agent_id] for agent_id in agent_ids])
        
        episode_reward = 0.0
        episode_actor_loss = []
        episode_critic_loss = []
        
        for step in range(config["max_steps"]):
            # 前向决策
            actions_array, graphs_array, graphs_soft_array = agent.select_action(obs_array, add_noise=True, noise_scale=current_noise)
            action_dict = {agent_ids[i]: actions_array[i] for i in range(num_followers)}
            graph_dict = {agent_ids[i]: graphs_array[i] for i in range(num_followers)}
            graphs_soft_dict = {agent_ids[i]: graphs_soft_array[i] for i in range(num_followers)}

            # 环境步进
            next_obs_dict, reward_dict, terminated_dict, truncated_dict, _ = env.step(action_dict, graph_dict, graphs_soft_dict)
            next_obs_array = np.array([next_obs_dict[agent_id] for agent_id in agent_ids])
            
            team_reward = reward_dict[agent_ids[0]]
            # print("terminated_dict:", terminated_dict, "truncated_dict:", truncated_dict)
            team_done = terminated_dict["__all__"] or truncated_dict["__all__"]
            # print( "team_done:", team_done)
            
            # 存入经验池
            buffer.store(obs_array, actions_array, float(team_reward), next_obs_array, float(team_done))
            episode_reward += team_reward
            
            # 核心：更新网络
            learning_starts = 5000  # 先积累一些经验再开始学习
            if buffer.size() >= learning_starts:
                sample_batch = buffer.sample(batch_size)
                a_loss, c_loss = agent.update(sample_batch)
                episode_actor_loss.append(a_loss)
                episode_critic_loss.append(c_loss)
            
            obs_array = next_obs_array
            
            if team_done:
                break
                
        # 计算整局平均 Loss
        avg_a_loss = np.mean(episode_actor_loss) if episode_actor_loss else 0.0
        avg_c_loss = np.mean(episode_critic_loss) if episode_critic_loss else 0.0

        step_avg_reward = episode_reward / (step + 1)

        # 控制台打印进度
        print(f"Episode: {episode:5d} | Steps: {step+1:3d} | Reward: {step_avg_reward:8.2f} | A_Loss: {avg_a_loss:.4f} | C_Loss: {avg_c_loss:.4f}")
        
        # ✅ TensorBoard 记录曲线
        writer.add_scalar("Training/Episode_Reward", episode_reward, episode)
        writer.add_scalar("Training/Episode_Length", step + 1, episode)
        writer.add_scalar("Loss/Actor_Loss", avg_a_loss, episode)
        writer.add_scalar("Loss/Critic_Loss", avg_c_loss, episode)
        writer.add_scalar("Training/Avg_Step_Reward", step_avg_reward, episode)

        # ✅ 每 50 局保存一次检查点，并覆盖 latest_checkpoint
        if (episode + 1) % 50 == 0:
            if os.path.exists(fixed_checkpoint_dir):
                shutil.rmtree(fixed_checkpoint_dir, ignore_errors=True)
            save_checkpoint(agent, episode + 1, fixed_checkpoint_dir)
            buffer.save(fixed_checkpoint_dir)
            print(f"💾 [Checkpoint 已更新] 最新模型 -> {fixed_checkpoint_dir}")

    # 训练彻底结束时保存最终模型
    if os.path.exists(fixed_checkpoint_dir):
        shutil.rmtree(fixed_checkpoint_dir, ignore_errors=True)
    save_checkpoint(agent, train_iterations, fixed_checkpoint_dir)
    buffer.save(fixed_checkpoint_dir)
    print(f"\n🎉 训练全部结束！最终模型保存在: {fixed_checkpoint_dir}")
    writer.close()

if __name__ == "__main__":
    main()
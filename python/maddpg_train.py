import os
import time
import uuid
import shutil
import numpy as np
import torch
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser("MADDPG + MAGIC (LSTM) for Formation Control")
    
    # ==========================================
    # 一、 环境与训练总体参数
    # ==========================================
    parser.add_argument("--num_robots", type=int, default=3, help="总机器人数量 (包含1个领航者和N个跟随者)")
    parser.add_argument("--max_steps", type=int, default=1000, help="每回合(Episode)环境交互的最大步数限制")
    parser.add_argument("--train_iterations", type=int, default=5000, help="总训练回合数 (Episodes)")
    parser.add_argument("--batch_size", type=int, default=256, help="每次网络更新时从经验池采样的批量大小")
    parser.add_argument("--buffer_capacity", type=int, default=10000, help="经验回放池(Replay Buffer)的最大容量")
    parser.add_argument("--sensing_radius", type=float, default=5.0, help="机器人的局部最大感知半径 (米)")
    parser.add_argument("--render", action="store_true", default=False, help="加上这个参数就开启画面渲染 (⚠️训练时建议设为False以提升速度)")
    
    # ==========================================
    # 二、 MADDPG 强化学习算法基础参数
    # ==========================================
    parser.add_argument("--lr_actor", type=float, default=5e-5, help="Actor 策略网络的学习率")
    parser.add_argument("--lr_critic", type=float, default=3e-4, help="Critic 价值网络的学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="强化学习奖励折扣因子 (Gamma，越接近1越看重长期收益)")
    parser.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数 (Tau，控制新老权重融合比例)")
    
    # ==========================================
    # 三、 MAGIC (多智能体图注意力通信) 核心架构参数
    # ==========================================
    # 3.1 基础维度设置
    parser.add_argument("--hid_size", type=int, default=64, help="所有 MLP 隐藏层及 LSTM 记忆单元的特征维度")
    parser.add_argument("--gat_hid_size", type=int, default=64, help="子处理器(GAT)内部进行图通信聚合时的特征维度")
    
    # 3.2 图注意力层(GAT)配置
    parser.add_argument("--gat_num_heads", type=int, default=4, help="第一轮通信 GAT 层使用的多头注意力(Multi-head)数量")
    parser.add_argument("--gat_num_heads_out", type=int, default=1, help="第二轮通信 GAT 层使用的多头注意力数量")
    parser.add_argument("--self_loop_type1", type=int, default=2, help="第一层GAT自环类型 (0:强行无自环, 1:强行加自环, 2:完全由调度器学习决定)")
    parser.add_argument("--self_loop_type2", type=int, default=2, help="第二层GAT自环类型 (0:强行无自环, 1:强行加自环, 2:完全由调度器学习决定)")
    parser.add_argument("--first_gat_normalize", action="store_true", default=False, help="是否对第一层 GAT 计算出的注意力权重进行归一化")
    parser.add_argument("--second_gat_normalize", action="store_true", default=False, help="是否对第二层 GAT 计算出的注意力权重进行归一化")
    
    # 3.3 子调度器(Sub-scheduler)图生成控制
    parser.add_argument("--use_gat_encoder", action="store_true", default=False, help="调度器在决定谁跟谁通信前，是否先用额外的 GAT 编码器提取特征 (否则用普通MLP)")
    parser.add_argument("--first_graph_complete", action="store_true", default=False, help="第一轮通信是否跳过调度网络，强行让所有人建立全连接图通信")
    parser.add_argument("--learn_second_graph", action="store_true", default=True, help="是否激活第二个调度网络来动态学习第二轮的通信拓扑图")
    parser.add_argument("--second_graph_complete", action="store_true", default=False, help="第二轮通信是否跳过调度网络，强行让所有人建立全连接图通信")
    parser.add_argument("--directed", action="store_true", default=True, help="学习出的通信图是否有向 (True=有向图，即A理B但不代表B理A; False=无向图)")
    parser.add_argument("--comm_mask_zero", action="store_true", default=False, help="是否强行切断所有人的通信 (仅用于做消融实验，证明通信的必要性)")
    
    # 3.4 调度器 GAT 编码器专属参数 (仅在 use_gat_encoder=True 时生效)
    parser.add_argument("--gat_encoder_out_size", type=int, default=64, help="调度器专属 GAT 编码器输出的隐藏层维度")
    parser.add_argument("--ge_num_heads", type=int, default=4, help="调度器专属 GAT 编码器的多头注意力数量")
    parser.add_argument("--gat_encoder_normalize", action="store_true", default=False, help="调度器专属 GAT 编码器是否进行注意力权重归一化")
    
    # 3.5 消息的额外加工层
    parser.add_argument("--message_encoder", action="store_true", default=False, help="在把隐藏状态当作消息发出去之前，是否先通过一个全连接层进行预处理")
    parser.add_argument("--message_decoder", action="store_true", default=False, help="在收到全局消息后，是否先通过一个全连接层进行解码再输入动作网络")
    parser.add_argument("--comm_init", type=str, default="zeros", help="通信相关网络全连接层权重的初始赋零方式 (如 'zeros' 防止初始通信引发混乱)")

    # [新增] CoDe 相关
    parser.add_argument("--intent_dim", type=int, default=32)
    parser.add_argument("--decoder_hidden_dim", type=int, default=64)
    parser.add_argument("--value_dim", type=int, default=64)
    parser.add_argument("--attn_dim", type=int, default=32)
    parser.add_argument("--gamma_t", type=float, default=0.90)
    parser.add_argument("--lambda_inf", type=float, default=1.0)
    parser.add_argument("--lambda_c", type=float, default=0.1)
    parser.add_argument("--lambda_k", type=float, default=1e-3)
    parser.add_argument("--lambda_e", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--renorm_after_decay", action="store_true", default=False)
    parser.add_argument("--pred_horizon", type=int, default=4)

    # [新增] 延迟相关
    parser.add_argument("--delay_mode", type=str, default="fixed", choices=["none", "fixed", "uniform"])
    parser.add_argument("--fixed_delay", type=int, default=2)
    parser.add_argument("--min_delay", type=int, default=1)
    parser.add_argument("--max_delay", type=int, default=3)

    args = parser.parse_args()
    
    args.num_followers = args.num_robots - 1
    args.nagents = args.num_followers # 对齐原版变量名
    args.action_dim = 2 
    args.obs_size = 34
    
    return args

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training starting on device: {device}")

    # ==========================================
    # 1. 训练参数与工程目录设置
    # ==========================================
    train_iterations = args.train_iterations
    base_save_dir = "/home/nankai/formation_test/data" # 你的数据保存目录
    
    # ⚠️ 断点续训设置 
    # 如果想从头训练，保持 None；如果想继续，填入 latest_checkpoint 路径
    resume_checkpoint = None  
    # resume_checkpoint = "/home/nankai/formation_test/data/MADDPG_Formation_e54cc4_2026-03-25_11-46-10/latest_checkpoint" 

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

    config = {
        "num_robots": args.num_robots, 
        "max_steps": args.max_steps, 
        "render": args.render,  # <--- 动态读取命令行参数
        "sensing_radius": args.sensing_radius
    }

    # ==========================================
    # 2. 初始化环境、智能体和经验池
    # ==========================================
    env = Formation2DMultiAgentEnv(config)
    agent = MADDPG_Agent(args)
    buffer = ReplayBuffer(args.buffer_capacity, args.num_followers, args.obs_size, args.action_dim, args.hid_size, device)
    
    batch_size = args.batch_size # 每次更新的批量大小

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
        obs_array = np.array([obs_dict[agent_id] for agent_id in env._agent_ids])
        

        h_in, c_in = agent.init_hidden()
        agent.reset_runtime()  # [新增] 每个 episode 都清 runtime delay buffer

        episode_reward = 0.0
        actor_losses = []
        critic_losses = []
        
        for step in range(args.max_steps):
            # 前向决策
            actions_array, graphs_array, graphs_soft_array, h_out, c_out, comm_snapshot = agent.select_action(
                obs_array, h_in, c_in, add_noise=True, noise_scale=current_noise
            )
            action_dict = {env._agent_ids[i]: actions_array[i] for i in range(args.num_followers)}
            graph_dict = {env._agent_ids[i]: graphs_array[i] for i in range(args.num_followers)}
            graphs_soft_dict = {env._agent_ids[i]: graphs_soft_array[i] for i in range(args.num_followers)}

            # 环境步进
            next_obs_dict, reward_dict, terminated_dict, truncated_dict, _ = env.step(action_dict, graph_dict, graphs_soft_dict)
            next_obs_array = np.array([next_obs_dict[agent_id] for agent_id in env._agent_ids])
            
            team_reward = reward_dict[env._agent_ids[0]]
            # print("terminated_dict:", terminated_dict, "truncated_dict:", truncated_dict)
            team_done = terminated_dict["__all__"] or truncated_dict["__all__"]
            # print( "team_done:", team_done)
            
            # 存入经验池
            # 存入 Buffer 时，把旧记忆(in)和新记忆(out)一起转成 numpy 存进去
            buffer.store({
                "obs": obs_array,
                "action": actions_array,
                "reward": float(team_reward),
                "next_obs": next_obs_array,
                "done": float(team_done),
                "h_in": h_in.squeeze(0).cpu().numpy(),
                "c_in": c_in.squeeze(0).cpu().numpy(),
                "h_out": h_out.squeeze(0).cpu().numpy(),
                "c_out": c_out.squeeze(0).cpu().numpy(),
                "prev_action": comm_snapshot["prev_action"],
                "route_hard": comm_snapshot["route_hard"],
                "route_soft": comm_snapshot["route_soft"],
                "sender_intents_recv": comm_snapshot["sender_intents_recv"],
                "sender_hidden_recv": comm_snapshot["sender_hidden_recv"],
                "recv_mask": comm_snapshot["recv_mask"],
                "time_lags": comm_snapshot["time_lags"],
                "episode_id": episode,
                "step_id": step,
            })

            obs_array = next_obs_array
            h_in, c_in = h_out, c_out
            episode_reward += team_reward

            learning_starts = 2000  # 先积累一些经验再开始学习
            if buffer.size() >= learning_starts:
                metrics = agent.update(buffer)
                actor_losses.append(metrics["actor_total_loss"])
                critic_losses.append(metrics["critic_loss"])
            
            if team_done:
                break
                
        # 计算整局平均 Loss
        avg_a_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_c_loss = np.mean(critic_losses) if critic_losses else 0.0

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
import os
import time
import uuid
import shutil
import numpy as np
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter # 原生 TensorBoard

from envs.formation_2d_env import Formation2DMultiAgentEnv 
from agents.maddpg_agent import MASAC_Agent           
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
        'log_alpha': agent.log_alpha.detach().cpu(),
        'alpha_optimizer': agent.alpha_optimizer.state_dict(),
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

        if 'log_alpha' in checkpoint:
            agent.log_alpha.data.copy_(checkpoint['log_alpha'].to(agent.device))
        if 'alpha_optimizer' in checkpoint:
            agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

        return checkpoint['episode']
    return 0

CURRICULUM_SCHEDULE = [
    (0,    {"open": 1.0}),                                   
    (500, {"open": 0.4, "star_map": 0.6}),                     
    (3000, {"open": 0.4, "star_map": 0.5, "z_map": 0.1}),                     
    (3500, {"open": 0.4, "star_map": 0.4, "z_map": 0.2}),                     
    (4000, {"open": 0.2, "star_map": 0.3, "z_map": 0.5}),                     
    # (5000, {"open": 0.1, "star_map": 0.2, "z_map": 0.2, "custom": 0.5}) 
]

MAP_SEED_BASE = {
    "open": 1000,
    "z_map": 2000,
    "star_map": 3000,
    "custom": 4000,
}

def choose_map_mode(episode: int) -> str:
    """
    根据当前的 episode，在课程表中寻找对应的概率分布并采样地图
    """
    current_probs = CURRICULUM_SCHEDULE[0][1]
    
    # 找到最后一个生效的阶段
    for start_ep, probs in CURRICULUM_SCHEDULE:
        if episode >= start_ep:
            current_probs = probs
            
    maps = list(current_probs.keys())
    probs = list(current_probs.values())
    
    return np.random.choice(maps, p=probs)


@torch.no_grad()
def evaluate_on_map(agent, env, map_mode: str, eval_episodes: int, max_steps: int) -> float:
    """
    在指定地图上跑若干局，返回平均 episode reward。
    """
    rewards = []

    rng_state = np.random.get_state()

    for i in range(eval_episodes):
        env.set_map_mode(map_mode)

        eval_seed = MAP_SEED_BASE[map_mode] + i
        obs_dict, _ = env.reset(seed=eval_seed)
        obs_array = np.array([obs_dict[agent_id] for agent_id in env._agent_ids], dtype=np.float32)

        h_in, c_in = agent.init_hidden()
        agent.reset_runtime()

        episode_reward = 0.0

        for step in range(max_steps):
            action_policy, action_exec, graphs_array, graphs_soft_array, h_out, c_out, comm_snapshot = agent.select_action(
                obs_array,
                h_in,
                c_in,
                deterministic=True,
            )

            action_dict = {env._agent_ids[i]: action_exec[i] for i in range(agent.num_followers)}
            graph_dict = {env._agent_ids[i]: graphs_array[i] for i in range(agent.num_followers)}
            graph_soft_dict = {env._agent_ids[i]: graphs_soft_array[i] for i in range(agent.num_followers)}

            next_obs_dict, reward_dict, terminated_dict, truncated_dict, _ = env.step(
                action_dict,
                graph_dict,
                graph_soft_dict,
            )

            next_obs_array = np.array([next_obs_dict[agent_id] for agent_id in env._agent_ids], dtype=np.float32)

            team_reward = reward_dict[env._agent_ids[0]]
            team_done = terminated_dict["__all__"] or truncated_dict["__all__"]

            episode_reward += team_reward
            obs_array = next_obs_array
            h_in, c_in = h_out, c_out

            if team_done:
                break

        step_avg_reward = episode_reward / (step + 1)
        rewards.append(step_avg_reward)

    np.random.set_state(rng_state)

    return float(np.mean(rewards))

def parse_args():
    parser = argparse.ArgumentParser(description="MADDPG + MAGIC Scheduler + CoDe Delay-aware Fusion for multi-robot formation control")

    # =========================================================
    # 一、环境与训练总体参数
    # =========================================================
    parser.add_argument("--num_robots", type=int, default=3, help="环境中的总机器人数量，包含 1 个 leader 和 N-1 个 follower。")
    parser.add_argument("--max_steps", type=int, default=1000, help="每个 episode 的最大交互步数，达到后即超时终止。")
    parser.add_argument("--train_iterations", type=int, default=10000, help="总训练回合数（episode 数）。")
    parser.add_argument("--batch_size", type=int, default=256, help="每次网络更新时从经验池采样的 batch 大小。")
    parser.add_argument("--buffer_capacity", type=int, default=10000, help="经验回放池最多可存储的 transition 数量。")
    parser.add_argument("--sensing_radius", type=float, default=5.0, help="每个 follower 的局部感知半径，超出该范围的队友不会进入观测。")
    parser.add_argument("--render", action="store_true", default=False, help="是否开启环境渲染。训练时通常关闭以提升速度。")

    # =========================================================
    # 二、MADDPG 强化学习参数
    # =========================================================
    parser.add_argument("--lr_actor", type=float, default=5e-5, help="Actor（策略网络）的学习率。")
    parser.add_argument("--lr_critic", type=float, default=3e-4, help="Critic（价值网络）的学习率。")
    parser.add_argument("--gamma", type=float, default=0.99, help="奖励折扣因子 gamma，越接近 1 越重视长期回报。")
    parser.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数 tau。")

    # =========================================================
    # 三、MAGIC 调度器 / 通信拓扑参数
    # =========================================================
    parser.add_argument("--hid_size", type=int, default=64, help="LSTM 隐状态维度，同时也是大多数 MLP 的基础隐藏维度。")
    parser.add_argument("--gat_hid_size", type=int, default=64, help="GAT 中间特征维度。当前如果只保留 scheduler，可主要作为兼容参数保留。")
    parser.add_argument("--gat_num_heads", type=int, default=4, help="第一轮 GAT 使用的多头注意力头数。")
    parser.add_argument("--gat_num_heads_out", type=int, default=1, help="第二轮 GAT 使用的多头注意力头数。")
    parser.add_argument("--self_loop_type1", type=int, default=2, help="第一层自环控制方式：0=无自环，1=强制自环，2=由调度机制控制。")
    parser.add_argument("--self_loop_type2", type=int, default=2, help="第二层自环控制方式：0=无自环，1=强制自环，2=由调度机制控制。")
    parser.add_argument("--first_gat_normalize", action="store_true", default=False, help="是否对第一层 GAT 的注意力权重做归一化。")
    parser.add_argument("--second_gat_normalize", action="store_true", default=False, help="是否对第二层 GAT 的注意力权重做归一化。")
    parser.add_argument("--use_gat_encoder", action="store_true", default=False, help="是否在 scheduler 前使用额外的 GAT 编码器提取全局关系特征。")
    parser.add_argument("--first_graph_complete", action="store_true", default=False, help="是否令第一轮通信图强制为全连接图，而不经过第一轮调度器学习。")
    parser.add_argument("--learn_second_graph", dest="learn_second_graph", action="store_true", help="是否启用第二轮调度器学习第二张通信图。")
    parser.add_argument("--no_learn_second_graph", dest="learn_second_graph", action="store_false", help="是否关闭第二轮调度器，只保留第一轮通信图。")
    parser.set_defaults(learn_second_graph=True)
    parser.add_argument("--second_graph_complete", action="store_true", default=False, help="是否令第二轮通信图强制为全连接图，而不经过第二轮调度器学习。")
    parser.add_argument("--directed", dest="directed", action="store_true", help="使用有向通信图，表示 A->B 和 B->A 可以不同。")
    parser.add_argument("--undirected", dest="directed", action="store_false", help="使用无向通信图，表示 A<->B 对称。")
    parser.set_defaults(directed=True)
    parser.add_argument("--comm_mask_zero", action="store_true", default=False, help="将通信图强制置零，用于做无通信消融实验。")

    # =========================================================
    # 四、Scheduler 专属 GAT 编码器参数
    # =========================================================
    parser.add_argument("--gat_encoder_out_size", type=int, default=64, help="scheduler 专属 GAT encoder 的输出维度。")
    parser.add_argument("--ge_num_heads", type=int, default=4, help="scheduler 专属 GAT encoder 的多头注意力头数。")
    parser.add_argument("--gat_encoder_normalize", action="store_true", default=False, help="scheduler 专属 GAT encoder 是否对注意力权重做归一化。")

    # =========================================================
    # 五、消息前处理 / 后处理
    # =========================================================
    parser.add_argument("--message_encoder", action="store_true", default=False, help="是否在发送端对 hidden state 先做一层线性映射后再作为消息发送。")
    parser.add_argument("--message_decoder", action="store_true", default=False, help="是否在接收端对融合后的消息再做一层线性映射。")
    parser.add_argument("--comm_init", type=str, default="zeros", help="通信相关层的初始化方式。通常 'zeros' 可减轻训练初期通信震荡。")

    # =========================================================
    # 六、CoDe：发送端意图建模参数
    # =========================================================
    parser.add_argument("--intent_dim", type=int, default=32, help="意图向量 e_t 的维度。")
    parser.add_argument("--decoder_hidden_dim", type=int, default=64, help="发送端 Intent Decoder 中 GRU 解码器的隐藏状态维度。")
    parser.add_argument("--pred_horizon", type=int, default=4, help="发送端 decoder 预测未来动作的时间跨度 K。")

    # =========================================================
    # 七、CoDe：接收端双对齐融合参数
    # =========================================================
    parser.add_argument("--value_dim", type=int, default=64, help="接收端融合后消息向量 c_t 的维度。")
    parser.add_argument("--attn_dim", type=int, default=32, help="intent alignment 中 query/key 的投影维度。")
    parser.add_argument("--gamma_t", type=float, default=0.90, help="timeliness alignment 中的时间衰减系数，越小表示越惩罚旧消息。")
    parser.add_argument("--renorm_after_decay", action="store_true", default=False, help="是否在时间衰减后对 attention 权重重新归一化。默认按论文正文不重新归一化。")

    # =========================================================
    # 八、CoDe：损失函数权重参数
    # =========================================================
    parser.add_argument("--lambda_inf", type=float, default=1.0, help="未来动作推断损失 L_inf 的权重。")
    parser.add_argument("--lambda_c", type=float, default=0.1, help="意图连续性损失 L_c 的权重。")
    parser.add_argument("--lambda_k", type=float, default=1e-3, help="KL 正则损失 L_k 的权重。")
    parser.add_argument("--lambda_e", type=float, default=1e-3, help="接收端 dual alignment 中熵正则项 L_e 的权重。")
    parser.add_argument("--eps", type=float, default=1e-8, help="数值稳定项，防止除零或 log(0)。")

    # =========================================================
    # 九、延迟通信设置
    # =========================================================
    parser.add_argument("--delay_mode", type=str, default="fixed", choices=["none", "fixed", "uniform"], help="通信延迟模式：none=无延迟，fixed=固定延迟，uniform=随机均匀延迟。")
    parser.add_argument("--fixed_delay", type=int, default=2, help="当 delay_mode=fixed 时，所有 sender->receiver 边统一采用的固定延迟步数。")
    parser.add_argument("--min_delay", type=int, default=1, help="当 delay_mode=uniform 时，最小延迟步数。")
    parser.add_argument("--max_delay", type=int, default=3, help="当 delay_mode=uniform 时，最大延迟步数。")

    parser.add_argument("--eval_every", type=int, default=50, help="每隔多少个 episode 做一次多地图综合验证。")
    parser.add_argument("--eval_episodes", type=int, default=3, help="每种地图评估多少个 episode。")

    # =========================================================
    # 十、SAC 参数
    # =========================================================
    parser.add_argument("--lr_alpha", type=float, default=3e-4, help="SAC 温度参数 alpha 的学习率。")
    parser.add_argument("--init_temperature", type=float, default=0.2, help="SAC 初始温度 alpha。")
    parser.add_argument("--target_entropy", type=float, default=-2.0, help="SAC 目标熵；二维连续动作通常先设为 -action_dim。")
    parser.add_argument("--log_std_min", type=float, default=-5.0, help="策略高斯分布 log_std 的下界。")
    parser.add_argument("--log_std_max", type=float, default=2.0, help="策略高斯分布 log_std 的上界。")

    args = parser.parse_args()

    # =========================================================
    # 十一、自动派生参数
    # =========================================================
    args.num_followers = args.num_robots - 1
    args.nagents = args.num_followers
    args.action_dim = 2
    args.obs_size = 55

    return args

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training starting on device: {device}")

    # ==========================================
    # 1. 训练参数与工程目录设置
    # ==========================================
    train_iterations = args.train_iterations
    base_save_dir = "/home/lpp/formation_test/data" # 你的数据保存目录
    
    # ⚠️ 断点续训设置 
    # 如果想从头训练，保持 None；如果想继续，填入 latest_checkpoint 路径
    resume_checkpoint = None  
    # resume_checkpoint = "/home/nankai/formation_test/data/MADDPG_Formation_0910d9_2026-04-03_17-30-05/best_avg_checkpoint" 

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
        "render": args.render,
        "sensing_radius": args.sensing_radius,
        "map_mode": "custom",
        "custom_map_path": "/home/nankai/formation_test/maps/underground_garage1.pgm",
    }

    # ==========================================
    # 2. 初始化环境、智能体和经验池
    # ==========================================
    env = Formation2DMultiAgentEnv(config)
    agent = MASAC_Agent(args)
    buffer = ReplayBuffer(args.buffer_capacity, args.num_followers, args.obs_size, args.action_dim, args.hid_size, args.intent_dim, args.value_dim ,device)
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
    best_checkpoint_dir = os.path.join(save_root_dir, "best_avg_checkpoint")
    best_avg_score = -1e18

    for episode in range(start_episode, train_iterations):

        # if episode < 2000:
        #     current_noise = 0.15
        # else:
        #     progress = min(1.0, (episode - 2000) / 1000)
        #     current_noise = 0.10 - progress * 0.09
        #     current_noise = max(current_noise, 0.01)

        # [新增] 课程训练：每局先选地图
        map_mode = choose_map_mode(episode)
        env.set_map_mode(map_mode)

        obs_dict, _ = env.reset()
        obs_array = np.array([obs_dict[agent_id] for agent_id in env._agent_ids], dtype=np.float32)

        h_in, c_in = agent.init_hidden()
        agent.reset_runtime()

        episode_reward = 0.0
        actor_losses = []
        critic_losses = []

        for step in range(args.max_steps):
            action_policy, action_exec, graphs_array, graphs_soft_array, h_out, c_out, comm_snapshot = agent.select_action(
                obs_array,
                h_in,
                c_in,
                deterministic=False,
            )

            action_dict = {env._agent_ids[i]: action_exec[i] for i in range(args.num_followers)}
            graph_dict = {env._agent_ids[i]: graphs_array[i] for i in range(args.num_followers)}
            graphs_soft_dict = {env._agent_ids[i]: graphs_soft_array[i] for i in range(args.num_followers)}

            next_obs_dict, reward_dict, terminated_dict, truncated_dict, _ = env.step(
                action_dict,
                graph_dict,
                graphs_soft_dict,
            )
            next_obs_array = np.array([next_obs_dict[agent_id] for agent_id in env._agent_ids], dtype=np.float32)

            team_reward = reward_dict[env._agent_ids[0]]
            team_done = terminated_dict["__all__"] or truncated_dict["__all__"]

            buffer.store({
                "obs": obs_array,
                "action_exec": action_exec,
                "action_policy": action_policy,
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
                "recv_mask": comm_snapshot["recv_mask"], 
                "time_lags": comm_snapshot["time_lags"],
                "episode_id": episode,
                "step_id": step,
            })

            obs_array = next_obs_array
            h_in, c_in = h_out, c_out
            episode_reward += team_reward

            learning_starts = 2000
            if buffer.size() >= learning_starts:
                metrics = agent.update(buffer)
                actor_losses.append(metrics["actor_loss"])
                critic_losses.append(metrics["critic_loss"])

            if team_done:
                break

        avg_a_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_c_loss = np.mean(critic_losses) if critic_losses else 0.0
        step_avg_reward = episode_reward / (step + 1)

        print(
            f"Episode: {episode:5d} | Map: {map_mode:>5s} | Steps: {step+1:3d} | "
            f"Reward: {step_avg_reward:8.2f} | A_Loss: {avg_a_loss:.4f} | C_Loss: {avg_c_loss:.4f}"
        )

        writer.add_scalar("Training/Episode_Reward", episode_reward, episode)
        writer.add_scalar("Training/Episode_Length", step + 1, episode)
        writer.add_scalar("Loss/Actor_Loss", avg_a_loss, episode)
        writer.add_scalar("Loss/Critic_Loss", avg_c_loss, episode)
        writer.add_scalar("Training/Avg_Step_Reward", step_avg_reward, episode)
        writer.add_scalar("Training/MapMode_Open", 1.0 if map_mode == "open" else 0.0, episode)

        # latest checkpoint
        if (episode + 1) % args.eval_every == 0:
            
            # 定义期末考试科目（你可以把想考的地图都写上）
            eval_maps = ["open", "z_map", "star_map", "custom"]
            scores = {}
            
            print(f"📊 Eval @ episode {episode+1}: ", end="")
            for m in eval_maps:
                score = evaluate_on_map(agent, env, m, args.eval_episodes, args.max_steps)
                scores[m] = score
                writer.add_scalar(f"Eval/{m}_Reward", score, episode)
                print(f"{m}={score:.2f}, ", end="")
                
            # 计算平均分作为保存 best_model 的依据
            score_avg = np.mean(list(scores.values()))
            writer.add_scalar("Eval/AvgReward", score_avg, episode)
            
            print(f"avg={score_avg:.2f}")

            if score_avg > best_avg_score:
                best_avg_score = score_avg
                if os.path.exists(best_checkpoint_dir):
                    shutil.rmtree(best_checkpoint_dir, ignore_errors=True)
                save_checkpoint(agent, episode + 1, best_checkpoint_dir)
                print(f"🏆 [Best Avg Checkpoint Updated] -> {best_checkpoint_dir}")

            # 覆盖保存最新的大脑
            save_checkpoint(agent, episode + 1, fixed_checkpoint_dir)
            # 覆盖保存最新的经验池 (由于文件较大，每 50 局存一次既安全又不拖慢训练)
            buffer.save(fixed_checkpoint_dir)
            print(f"[Latest Checkpoint Saved] 进度已存档 -> {fixed_checkpoint_dir}")

    # 训练彻底结束时保存最终模型
    save_checkpoint(agent, train_iterations, fixed_checkpoint_dir)
    buffer.save(fixed_checkpoint_dir)
    print(f"\n🎉 训练全部结束！最终模型保存在: {fixed_checkpoint_dir}")
    writer.close()

if __name__ == "__main__":
    main()
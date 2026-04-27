import os
import time
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from envs.formation_2d_env import Formation2DMultiAgentEnv
from agents.maddpg_agent import MADDPG_Agent


# ==========================================
# 0. 参数解析（必须与训练时网络结构保持一致）
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="MADDPG + MAGIC Scheduler + CoDe Evaluation with trajectory visualization"
    )

    # =========================================================
    # 评估专属参数
    # =========================================================
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/nankai/formation_test/data/MADDPG_Formation_660865_2026-04-26_00-18-35/latest_checkpoint",
        help="要评估的模型文件夹路径（包含 maddpg_checkpoint.pt 的目录）。",
    )
    parser.add_argument("--eval_episodes", type=int, default=3, help="评估多少局。")
    parser.add_argument(
        "--map_mode",
        type=str,
        default="star_map",
        choices=["open", "z_map", "star_map", "hybrid", "custom"],
        help="评估地图模式。",
    )
    parser.add_argument(
        "--custom_map_path",
        type=str,
        default="/home/nankai/formation_test/maps/underground_garage5.pgm",
        help="真实地图路径。",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="是否实时渲染环境窗口。临时观察行为时打开；批量导图时可关闭。",
    )
    parser.add_argument(
        "--save_fig",
        action="store_true",
        default=True,
        help="是否保存论文用轨迹图。",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/nankai/formation_test/eval_outputs",
        help="评估结果与轨迹图保存目录。",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        default=True,
        help="是否把轨迹数据保存成 npz，便于后处理。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.01,
        help="render 打开时每步暂停秒数；批量导图时可设为 0。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="评估基础随机种子。每个 episode 会在此基础上递增。",
    )

    # =========================================================
    # 环境参数
    # =========================================================
    parser.add_argument("--num_robots", type=int, default=3, help="总机器人数量，包含 1 个 leader 和 N-1 个 follower。")
    parser.add_argument("--max_steps", type=int, default=1000, help="每个评估 episode 的最大步数。")
    parser.add_argument("--sensing_radius", type=float, default=5.0, help="每个 follower 的局部感知半径。")

    # =========================================================
    # MADDPG 基础参数（评估时主要用于初始化 Agent）
    # =========================================================
    parser.add_argument("--lr_actor", type=float, default=5e-5)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)

    # =========================================================
    # MAGIC 调度器参数
    # =========================================================
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

    parser.add_argument("--learn_second_graph", dest="learn_second_graph", action="store_true")
    parser.add_argument("--no_learn_second_graph", dest="learn_second_graph", action="store_false")
    parser.set_defaults(learn_second_graph=True)

    parser.add_argument("--second_graph_complete", action="store_true", default=False)

    parser.add_argument("--directed", dest="directed", action="store_true")
    parser.add_argument("--undirected", dest="directed", action="store_false")
    parser.set_defaults(directed=True)

    parser.add_argument("--comm_mask_zero", action="store_true", default=True)

    parser.add_argument("--gat_encoder_out_size", type=int, default=64)
    parser.add_argument("--ge_num_heads", type=int, default=4)
    parser.add_argument("--gat_encoder_normalize", action="store_true", default=False)

    parser.add_argument("--message_encoder", action="store_true", default=False)
    parser.add_argument("--message_decoder", action="store_true", default=False)
    parser.add_argument("--comm_init", type=str, default="zeros")

    # =========================================================
    # CoDe 参数
    # =========================================================
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
    parser.add_argument("--pred_horizon", type=int, default=5)

    # =========================================================
    # 延迟设置
    # =========================================================
    parser.add_argument("--delay_mode", type=str, default="fixed", choices=["none", "fixed", "uniform"], help="通信延迟模式。")
    parser.add_argument("--fixed_delay", type=int, default=2, help="固定延迟模式下的延迟步数。")
    parser.add_argument("--min_delay", type=int, default=1, help="随机延迟模式下的最小延迟。")
    parser.add_argument("--max_delay", type=int, default=3, help="随机延迟模式下的最大延迟。")

    args = parser.parse_args()

    # 派生参数
    args.num_followers = args.num_robots - 1
    args.nagents = args.num_followers
    args.action_dim = 2
    args.obs_size = 58

    return args


# ==========================================
# 1. 只加载模型权重
# ==========================================
def load_evaluate_model(agent, path: str) -> int:
    checkpoint_file = os.path.join(path, "maddpg_checkpoint.pt")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"找不到模型文件，请检查路径: {checkpoint_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_file, map_location=device)

    agent.actor.load_state_dict(checkpoint["actor"])
    if "target_actor" in checkpoint:
        agent.target_actor.load_state_dict(checkpoint["target_actor"])
    else:
        agent.target_actor.load_state_dict(checkpoint["actor"])

    episode = int(checkpoint.get("episode", -1))
    print(f"✅ 成功加载第 {episode} 轮的模型。")
    return episode


# ==========================================
# 2. 轨迹记录与可视化
# ==========================================
def _record_positions(env, traj: Dict[str, List[np.ndarray]]) -> None:
    traj["leader"].append(np.array(env.leader_pos, dtype=np.float32).copy())
    for i, pos in enumerate(env.follower_pos):
        traj[f"follower_{i}"] .append(np.array(pos, dtype=np.float32).copy())


def _map_title(map_mode: str) -> str:
    mapping = {
        "open": "Open Map",
        "star_map": "Pillar Hall Map",
        "hybrid": "Hybrid Indoor Map",
        "z_map": "Z-Corridor Map",
        "custom": "Custom Map",
    }
    return mapping.get(map_mode, map_mode)


def _plot_single_trajectory(env, traj: Dict[str, List[np.ndarray]], save_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    height, width = env.map_grid.shape
    res = env.map_resolution  # 你的分辨率是 0.1 米/像素
    
    # 你的环境是把原点 (0,0) 放在地图中心的，所以计算出物理边界：
    extent = [
        -width / 2 * res,  # X 轴下界
         width / 2 * res,  # X 轴上界
        -height / 2 * res, # Y 轴下界
         height / 2 * res  # Y 轴上界
    ]

    # 背景障碍物
    grid = np.array(env.map_grid, dtype=np.uint8)
    ax.imshow(grid, cmap="gray_r", origin="lower", extent=extent)

    # 领导者轨迹
    leader = np.array(traj["leader"])
    if len(leader) > 0:
        ax.plot(leader[:, 0], leader[:, 1], linewidth=2.5, label="Leader")
        ax.scatter(leader[0, 0], leader[0, 1], s=60, marker="o")
        ax.scatter(leader[-1, 0], leader[-1, 1], s=80, marker="*")

    # 跟随者轨迹
    follower_keys = sorted([k for k in traj.keys() if k.startswith("follower_")])
    for idx, key in enumerate(follower_keys):
        arr = np.array(traj[key])
        if len(arr) == 0:
            continue
        ax.plot(arr[:, 0], arr[:, 1], linewidth=2.0, label=f"Follower {idx+1}")
        ax.scatter(arr[0, 0], arr[0, 1], s=40, marker="o")
        ax.scatter(arr[-1, 0], arr[-1, 1], s=55, marker="x")

    ax.set_title(title)
    ax.set_xlabel("Map X (meters)")
    ax.set_ylabel("Map Y (meters)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


# ==========================================
# 3. 单局 rollout + 指标
# ==========================================
def run_one_episode(agent, env, args, episode_idx: int) -> Tuple[float, int, Dict, Dict[str, List[np.ndarray]]]:
    obs_dict, _ = env.reset(seed=args.seed + episode_idx)
    obs_array = np.array([obs_dict[agent_id] for agent_id in env._agent_ids], dtype=np.float32)

    h_in, c_in = agent.init_hidden()
    agent.reset_runtime()

    episode_reward = 0.0
    last_info = None

    traj = {"leader": []}
    for i in range(args.num_followers):
        traj[f"follower_{i}"] = []
    _record_positions(env, traj)

    # 初始化一个“平滑动作”变量
    smoothed_action = np.zeros((args.num_followers, args.action_dim), dtype=np.float32)
    # 平滑系数 tau_action (0 到 1 之间)。越接近 1 越平滑 (惯性越大)，越接近 0 越相信网络原始输出。
    # 0.8 是一个非常经典的经验值
    tau_action = 0.8 

    for step in range(args.max_steps):
        action_policy, action_exec, graphs_array, graphs_soft_array, h_out, c_out, comm_snapshot = agent.select_action(
            obs_array,
            h_in,
            c_in,
            add_noise=False,
        )

        smoothed_action = tau_action * smoothed_action + (1.0 - tau_action) * action_exec

        action_dict = {env._agent_ids[i]: smoothed_action[i] for i in range(args.num_followers)}
        graph_dict = {env._agent_ids[i]: graphs_array[i] for i in range(args.num_followers)}
        graph_soft_dict = {env._agent_ids[i]: graphs_soft_array[i] for i in range(args.num_followers)}

        next_obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(
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
        last_info = info_dict[env._agent_ids[0]]

        _record_positions(env, traj)

        if args.render and args.sleep > 0:
            time.sleep(args.sleep)

        if team_done:
            break

    return episode_reward, step + 1, last_info or {}, traj


# ==========================================
# 4. 主评估流程
# ==========================================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估启动，运行设备: {device}")
    print(f"即将测试的地图模式: {args.map_mode}")

    config = {
        "num_robots": args.num_robots,
        "max_steps": args.max_steps,
        "render": args.render,
        "sensing_radius": args.sensing_radius,
        "map_mode": args.map_mode,
        "custom_map_path": args.custom_map_path,
    }

    env = Formation2DMultiAgentEnv(config)
    agent = MADDPG_Agent(args)
    loaded_episode = load_evaluate_model(agent, args.model_dir)

    os.makedirs(args.save_dir, exist_ok=True)

    rewards = []
    lengths = []
    successes = []
    collisions = []
    formation_errors = []

    best_reward = -1e18
    best_record = None

    for episode in range(args.eval_episodes):
        episode_reward, ep_len, info, traj = run_one_episode(agent, env, args, episode)

        success = float(info.get("success", 0.0))
        collision = float(info.get("collision", 0.0))
        formation_error = float(info.get("formation_error", np.nan))

        rewards.append(episode_reward)
        lengths.append(ep_len)
        successes.append(success)
        collisions.append(collision)
        formation_errors.append(formation_error)

        print(
            f"评估 {episode + 1}/{args.eval_episodes} | 步数: {ep_len:3d} | "
            f"总得分: {episode_reward:8.2f} | 成功: {int(success)} | 碰撞: {int(collision)} | 编队误差: {formation_error:.3f}"
        )

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_record = {
                "episode_idx": episode,
                "traj": traj,
                "info": info,
            }

        if args.save_npz:
            npz_path = os.path.join(
                args.save_dir,
                f"rollout_{args.map_mode}_ep{episode+1:02d}.npz",
            )
            payload = {k: np.array(v, dtype=np.float32) for k, v in traj.items()}
            payload["episode_reward"] = np.array([episode_reward], dtype=np.float32)
            payload["episode_len"] = np.array([ep_len], dtype=np.int32)
            payload["success"] = np.array([success], dtype=np.float32)
            payload["collision"] = np.array([collision], dtype=np.float32)
            payload["formation_error"] = np.array([formation_error], dtype=np.float32)
            np.savez(npz_path, **payload)

        if args.save_fig:
            fig_name = f"traj_{args.map_mode}_ep{episode+1:02d}.png"
            fig_path = os.path.join(args.save_dir, fig_name)
            title = f"{_map_title(args.map_mode)} | ep {episode+1} | model ep {loaded_episode}"
            _plot_single_trajectory(env, traj, fig_path, title)
            print(f"🖼️ 第 {episode+1} 局轨迹图已保存: {fig_path}")

    print("\n================ 评估汇总 ================")
    print(f"Map               : {_map_title(args.map_mode)}")
    print(f"Model Episode     : {loaded_episode}")
    print(f"Avg Reward        : {np.mean(rewards):.3f}")
    print(f"Success Rate      : {np.mean(successes):.3f}")
    print(f"Collision Rate    : {np.mean(collisions):.3f}")
    print(f"Avg Episode Len   : {np.mean(lengths):.3f}")
    print(f"Avg Formation Err : {np.nanmean(formation_errors):.3f}")

    # if args.save_fig and best_record is not None:
    #     fig_name = f"traj_{args.map_mode}_best.png"
    #     fig_path = os.path.join(args.save_dir, fig_name)
    #     title = f"{_map_title(args.map_mode)} | best rollout | model ep {loaded_episode}"
    #     _plot_single_trajectory(env, best_record["traj"], fig_path, title)
    #     print(f"🖼️ 轨迹图已保存: {fig_path}")

    print("\n评估结束。")


if __name__ == "__main__":
    main()
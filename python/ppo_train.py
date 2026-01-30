from email import parser
import os
import ray
import numpy as np
import torch


from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from envs.formation_pybullet_env import FormationPyBulletEnv
from models.formation_net_rllib import ConstrainedFormationNetRLLib

def main():

    num_robots = 3


    # 初始化C++模块（需要先编译）
    import formation_core

    cpp_enumerator = formation_core.FormationEnumerator(num_robots)
    all_formations = cpp_enumerator.get_all_formations()
    adj_np_list = []
    for cg in all_formations:
        # 获取单个控制图的邻接矩阵（numpy.ndarray，shape=(N,N)）
        adj_np = cg.get_adjacency_matrix()
        adj_np_list.append(adj_np)

    control_graphs_np = np.stack(adj_np_list, axis=0) # shape=(num_graphs, N, N)
    num_graphs = control_graphs_np.shape[0]  # 控制图数量


    # 初始化Ray
    ray.init()
    
    # 创建环境配置
    env_config = {
        "num_robots": num_robots,
        "max_steps": 100,
        "safety_threshold": 0.5,  # 最小安全距离(避碰距离)
        "max_comm_distance": 5.0  # 最大通信距离
    }
    
    # 配置PPO
    config = (
        PPOConfig()
        .environment(FormationPyBulletEnv, env_config=env_config)
        .training(
            gamma=0.99,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            lr=3e-4,
            train_batch_size=1000,
            mini_batch_size=128
        )
        .model(
            custom_model=ConstrainedFormationNetRLLib,
            custom_model_config={
                "feature_dim": 21,
                "num_graphs": num_graphs,
                "max_robots": 10,
                "control_graphs": control_graphs_np
            }
        )
        .rollouts(
            num_rollout_workers=4,
            rollout_fragment_length=100,
            batch_mode="truncate_episodes"
        )
        .resources(
            num_gpus=0.5  # 如果有GPU可用
        )
    )
    
    # 创建训练器
    trainer = config.build()
    
    # 训练
    print(f"Starting PPO training for {1000} episodes")
    for i in range(1000):
        result = trainer.train()
        
        # 打印进度
        if i % 10 == 0:
            print(f"Episode {i}: "
                  f"Reward={result['episode_reward_mean']:.2f} | "
                  f"Length={result['episode_len_mean']:.2f} | "
                  f"Loss={result['total_loss']:.4f}")
        
        # 保存检查点
        if i % 50 == 0:
            checkpoint_path = trainer.save()
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # 保存最终模型
    final_path = os.path.join("outputs", "final_model.pth")
    trainer.save(final_path)
    print(f"Final model saved to {final_path}")
    
    # 关闭Ray
    ray.shutdown()

if __name__ == "__main__":
    main()
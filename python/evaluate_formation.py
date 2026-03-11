import os
import ray
import time
import numpy as np

from ray.tune.registry import register_env
from envs.formation_2d_env import Formation2DEnv
from models.formation_net_rllib import ConstrainedFormationNetRLLib
from ray.rllib.algorithms.ppo import PPO

# 注册环境
register_env("Formation2DEnv-v0", lambda config: Formation2DEnv(config))

def main():
    max_robots = 10  
    num_robots = 3
    
    # ==========================================
    # 🔴 关键步骤：填入你刚刚训练保存的 Checkpoint 路径！
    # 比如: ""
    # ==========================================
    checkpoint_path = "/home/nankai/formation_test/data/PPO_FormationPyBulletEnv-v0_16355e_2026-03-10_15-56-13/latest_checkpoint" 

    # 模拟候选图 (必须和训练时完全一致)
    try:
        import formation_core
        cpp_enumerator = formation_core.FormationEnumerator(num_robots)
        all_formations = cpp_enumerator.get_all_formations()
        adj_np_list = []
        for cg in all_formations:
            adj_np = cg.get_adjacency_matrix()
            adj_np_list.append(adj_np)
        control_graphs_np = np.stack(adj_np_list, axis=0)
    except ImportError:
        print("Warning: formation_core module not found, using manual graphs")
        g1 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
        g2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        g3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)
        control_graphs_np = np.stack([g1, g2, g3], axis=0)
    
    ray.init(ignore_reinit_error=True, num_cpus=1, num_gpus=0)

    # ✅ 评估时的环境配置 (关键：开启 render)
    env_config = {
        "num_robots": num_robots,
        "max_robots": max_robots,
        "max_steps": 150,  # 评估时可以给多一点步数，让它走完路径
        "safety_threshold": 0.5,
        "candidate_graphs": control_graphs_np,
        "render": True  # <--- 开启画面渲染！
    }

    # 保持和训练时完全一致的架构配置
    config = {
        "env": "Formation2DEnv-v0",
        "env_config": env_config,
        "framework": "torch",
        "model": {
            "custom_model": ConstrainedFormationNetRLLib,
            "custom_model_config": {
                "feature_dim": 21,
                "control_graphs": control_graphs_np,
                "num_robots": num_robots,
                "max_robots": max_robots
            }
        },
        "num_workers": 0,
    }

    print("\n正在初始化模型并加载权重...")
    trainer = PPO(config=config)
    
    try:
        trainer.restore(checkpoint_path)
        print(f"✅ 成功加载模型权重: {checkpoint_path}\n")
    except Exception as e:
        print(f"❌ 加载模型失败，请检查路径是否正确！\n错误信息: {e}")
        return 

    # 创建独立的测试环境
    env = Formation2DEnv(env_config)
    
    test_episodes = 5  # 测试的局数
    
    for episode in range(test_episodes):
        # 注意：Gymnasium 的 reset 返回 (obs, info)
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        
        print(f"▶️ 开始第 {episode + 1}/{test_episodes} 局测试...")
        
        while not done:
            # 1. 大脑根据观测算出最优动作
            action = trainer.compute_single_action(obs)
            
            # 2. 机器人在环境中执行动作
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # 3. 稍微加点延时，不然画面闪得太快看不清
            # 如果你觉得画面太快，把这里改大一点 (比如 0.1)；如果觉得慢，改成 0.01
            time.sleep(0.05) 
            
        print(f"⏹️ 第 {episode + 1} 局结束，总奖励: {total_reward:.2f}")

    # 清理并退出
    env.close()
    ray.shutdown()

if __name__ == "__main__":
    main()
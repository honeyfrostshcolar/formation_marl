import ray
import numpy as np
import pybullet as p
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from gymnasium.wrappers import EnvCompatibility

from envs.formation_pybullet_env import FormationPyBulletEnv
from models.formation_net_rllib import ConstrainedFormationNetRLLib
from policies.formation_policy import CustomFormationPolicy

# 注册环境（与训练时一致）
def make_formation_env(cfg):
    return EnvCompatibility(FormationPyBulletEnv(cfg))

register_env("FormationPyBulletEnv-v0", make_formation_env)

def main():
    # 初始化 Ray
    ray.init(ignore_reinit_error=True)

    checkpoint_path = "/home/lpp/formation_test/data/PPO_FormationPyBulletEnv-v0_0abc9e_2026-02-28_16-01-30"  
    trainer = PPO.from_checkpoint(checkpoint_path)

    num_robots = 3

    # 初始化C++模块获取控制图（如果需要与训练时相同的图数据）
    try:
        import formation_core
        cpp_enumerator = formation_core.FormationEnumerator(num_robots)
        all_formations = cpp_enumerator.get_all_formations()
        adj_np_list = []
        for cg in all_formations:
            adj_np = cg.get_adjacency_matrix()
            adj_np_list.append(adj_np)
        control_graphs_np = np.stack(adj_np_list, axis=0)
        num_graphs = control_graphs_np.shape[0]
    except ImportError:
        print("Warning: formation_core module not found, using dummy graphs")
        control_graphs_np = np.eye(num_robots, dtype=np.float32)
        control_graphs_np = np.expand_dims(control_graphs_np, axis=0)
        num_graphs = 1

    # 创建环境（GUI 模式）
    env_config = {
        "num_robots": num_robots,
        "max_steps": 100,
        "safety_threshold": 0.5,
        "max_comm_distance": 5.0,
        "candidate_graphs": control_graphs_np,
        "render": True  # 直接传递给环境
    }
    env = FormationPyBulletEnv(env_config)  # 注意：我们不再用 EnvCompatibility 包装，因为环境本身可能已兼容

    p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,0])
    p.stepSimulation()
    import time
    time.sleep(2)  # 等待 GUI 刷新
    input("Press Enter to start episodes...") 

    # 运行几个 episode
    for episode in range(50):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = trainer.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # 可以添加延时，以便观察
            # import time; time.sleep(0.01)
        print(f"Episode {episode} reward: {total_reward:.2f}")

    env.close()
    ray.shutdown()

if __name__ == "__main__":
    main()
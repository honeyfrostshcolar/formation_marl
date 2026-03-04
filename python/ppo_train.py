import os
import ray
import time
import numpy as np
import torch
import uuid 

# 清空PYTHONPATH（避免导入冲突）
if "PYTHONPATH" in os.environ:
    del os.environ["PYTHONPATH"]

# 导入自定义模块
from envs.formation_pybullet_env import FormationPyBulletEnv
from models.formation_net_rllib import ConstrainedFormationNetRLLib
from policies.formation_policy import CustomFormationPolicy

# 新增：导入gymnasium spaces（用于定义观测/动作空间）
from gymnasium.wrappers import EnvCompatibility
from ray.tune.registry import register_env
from gymnasium import spaces

# 定义环境创建函数并注册
def make_formation_env(cfg):
    return EnvCompatibility(FormationPyBulletEnv(cfg))
register_env("FormationPyBulletEnv-v0", make_formation_env)

def main():
    num_robots = 3
    train_iterations = 200

    resume_checkpoint = "/home/lpp/formation_test/data/PPO_FormationPyBulletEnv-v0_bf0000_2026-03-03_22-11-43"
    # resume_checkpoint = None  # 从头训练时设为None

    base_save_dir = "/home/lpp/formation_test/data"
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # 生成随机后缀（避免同一时间多次训练冲突）
    random_suffix = uuid.uuid4().hex[:6]  # 取6位随机字符串
    task_dir = f"PPO_FormationPyBulletEnv-v0_{random_suffix}_{timestamp}"
    # 完整的保存根目录
    save_root_dir = os.path.join(base_save_dir, task_dir)
    os.makedirs(save_root_dir, exist_ok=True)
    print(f"本次训练的根目录：{save_root_dir}")

    # 初始化C++模块获取控制图
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

    # 初始化Ray
    ray.init(ignore_reinit_error=True, num_cpus=1, num_gpus=0)

    # 环境配置
    env_config = {
        "num_robots": num_robots,
        "max_steps": 100,
        "safety_threshold": 0.5,
        "max_comm_distance": 5.0,
        "candidate_graphs": control_graphs_np
    }

    # ✅ 关键新增：手动定义观测空间和动作空间（和环境里的定义保持一致）
    num_followers = num_robots - 1
    # 观测空间：21维连续空间（和FormationPyBulletEnv里的定义一致）
    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
    )
    # 动作空间：1 + 2*num_followers 维连续空间（和环境里的定义一致）
    action_space = spaces.Box(
        low=-1.0, high=1.0, 
        shape=(1 + 2 * num_followers,), 
        dtype=np.float32
    )

    # 使用字典配置（替代链式API）
    from ray.rllib.algorithms.ppo import PPO
    config = {} 
    config.update({
        "env": "FormationPyBulletEnv-v0",
        "env_config": env_config,
        "disable_env_checking": True,  # 禁用环境检查
        "gamma": 0.99,
        "clip_param": 0.2,
        "entropy_coeff": 0.01,
        "vf_loss_coeff": 0.5,
        "lr": 3e-4,
        "train_batch_size": 1000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "model": {
            "custom_model": ConstrainedFormationNetRLLib,
            "custom_model_config": {
                "feature_dim": 21,
                "num_graphs": num_graphs,
                "max_robots": 10,
                "control_graphs": control_graphs_np,
                "num_robots": num_robots
            }
        },
        "framework": "torch",
        "num_workers": 0,
        "rollout_fragment_length": 200,
        "batch_mode": "truncate_episodes",
        "num_gpus": 0,
        "num_cpus_per_worker": 1,
        "num_gpus_per_worker": 0,
        "evaluation_num_workers": 0,
        "evaluation_interval": 1,
        "evaluation_duration": 5,

        # ✅ 关键修改：给策略手动指定observation_space和action_space
        "multiagent": {
            "policies": {
                "default_policy": (
                    CustomFormationPolicy,       # 策略类
                    observation_space,           # 观测空间（手动指定）
                    action_space,                # 动作空间（手动指定）
                    {}                           # 策略配置（空即可）
                )
            },
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: "default_policy"
        },

        "logger_config": {
        "type": "ray.tune.logger.TBXLogger",  # 启用TensorBoard日志
        "logdir": save_root_dir,  # 日志保存到自定义的训练目录
        }
    })

    # 创建训练器
    trainer = PPO(config=config)

    if resume_checkpoint is not None:
        if os.path.exists(resume_checkpoint):
            print(f"从检查点恢复训练：{resume_checkpoint}")
            trainer.restore(resume_checkpoint)
            start_iter = trainer.iteration  # 获取训练器当前的迭代数
            print(f"恢复后的当前迭代：{start_iter}")
        else:
            print(f"检查点路径不存在：{resume_checkpoint}，将从头开始训练")
            start_iter = 0
    else:
        print("从头开始训练")
        start_iter = 0

    
    # 训练循环：从start_iter开始，不是从0开始
    print(f"Starting PPO training from iteration {start_iter} to {train_iterations}")
    for i in range(start_iter, train_iterations):
        result = trainer.train()
        print(f"Iteration {i}:")
        print(f"  Episode reward mean: {result.get('episode_reward_mean', 0.0):.2f}")
        print(f"  Episode length mean: {result.get('episode_len_mean', 0.0):.2f}")
        print(f"  Total loss: {result.get('total_loss', 0.0):.4f}")
        
        # 可选：每10个迭代保存一次检查点
        if i % 10 == 0:
            checkpoint_path = trainer.save(save_root_dir)
            print(f"  Checkpoint saved to {checkpoint_path}")

    # 保存最终检查点
    checkpoint_path = trainer.save(save_root_dir)
    print(f"Final checkpoint saved to {checkpoint_path}")

    ray.shutdown()

if __name__ == "__main__":
    main()

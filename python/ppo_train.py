import os
import ray
import time
import numpy as np
import uuid 
import logging
import shutil  # 用于覆盖旧检查点文件夹

from ray.tune.registry import register_env
from envs.formation_2d_env import Formation2DEnv
from models.formation_net_rllib import ConstrainedFormationNetRLLib
from ray.rllib.algorithms.ppo import PPO

# 清空PYTHONPATH（避免导入冲突）
if "PYTHONPATH" in os.environ:
    del os.environ["PYTHONPATH"]

# 注册环境
register_env("Formation2DEnv-v0", lambda config: Formation2DEnv(config))

def main():
    max_robots = 10  
    num_robots = 3
    train_iterations = 1000  # 训练总轮数

    base_save_dir = "/home/lpp/formation_test/data"
    
    # ==========================================
    # ✅ 断点续训设置 (Resume Training)
    # ==========================================
    # 如果你想从头训练，保持 None
    resume_checkpoint = None  
    
    # 如果你想继续训练，把下面这行的注释打开，并填入你上一次保存的 latest_checkpoint 路径：
    # resume_checkpoint = "/home/lpp/formation_test/data/PPO_FormationPyBulletEnv-v0_xxxxxx_xxxx/latest_checkpoint"
    
    # ==========================================

    # 生成本次运行专属的文件夹名字
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    random_suffix = uuid.uuid4().hex[:6]
    task_dir = f"PPO_FormationPyBulletEnv-v0_{random_suffix}_{timestamp}"
    save_root_dir = os.path.join(base_save_dir, task_dir)
    os.makedirs(save_root_dir, exist_ok=True)
    
    # 我们在这个任务文件夹下，固定用 "latest_checkpoint" 来保存最新模型
    fixed_checkpoint_dir = os.path.join(save_root_dir, "latest_checkpoint")
    print(f"本次训练的根目录：{save_root_dir}")

    # 模拟候选图
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
    
    num_graphs = control_graphs_np.shape[0]

    # 关闭RLlib的非错误日志，保持控制台清爽
    logging.getLogger("ray.rllib").setLevel(logging.ERROR)
    logging.getLogger("ray.tune").setLevel(logging.ERROR)

    ray.init(ignore_reinit_error=True, num_cpus=1, num_gpus=0)

    env_config = {
        "num_robots": num_robots,
        "max_robots": max_robots,
        "max_steps": 100,
        "safety_threshold": 0.5,
        "candidate_graphs": control_graphs_np,
        "render": True  # ⚠️ 训练时必须关闭渲染以保证速度！
    }

    config = {
        "env": "Formation2DEnv-v0",
        "env_config": env_config,
        "framework": "torch",
        
        # ✅ 我们已经加入了路径规划，这里必须恢复马尔可夫链的长期折现参数
        "gamma": 0.99,
        "use_gae": True,
        "lambda": 0.95,
        "batch_mode": "truncate_episodes", 
        
        "lr": 1e-4,
        "train_batch_size": 1000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "clip_param": 0.2,
        
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
        "logger_config": {
            "type": "ray.tune.logger.TBXLogger",
            "logdir": save_root_dir,
        }
    }

    # 创建训练器
    trainer = PPO(config=config)

    # ==========================================
    # ✅ 执行断点续训加载逻辑
    # ==========================================
    if resume_checkpoint is not None:
        if os.path.exists(resume_checkpoint):
            print(f"\n[恢复训练] 正在加载检查点：{resume_checkpoint}")
            trainer.restore(resume_checkpoint)
            start_iter = trainer.iteration  # 获取之前已经训练的迭代数
            print(f"[恢复成功] 将从第 {start_iter} 轮继续训练！\n")
        else:
            print(f"\n[警告] 检查点路径不存在：{resume_checkpoint}，将从头开始训练！\n")
            start_iter = 0
    else:
        print("\n[全新训练] 从头开始训练...\n")
        start_iter = 0

    # ==========================================
    # ✅ 训练主循环
    # ==========================================
    print(f"目标：训练至第 {train_iterations} 轮")
    for i in range(start_iter, train_iterations):
        result = trainer.train()

        print(f"Iteration {i}:")
        print(f"  Episode reward mean: {result.get('episode_reward_mean', 0.0):.2f}")
        print(f"  Episode length mean: {result.get('episode_len_mean', 0.0):.2f}")
        
        # 提取 Loss 数据 (适配较新版本的 Ray 字典结构)
        learner_stats = result.get('info', {}).get('learner', {})
        # 根据 Ray 版本不同，可能在 default_policy 下，也可能直接在 learner 里
        if 'default_policy' in learner_stats:
            stats = learner_stats['default_policy'].get('learner_stats', {})
        else:
            stats = learner_stats.get('learner_stats', {})

        real_total_loss = stats.get('total_loss', 0.0)
        real_policy_loss = stats.get('policy_loss', 0.0)
        real_vf_loss = stats.get('vf_loss', 0.0)
        
        print(f"  Total loss: {real_total_loss:.4f} | Policy loss: {real_policy_loss:.4f} | Value loss: {real_vf_loss:.4f}")
        
        # ✅ 每10个迭代保存一次，并覆盖同一个文件夹
        if (i + 1) % 10 == 0:
            if os.path.exists(fixed_checkpoint_dir):
                shutil.rmtree(fixed_checkpoint_dir, ignore_errors=True)
            trainer.save(fixed_checkpoint_dir)
            print(f"  [Checkpoint 已更新] 最新模型在此处 -> {fixed_checkpoint_dir}")

    # 保存最终检查点
    if os.path.exists(fixed_checkpoint_dir):
        shutil.rmtree(fixed_checkpoint_dir, ignore_errors=True)
    trainer.save(fixed_checkpoint_dir)
    print(f"\n🎉 训练全部结束！最终模型保存在: {fixed_checkpoint_dir}")

    ray.shutdown()

if __name__ == "__main__":
    main()
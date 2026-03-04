import ray
from ray.rllib.algorithms.ppo.ppo import PPOConfig

# 强制清空所有可能的路径污染
import os
for k in ["PYTHONPATH", "RAY_PATH"]:
    if k in os.environ:
        del os.environ[k]

# 初始化Ray
ray.init(ignore_reinit_error=True, num_cpus=1, num_gpus=0)

# 极简配置：只用默认环境，无自定义
config = PPOConfig()
config = config.environment("CartPole-v1")
config = config.training(lr=3e-4, sgd_minibatch_size=128)
config = config.framework("torch")
config = config.rollouts(num_rollout_workers=0)  # 单进程，减少冲突

# 创建训练器
trainer = config.build()

# 跑1轮训练
result = trainer.train()
print("极简版训练成功！奖励均值：", result["episode_reward_mean"])

ray.shutdown()

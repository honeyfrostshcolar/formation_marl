import time
import numpy as np
from envs.formation_2d_env import Formation2DEnv

# 创建简单的候选图（单位矩阵）
candidate_graphs = [np.eye(3, dtype=np.float32)]  # 形状 (3,3)

env = Formation2DEnv({
    "render": True,
    "candidate_graphs": candidate_graphs,
    "num_robots": 3
})
obs, info = env.reset()
for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    time.sleep(0.05)
    if done:
        obs, info = env.reset()
input("Press Enter to exit")
env.close()
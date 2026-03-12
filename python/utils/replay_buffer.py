# 文件路径: utils/replay_buffer.py
import numpy as np
import torch

class ReplayBuffer:
    """
    专为多智能体连续控制 (MADDPG) 定制的经验回放池
    极致精简版，去除了所有离散动作和RNN序列的冗余操作
    """
    def __init__(self, capacity, num_followers, obs_dim, action_dim, device):
        self.capacity = int(capacity)
        self.num_followers = num_followers
        self.device = device
        
        # 预先分配内存，极大地提升运行速度 (千万不要用 Python 的 list 去 append)
        # 状态和动作的维度都是 (容量, 小弟数量, 维度)
        self.obs_buffer = np.zeros((self.capacity, num_followers, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.capacity, num_followers, action_dim), dtype=np.float32)
        self.next_obs_buffer = np.zeros((self.capacity, num_followers, obs_dim), dtype=np.float32)
        
        # 奖励和结束标志：
        # 因为我们只有一个上帝视角的 Critic，评价的是整个团队的表现，
        # 所以团队奖励和结束标志只需要一维即可 (容量, 1)
        self.reward_buffer = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done_buffer = np.zeros((self.capacity, 1), dtype=np.float32)
        
        # 内存指针和当前大小记录
        self.ptr = 0
        self.size_tracker = 0

    def store(self, obs, action, reward, next_obs, done):
        """
        存入一步经验 (transition)
        obs, action, next_obs 是 numpy 数组，形状应为 (num_followers, dim)
        reward, done 是单个浮点数 (团队总奖励，团队是否全剧终)
        """
        self.obs_buffer[self.ptr] = obs
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.next_obs_buffer[self.ptr] = next_obs
        self.done_buffer[self.ptr] = done
        
        # 环形缓冲区逻辑：满了就从头开始覆盖最老的数据
        self.ptr = (self.ptr + 1) % self.capacity
        self.size_tracker = min(self.size_tracker + 1, self.capacity)

    def sample(self, batch_size):
        """
        随机抽样一批经验给算法训练
        直接在函数内部转换为 PyTorch 的 Tensor 并推送到 GPU
        """
        # 随机生成 batch_size 个索引
        idxs = np.random.choice(self.size_tracker, batch_size, replace=False)
        
        # 直接切片并转换为 Tensor
        obs_batch = torch.FloatTensor(self.obs_buffer[idxs]).to(self.device)
        action_batch = torch.FloatTensor(self.action_buffer[idxs]).to(self.device)
        reward_batch = torch.FloatTensor(self.reward_buffer[idxs]).to(self.device)
        next_obs_batch = torch.FloatTensor(self.next_obs_buffer[idxs]).to(self.device)
        done_batch = torch.FloatTensor(self.done_buffer[idxs]).to(self.device)
        
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def size(self):
        """返回当前经验池中真实的数据量"""
        return self.size_tracker
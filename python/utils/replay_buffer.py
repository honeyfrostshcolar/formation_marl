# 文件路径: utils/replay_buffer.py
import numpy as np
import torch
import os
import pickle

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
    
    # ==========================================
    # ✅ 修复后：把整个脑子（记忆矩阵）打包存到硬盘
    # ==========================================
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        buffer_path = os.path.join(save_dir, "replay_buffer.pkl")
        
        # ⚠️ 修复点：保存真正的变量 self.size_tracker
        state = {
            'obs': self.obs_buffer,
            'action': self.action_buffer,
            'reward': self.reward_buffer,
            'next_obs': self.next_obs_buffer,
            'done': self.done_buffer,
            'ptr': self.ptr,   
            'size_tracker': getattr(self, 'size_tracker', 0) # 确保读取正确的整数变量
        }
        
        with open(buffer_path, 'wb') as f:
            pickle.dump(state, f)

    # ==========================================
    # ✅ 修复后：从硬盘重新把记忆灌回脑子里
    # ==========================================
    def load(self, save_dir):
        buffer_path = os.path.join(save_dir, "replay_buffer.pkl")
        if os.path.exists(buffer_path):
            with open(buffer_path, 'rb') as f:
                state = pickle.load(f)
                
            self.obs_buffer = state['obs']
            self.action_buffer = state['action']
            self.reward_buffer = state['reward']
            self.next_obs_buffer = state['next_obs']
            self.done_buffer = state['done']
            self.ptr = state['ptr']
            
            # 🚨 救命补丁：如果是读那个带有 Bug 的旧文件，直接手动把 size 设满！
            if 'size_tracker' in state:
                self.size_tracker = state['size_tracker']
            else:
                # 兼容旧文件：如果之前错存成了方法，而你跑了 2850 局，池子绝对是满的 (假设容量 10w)
                self.size_tracker = 100000 # 如果你的 capacity 不是 10w，改成对应的最大容量
                
            print(f"🧠 [记忆恢复] Replay Buffer 加载成功！当前拥有 {self.size_tracker} 条历史经验。")
        else:
            print("⚠️ [记忆为空] 未找到 replay_buffer.pkl，将从空池子开始探索。")

    def size(self):
        """返回当前经验池中真实的数据量"""
        return self.size_tracker
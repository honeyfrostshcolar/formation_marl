# 文件路径: utils/replay_buffer.py
import os
import pickle
from typing import Dict, List

import numpy as np
import torch


class ReplayBuffer:
    """
    专为多智能体连续控制 (MADDPG) 定制的经验回放池
    极致精简版，去除了所有离散动作和RNN序列的冗余操作
    """
    def __init__(self, capacity, num_followers, obs_dim, action_dim, hid_size, intent_dim, value_dim, device):
        self.capacity = int(capacity)
        self.num_followers = num_followers
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hid_size = hid_size
        self.intent_dim = intent_dim
        self.value_dim = value_dim
        self.device = device

        c = self.capacity
        n = num_followers
        self.obs_buffer = np.zeros((c, n, obs_dim), dtype=np.float32)
        
        self.action_exec_buffer = np.zeros((c, n, action_dim), dtype=np.float32)
        self.action_policy_buffer = np.zeros((c, n, action_dim), dtype=np.float32)

        self.reward_buffer = np.zeros((c, 1), dtype=np.float32)
        self.next_obs_buffer = np.zeros((c, n, obs_dim), dtype=np.float32)
        self.done_buffer = np.zeros((c, 1), dtype=np.float32)

        self.h_in_buffer = np.zeros((c, n, hid_size), dtype=np.float32)
        self.c_in_buffer = np.zeros((c, n, hid_size), dtype=np.float32)
        self.h_out_buffer = np.zeros((c, n, hid_size), dtype=np.float32)
        self.c_out_buffer = np.zeros((c, n, hid_size), dtype=np.float32)

        self.prev_action_buffer = np.zeros((c, n, action_dim), dtype=np.float32)
        self.route_hard_buffer = np.zeros((c, n, n), dtype=np.float32)
        self.route_soft_buffer = np.zeros((c, n, n), dtype=np.float32)

        self.sender_intents_recv_buffer = np.zeros((c, n, n, intent_dim), dtype=np.float32)
        self.sender_hidden_recv_buffer = np.zeros((c, n, n, hid_size), dtype=np.float32)
        self.recv_mask_buffer = np.zeros((c, n, n), dtype=np.float32)
        self.time_lags_buffer = np.zeros((c, n, n), dtype=np.float32)

        self.episode_id_buffer = np.full((c,), -1, dtype=np.int64)
        self.step_id_buffer = np.full((c,), -1, dtype=np.int64)

        self.ptr = 0
        self.size_tracker = 0

    def store(self, transition : dict):
        """
        存入一步经验 (transition)
        obs, action, next_obs 是 numpy 数组，形状应为 (num_followers, dim)
        reward, done 是单个浮点数 (团队总奖励，团队是否全剧终)
        """
        i = self.ptr

        self.obs_buffer[i] = transition["obs"]
        self.action_exec_buffer[i] = transition["action_exec"]
        self.action_policy_buffer[i] = transition["action_policy"]

        self.reward_buffer[i] = transition["reward"]
        self.next_obs_buffer[i] = transition["next_obs"]
        self.done_buffer[i] = transition["done"]

        self.h_in_buffer[i] = transition["h_in"]
        self.c_in_buffer[i] = transition["c_in"]
        self.h_out_buffer[i] = transition["h_out"]
        self.c_out_buffer[i] = transition["c_out"]

        self.prev_action_buffer[i] = transition["prev_action"]  # 上一时刻动作向量
        self.route_hard_buffer[i] = transition["route_hard"]  # 硬路由矩阵
        self.route_soft_buffer[i] = transition["route_soft"]  # 软路由矩阵

        self.sender_intents_recv_buffer[i] = transition["sender_intents_recv"]  # 接收者视角下，每个发送者的意图
        self.sender_hidden_recv_buffer[i] = transition["sender_hidden_recv"]  # 接收者看到的发送者隐藏状态
        self.recv_mask_buffer[i] = transition["recv_mask"]  # 接收者视角下，每个接收者实际收到了哪些发送者的消息
        self.time_lags_buffer[i] = transition["time_lags"]  # 每个消息的延迟（当前步减去消息发送步），0表示刚收到，正数表示消息是几秒前发送的。

        self.episode_id_buffer[i] = transition["episode_id"]  # episode id
        self.step_id_buffer[i] = transition["step_id"]  # step id

        # 环形缓冲区逻辑：满了就从头开始覆盖最老的数据
        self.ptr = (self.ptr + 1) % self.capacity
        self.size_tracker = min(self.size_tracker + 1, self.capacity)

    def _to_tensor_batch(self, data: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def sample_transitions(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.choice(self.size_tracker, batch_size, replace=False)
        return {
            "obs": self._to_tensor_batch(self.obs_buffer[idxs]),
            "action": self._to_tensor_batch(self.action_exec_buffer[idxs]),
            "reward": self._to_tensor_batch(self.reward_buffer[idxs]),
            "next_obs": self._to_tensor_batch(self.next_obs_buffer[idxs]),
            "done": self._to_tensor_batch(self.done_buffer[idxs]),
            "h_in": self._to_tensor_batch(self.h_in_buffer[idxs]),
            "c_in": self._to_tensor_batch(self.c_in_buffer[idxs]),
            "h_out": self._to_tensor_batch(self.h_out_buffer[idxs]),
            "c_out": self._to_tensor_batch(self.c_out_buffer[idxs]),
            "prev_action": self._to_tensor_batch(self.prev_action_buffer[idxs]),
            "route_hard": self._to_tensor_batch(self.route_hard_buffer[idxs]),
            "route_soft": self._to_tensor_batch(self.route_soft_buffer[idxs]),
            "sender_intents_recv": self._to_tensor_batch(self.sender_intents_recv_buffer[idxs]),
            "sender_hidden_recv": self._to_tensor_batch(self.sender_hidden_recv_buffer[idxs]),
            "recv_mask": self._to_tensor_batch(self.recv_mask_buffer[idxs]),
            "time_lags": self._to_tensor_batch(self.time_lags_buffer[idxs]),
        }
    
    def _is_valid_anchor(self, idx: int, pred_horizon: int) -> bool:
        """
        判断 idx 能不能作为当前时刻 t 的锚点。

        合法条件：
        1. 需要拿到 t-1，所以 idx 必须 >= 1
        2. 需要拿到 t 到 t+K-1，所以 idx + pred_horizon - 1 不能越界
        3. t-1 到 t+K-1 必须都在同一个 episode
        4. 这些 step_id 必须严格连续
        """
        if self.size_tracker <= 0:
            return False

        # 需要 t-1
        if idx <= 0:
            return False

        # 需要 [t, t+K-1]
        if idx + pred_horizon - 1 >= self.size_tracker:
            return False

        ep = self.episode_id_buffer[idx]
        if ep < 0:
            return False

        # 检查 [t-1, t, ..., t+K-1]
        indices = np.arange(idx - 1, idx + pred_horizon)

        eps = self.episode_id_buffer[indices]
        steps = self.step_id_buffer[indices]

        # 必须同一条 episode
        if not np.all(eps == ep):
            return False

        # step 必须连续递增 1
        if not np.all(np.diff(steps) == 1):
            return False

        return True
    
    def _get_valid_anchor_indices(self, pred_horizon: int) -> List[int]:
        """
        返回当前 replay buffer 中所有合法的 intent sequence 锚点。
        """
        if self.size_tracker <= 1:
            return []

        valid = []
        # idx 作为当前时刻 t
        for idx in range(1, self.size_tracker):
            if self._is_valid_anchor(idx, pred_horizon):
                valid.append(idx)
        return valid


    def num_valid_intent_anchors(self, pred_horizon: int) -> int:
        """
        返回当前 replay buffer 中可用于 sample_intent_sequences 的合法锚点数量。
        """
        return len(self._get_valid_anchor_indices(pred_horizon))

    
    def sample_intent_sequences(self, batch_size: int, pred_horizon: int) -> Dict[str, torch.Tensor]:
        """
        采样 CoDe 的发送端辅助训练序列。

        以 anchor = t 为中心，返回：
        - h_prev           : t-1 时刻的历史隐藏状态
        - prev_action_prev : t-2 -> t-1 的上一动作（按你当前 prev_action_buffer 的定义来）
        - h_curr           : t 时刻的历史隐藏状态
        - prev_action_curr : t-1 -> t 的上一动作
        - obs_decode       : [o_t, o_{t+1}, ..., o_{t+K-1}]
        - target_action_seq: [a_t, a_{t+1}, ..., a_{t+K-1}]
        """
        valid = self._get_valid_anchor_indices(pred_horizon)

        if len(valid) < batch_size:
            raise ValueError(
                f"Not enough valid intent anchors: need {batch_size}, "
                f"but only have {len(valid)} valid anchors."
            )

        anchors = np.random.choice(valid, batch_size, replace=False)

        # t-1
        h_prev = self.h_out_buffer[anchors - 1]
        prev_action_prev = self.prev_action_buffer[anchors - 1]

        # t
        h_curr = self.h_out_buffer[anchors]
        prev_action_curr = self.prev_action_buffer[anchors]

        # [o_t, ..., o_{t+K-1}]
        obs_decode = np.stack(
            [self.obs_buffer[a : a + pred_horizon] for a in anchors],
            axis=0
        )  # [B, K, N, O]

        # [a_t, ..., a_{t+K-1}]
        # 注意：这里应该用无噪 policy 动作做 decoder 的监督
        target_action_seq = np.stack(
            [self.action_policy_buffer[a : a + pred_horizon] for a in anchors],
            axis=0
        )  # [B, K, N, A]

        # 现在转成 [B, N, K, ...]
        obs_decode = np.transpose(obs_decode, (0, 2, 1, 3))
        target_action_seq = np.transpose(target_action_seq, (0, 2, 1, 3))

        return {
            "h_prev": self._to_tensor_batch(h_prev),                     # [B, N, H]
            "prev_action_prev": self._to_tensor_batch(prev_action_prev), # [B, N, A]
            "h_curr": self._to_tensor_batch(h_curr),                     # [B, N, H]
            "prev_action_curr": self._to_tensor_batch(prev_action_curr), # [B, N, A]
            "obs_decode": self._to_tensor_batch(obs_decode),             # [B, N, K, O]
            "target_action_seq": self._to_tensor_batch(target_action_seq), # [B, N, K, A]
        }
    
    # ==========================================
    # ✅ 修复后：把整个脑子（记忆矩阵）打包存到硬盘
    # ==========================================
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        buffer_path = os.path.join(save_dir, "replay_buffer.pkl")
        
        # ⚠️ 修复点：保存真正的变量 self.size_tracker
        state = {
            'obs': self.obs_buffer,
            'action_exec': self.action_exec_buffer,
            'action_policy': self.action_policy_buffer,
            'reward': self.reward_buffer,
            'next_obs': self.next_obs_buffer,
            'done': self.done_buffer,
            'ptr': self.ptr, 
            'h_in': self.h_in_buffer,
            'c_in': self.c_in_buffer,
            'h_out': self.h_out_buffer,
            'c_out': self.c_out_buffer,  
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
            self.action_exec_buffer = state['action_exec']
            self.action_policy_buffer = state['action_policy']
            self.reward_buffer = state['reward']
            self.next_obs_buffer = state['next_obs']
            self.done_buffer = state['done']
            self.ptr = state['ptr']
            self.h_in_buffer = state['h_in']
            self.c_in_buffer = state['c_in']
            self.h_out_buffer = state['h_out']
            self.c_out_buffer = state['c_out']
            
            # 🚨 救命补丁：如果是读那个带有 Bug 的旧文件，直接手动把 size 设满！
            if 'size_tracker' in state:
                self.size_tracker = state['size_tracker']
            else:
                # 兼容旧文件：如果之前错存成了方法，而你跑了 2850 局，池子绝对是满的 (假设容量 10w)
                self.size_tracker = self.capacity # 如果你的 capacity 不是 10w，改成对应的最大容量
                
            print(f"🧠 [记忆恢复] Replay Buffer 加载成功！当前拥有 {self.size_tracker} 条历史经验。")
        else:
            print("⚠️ [记忆为空] 未找到 replay_buffer.pkl，将从空池子开始探索。")

    def size(self):
        """返回当前经验池中真实的数据量"""
        return self.size_tracker
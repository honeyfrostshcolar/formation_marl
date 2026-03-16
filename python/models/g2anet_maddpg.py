import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class G2ANet_MADDPG_Actor(nn.Module):
    """
    专为 MADDPG 设计的分布式 Actor 网络
    融合了 G2ANet 机制，输出确定性连续动作
    """
    def __init__(self, action_dim=2, max_visible_teammates=3):
        super(G2ANet_MADDPG_Actor, self).__init__()
        
        self.max_visible_teammates = max_visible_teammates # 最大可见兄弟数量（不包括自己）
        self.rnn_hidden_dim = 64
        self.attention_dim = 32
        
        # 1. 特征编码层
        # 自己: 雷达(21) + 老大相对位置(2) = 23维
        self.self_encoder = nn.Sequential(
            nn.Linear(23, self.rnn_hidden_dim),
            nn.ReLU()
        )
        # 队友: 相对位置(dx, dy) + is_valid标志位 = 3维
        self.teammate_encoder = nn.Sequential(
            nn.Linear(3, self.rnn_hidden_dim),
            nn.ReLU()
        )

        # 2. Hard Attention (判断连线)
        self.hard_encoding = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # 3. Soft Attention (QKV)
        self.q = nn.Linear(self.rnn_hidden_dim, self.attention_dim, bias=False)
        self.k = nn.Linear(self.rnn_hidden_dim, self.attention_dim, bias=False)
        self.v = nn.Linear(self.rnn_hidden_dim, self.attention_dim)

        # 4. 动作解码层 (MADDPG 的精髓所在)
        self.action_decoder = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim + self.attention_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim) # 输出 2 维动作
        )

    def forward(self, obs):
        # obs 维度: (batch_size, 23 + 3 * 3 = 32)
        batch_size = obs.shape[0]

        # 拆解观测值
        self_features = obs[:, :23] 
        teammates_obs = obs[:, 23:].view(batch_size, self.max_visible_teammates, 3)
        valid_mask = teammates_obs[:, :, 2] 

        # 编码
        h_self = self.self_encoder(self_features) 
        h_teammates = self.teammate_encoder(teammates_obs) 

        # --- Hard Attention ---
        h_self_expanded = h_self.unsqueeze(1).expand(-1, self.max_visible_teammates, -1)
        hard_input = torch.cat([h_self_expanded, h_teammates], dim=-1) 

        # 端到端的“自动博弈抽象”
        hard_logits = self.hard_encoding(hard_input)
        hard_weights = F.gumbel_softmax(hard_logits, tau=1.0, hard=True)[:, :, 1]
        
        # 物理截断：如果是假节点，强制断开
        hard_weights = hard_weights * valid_mask 

        # --- Soft Attention ---
        q = self.q(h_self).unsqueeze(1) 
        k = self.k(h_teammates)         
        v = F.relu(self.v(h_teammates)) 

        score = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.attention_dim)

        score = score.masked_fill(valid_mask.unsqueeze(1) == 0, -1e9)

        soft_weight = F.softmax(score, dim=-1) 

        teammates_dx_dy = teammates_obs[:, :, :2]
        physical_dist = torch.norm(teammates_dx_dy, dim=-1) + 1e-6

        distance_bias = torch.exp(-physical_dist) * valid_mask

        phys_weights = distance_bias / (distance_bias.sum(dim=-1, keepdim=True) + 1e-6)
        phys_weights = phys_weights.unsqueeze(1)

        alpha = 0.2
        soft_weight = (1 - alpha) * soft_weight + alpha * phys_weights

        # --- 信息融合 ---
        # GNN 的“消息传递与聚合
        attention_out = torch.bmm(soft_weight * hard_weights.unsqueeze(1), v).squeeze(1)
        final_input = torch.cat([h_self, attention_out], dim=-1)

        # --- 动作输出 (MADDPG 专属) ---
        raw_action = self.action_decoder(final_input)
        
        # ✅ 使用 Tanh 将动作死死限制在 [-1.0, 1.0] 之间
        action = torch.tanh(raw_action)
        
        return action, hard_weights, soft_weight
    

class Centralized_Critic(nn.Module):
    """
    上帝视角评论家 (集中式 Critic)
    输入：所有人的观测 + 所有人的动作
    输出：一个全局 Q 值打分
    """
    def __init__(self, num_followers, obs_dim, action_dim):
        super(Centralized_Critic, self).__init__()
        # Critic 需要看全图，所以输入维度是 N 个人的 obs 和 N 个人的 action 拼接在一起
        self.global_obs_dim = num_followers * obs_dim
        self.global_action_dim = num_followers * action_dim
        
        self.fc1 = nn.Linear(self.global_obs_dim + self.global_action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1) # 输出一个 Q 值

    def forward(self, global_obs, global_actions):
        # 将全局状态和全局动作拼接
        x = torch.cat([global_obs, global_actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
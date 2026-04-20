import torch
import torch.nn as nn
import numpy as np
from utils.comm_utils import build_mlp

class LeaderBeliefEncoder(nn.Module):
    """
    领航者信念编码器 (Theory of Mind)
    输入：小弟的局部雷达 (41维) + 小弟的历史记忆 (h_t)
    输出：老大的意图分布 (mu, logvar), 采样意图 (belief_z), 以及不确定性 (entropy)
    """
    def __init__(self, hid_size, lidar_dim=41, belief_dim=32):
        super().__init__()
        # 雷达特征提取
        self.lidar_mlp = build_mlp(
            in_dim=lidar_dim,
            hidden_dims=[64, 64],
            out_dim=hid_size,
            activate_last=True
        )
        # 融合历史记忆与雷达特征
        self.fusion = build_mlp(
            in_dim=hid_size * 2,
            hidden_dims=[64],
            out_dim=64,
            activate_last=True
        )
        # 输出高斯分布参数
        self.mu_head = nn.Linear(64, belief_dim)
        self.logvar_head = nn.Linear(64, belief_dim)
        
    def forward(self, h_t, lidar_obs, leader_rel, leader_heading):

        raw_obs = torch.cat([lidar_obs, leader_rel, leader_heading], dim=-1)
        # 1. 提取环境特征
        l_feat = self.lidar_mlp(raw_obs)
        # 2. 结合历史上下文进行推断
        context = self.fusion(torch.cat([h_t, l_feat], dim=-1))
        
        # 3. 生成高斯分布
        mu = self.mu_head(context)
        logvar = self.logvar_head(context).clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        
        # 4. 重参数化采样 Belief Vector
        eps = torch.randn_like(std)
        belief_z = mu + eps * std
        
        # 5. 计算不确定性 (熵正比于 logvar)
        # entropy 维度: [B, N, 1]，代表每个小弟对老大意图的“迷茫程度”
        entropy = 0.5 * (1.0 + np.log(2 * np.pi) + logvar).mean(dim=-1, keepdim=True)
        
        return mu, logvar, belief_z, entropy


class LeaderBeliefDecoder(nn.Module):
    """
    辅助任务解码器：迫使 Belief 向量真的包含老大的未来走向
    输入：belief_z + 历史记忆 h_t
    输出：预测老大未来 K 步的相对位置 (dx, dy)
    """
    def __init__(self, hid_size, pred_horizon, belief_dim=32):
        super().__init__()
        self.pred_horizon = pred_horizon
        # 预测未来 K 步的 (x, y) 相对坐标，所以输出维度是 K * 2
        self.mlp = build_mlp(
            in_dim=belief_dim + hid_size,
            hidden_dims=[64, 64],
            out_dim=self.pred_horizon * 2
        )
        
    def forward(self, belief_z, h_t):
        x = torch.cat([belief_z, h_t], dim=-1)
        preds = self.mlp(x) # [B, N, K * 2]
        b, n, _ = preds.shape
        return preds.reshape(b, n, self.pred_horizon, 2)
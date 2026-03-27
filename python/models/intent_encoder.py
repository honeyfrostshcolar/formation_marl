from typing import Tuple

import torch
import torch.nn as nn

from .comm_utils import build_mlp


class IntentEncoder(nn.Module):
    """意图编码器。

    输入:
    - h_t: [B, N, H] [batch_size, num_followers, hid_size]
    - prev_action: [B, N, A] [batch_size, num_followers, action_dim]

    输出:
    - mu/logvar/intent: [B, N, E] [batch_size, num_followers, intent_dim]

    作用：
    历史轨迹 h_t 反映“过去做了什么”，
    前一时刻动作 a_{t-1} 提供“最近的行为落点”，
    二者拼接后更有利于估计当前的长期行为趋势。
    """

    def __init__(self, args):
        super().__init__()
        self.hid_size = args.hid_size
        self.action_dim = args.action_dim
        self.intent_dim = args.intent_dim

        in_dim = self.hid_size + self.action_dim
        self.backbone = build_mlp(
            in_dim=in_dim,
            hidden_dims=[self.hid_size, self.hid_size],
            out_dim=self.intent_dim,
            activate_last=True,
        )
        self.mu_head = nn.Linear(self.intent_dim, self.intent_dim)
        self.logvar_head = nn.Linear(self.intent_dim, self.intent_dim)

    def forward(
        self,
        h_t: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = h_t.shape
        x = torch.cat([h_t, prev_action], dim=-1)
        feat = self.backbone(x.reshape(b * n, -1)).reshape(b, n, -1)
        mu = self.mu_head(feat)

        # 对 logvar 做截断，避免 exp 后数值过大或过小
        logvar = self.logvar_head(feat).clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        intent = mu + eps * std
        return mu, logvar, intent

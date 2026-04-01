"""
双对齐融合模块。

接收端如何融合多源异步消息。

整体分三步：
1. Intent Alignment
   - 用 intent 做 query / key
   - 用 (intent, hidden_state) 做 value
2. Timeliness Alignment
   - 对 attention 权重乘上时间衰减 gamma_t ^ delta_t
3. Message Fusion
   - 用衰减后的权重聚合 value，得到 c_i^t

同时这里还实现了论文里的 entropy regularization，也就是 L_e。

关键变化:
1. 不再假设所有接收者看到同一份 sender tensor。
2. 直接使用 receiver-major 输入:
   - receiver_intents: [B, N, E]
   - sender_intents:   [B, N, N, E]   # 第二维是 receiver, 第三维是 sender
   - sender_hidden:    [B, N, N, H]
   - recv_mask:        [B, N, N]
   - time_lags:        [B, N, N]
这样才能真正表达“不同接收者拿到的延迟消息不同”。
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from utils.comm_utils import build_mlp


class DualAlignmentFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.intent_dim = args.intent_dim
        self.hidden_dim = args.hid_size
        self.value_dim = args.value_dim
        self.attn_dim = args.attn_dim
        self.gamma_t = args.gamma_t
        self.lambda_e = args.lambda_e
        self.eps = args.eps
        self.renorm_after_decay = args.renorm_after_decay

        self.WQ = nn.Linear(self.intent_dim, self.attn_dim, bias=False)
        self.WK = nn.Linear(self.intent_dim, self.attn_dim, bias=False)
        self.WV = build_mlp(
            in_dim=self.intent_dim + self.hidden_dim,
            hidden_dims=[self.hidden_dim, self.hidden_dim],
            out_dim=self.value_dim,
        )

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask = mask.bool()
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        # 避免某一整行全是 -inf 导致 NaN
        no_valid = (~mask).all(dim=dim, keepdim=True)
        masked_logits = torch.where(no_valid, torch.zeros_like(masked_logits), masked_logits)
        alpha = torch.softmax(masked_logits, dim=dim)
        alpha = alpha * mask.float()
        denom = alpha.sum(dim=dim, keepdim=True).clamp_min(1e-8)
        return torch.where(no_valid, torch.zeros_like(alpha), alpha / denom)

    def forward(
        self,
        receiver_intents: torch.Tensor,
        sender_intents: torch.Tensor,
        sender_hidden: torch.Tensor,
        available_mask: torch.Tensor,     # [B, N, N] 0/1, 消息是否已到达 (hard mask)
        route_gate: torch.Tensor,         # [B, N, N] 可微门控，来自当前 scheduler (soft mask)
        time_lags: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, n, _, _ = sender_intents.shape
        device = receiver_intents.device
        dtype = receiver_intents.dtype

        if time_lags is None:
            time_lags = torch.zeros(b, n, n, device=device, dtype=dtype)

        
        # ===== 1) QKV =====
        query = self.WQ(receiver_intents)                    # [B, N, D]
        key = self.WK(sender_intents)                        # [B, N, N, D]
        value = self.WV(torch.cat([sender_intents, sender_hidden], dim=-1))  # [B, N, N, V]

        logits = (query.unsqueeze(2) * key).sum(dim=-1) / math.sqrt(query.shape[-1])

        # ===== 2) 只用硬可用性做 softmax 归一化 =====
        eye = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)  # [1, N, N]
        hard_available = (available_mask > 0.5) & (~eye)   # bool mask

        alpha = self._masked_softmax(logits, hard_available, dim=-1)

        # ===== 3) 时间衰减 =====
        decay = torch.pow(
            torch.tensor(self.gamma_t, device=device, dtype=dtype),
            time_lags.clamp_min(0.0),
        )

        # ===== 4) 当前 scheduler 的软门控 =====
        # route_gate 是浮点张量，必须保留梯度
        route_gate = route_gate * (~eye).float()

        # 最终门控：
        # - alpha: 内容相关性
        # - decay: 时效性
        # - route_gate: 当前 scheduler 想不想连这条边
        # - available_mask: 这条消息是否真的到达过
        gated_alpha = alpha * decay * route_gate * available_mask.float()

        if self.renorm_after_decay:
            denom = gated_alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            gated_alpha = gated_alpha / denom

        # ===== 5) 熵正则 =====
        # 注意：这里用 gated_alpha，而不是 alpha
        # 这样 L_e 才能对 scheduler 产生梯度
        gated_safe = gated_alpha.clamp_min(self.eps)
        entropy_term = gated_safe * gated_safe.log()
        L_e = -self.lambda_e * entropy_term.sum(dim=(-1, -2)).mean()

        # ===== 6) 聚合 =====
        combined = (gated_alpha.unsqueeze(-1) * value).sum(dim=2)

        return {
            "combined": combined,
            "alpha": alpha,
            "alpha_hat": gated_alpha,
            "L_e": L_e,
        }


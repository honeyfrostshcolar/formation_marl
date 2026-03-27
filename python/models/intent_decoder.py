from typing import List

import torch
import torch.nn as nn

from .comm_utils import build_mlp


class IntentDecoder(nn.Module):
    """意图解码器。

    这里直接按场景固定为连续动作:
    - decoder 输入上一时刻动作向量，不再保留离散动作 one-hot 分支
    - 训练时用 MSE 去拟合未来 K 步动作
    """

    def __init__(self, args):
        super().__init__()
        self.obs_dim = args.obs_size
        self.action_dim = args.action_dim
        self.intent_dim = args.intent_dim
        self.history_dim = args.hid_size
        self.decoder_hidden_dim = args.decoder_hidden_dim

        # GRU 解码器
        self.G = nn.GRUCell(
            input_size=self.obs_dim + self.action_dim,
            hidden_size=self.decoder_hidden_dim,
        )
        # 如果 h_t 维度和 decoder hidden 不一样，就先投影一下
        if self.history_dim == self.decoder_hidden_dim:
            self.init_proj = nn.Identity()
        else:
            self.init_proj = nn.Linear(self.history_dim, self.decoder_hidden_dim)

        # 动作预测网络
        self.F = build_mlp(
            in_dim=self.decoder_hidden_dim + self.intent_dim,
            hidden_dims=[self.decoder_hidden_dim, self.decoder_hidden_dim],
            out_dim=self.action_dim,
        )

    def forward(
        self,
        intent: torch.Tensor,
        history_h_t: torch.Tensor,
        obs_decode: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        intent: [B*, E]
        history_h_t: [B*, H]
        obs_decode: [B*, K, O]
        prev_action: [B*, A] 上一步的动作向量
        """
        _, pred_horizon, _ = obs_decode.shape
        hidden = self.init_proj(history_h_t)
        action_in = prev_action

        preds: List[torch.Tensor] = []
        for step in range(pred_horizon):
            decoder_in = torch.cat([obs_decode[:, step], action_in], dim=-1)
            hidden = self.G(decoder_in, hidden)
            pred = self.F(torch.cat([intent, hidden], dim=-1))
            preds.append(pred)
            action_in = pred

        return torch.stack(preds, dim=1)


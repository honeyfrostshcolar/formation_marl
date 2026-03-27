"""在线推理专用的运行时延迟缓存，只用于与环境交互的阶段

它只服务于 collect rollout（数据采集） / select_action（选择动作） 阶段，不参与梯度图。
设计目标:
- 保持发送端与接收端分离
- 当前时刻先生成 sender packet，再根据延迟决定哪些 receiver 能收到
- receiver 始终持有“每个 sender 最近一次已到达的消息”
"""

from typing import Dict, List

import numpy as np
import torch


class RuntimeDelayBuffer:
    def __init__(self, args):
        self.args = args
        self.current_step = 0
        self.future_queue: List[Dict] = []
        self.latest_packets = None

    def reset(self, n_agents: int):
        self.current_step = 0
        self.future_queue = []
        self.latest_packets = [[None for _ in range(n_agents)] for _ in range(n_agents)]

    def _sample_delay_matrix(self, n_agents: int) -> np.ndarray:
        if self.args.delay_mode == "none":
            delay = np.zeros((n_agents, n_agents), dtype=np.int64)
        elif self.args.delay_mode == "fixed":
            delay = np.full((n_agents, n_agents), self.args.fixed_delay, dtype=np.int64)
        elif self.args.delay_mode == "uniform":
            delay = np.random.randint(self.args.min_delay, self.args.max_delay + 1, size=(n_agents, n_agents), dtype=np.int64)
        else:
            raise ValueError(f"Unknown delay_mode={self.args.delay_mode}")
        np.fill_diagonal(delay, 0)
        return delay

    def push_current_packets(
        self,
        sender_intents: torch.Tensor,   # [N, E]
        sender_hidden: torch.Tensor,    # [N, H]
        route_mask: torch.Tensor,       # [N, N], sender -> receiver
    ) -> torch.Tensor:
        
        """
        在当前时间步，根据路由掩码 route_mask 决定哪些发送者向哪些接收者发送消息，
        将消息包放入 future_queue 队列，并附带送达时间步。

        """


        n_agents = route_mask.shape[0]
        delay_mat = self._sample_delay_matrix(n_agents) # 随机采样延迟矩阵

        for s in range(n_agents):
            for r in range(n_agents):
                if s == r:
                    continue
                if float(route_mask[s, r].item()) <= 0.0:
                    continue
                packet = {
                    "sender": s,
                    "receiver": r,
                    "intent": sender_intents[s].detach().clone(),
                    "hidden": sender_hidden[s].detach().clone(),
                    "send_step": self.current_step,
                    "deliver_step": self.current_step + int(delay_mat[s, r]),
                }
                self.future_queue.append(packet)
        return torch.as_tensor(delay_mat, dtype=sender_intents.dtype, device=sender_intents.device)

    def collect_receiver_inputs(
        self,
        receiver_intents: torch.Tensor,  # [1, N, E] only for shapes/device
        hidden_dim: int,
    ):
        """
        输入：
        receiver_intents : (1, num_agents, intent_dim)
        hidden_dim : int  # LSTM hidden state dimension
        输出：
        sender_intents : (1, num_agents, num_agents, intent_dim)每个接收者视角下，每个发送者的意图
        sender_hidden : (1, num_agents, num_agents, hidden_dim)每个接收者看到的发送者隐藏状态
        recv_mask : (1, num_agents, num_agents)  # 接收者视角下，每个接收者实际收到了哪些发送者的消息
        time_lags : (1, num_agents, num_agents)  # 每个消息的延迟（当前步减去消息发送步），0表示刚收到，正数表示消息是几秒前发送的。
        
        在每个时间步结束时被调用，检查队列中哪些消息已送达，
        更新每个接收者维护的“每个发送者的最新消息”缓存（latest_packets），
        然后生成当前时间步每个接收者实际收到的消息张量（sender_intents, sender_hidden, recv_mask, time_lags），
        供后续消息融合模块使用。
        
        """

        device = receiver_intents.device
        dtype = receiver_intents.dtype
        _, n_agents, intent_dim = receiver_intents.shape

        remaining = []
        for packet in self.future_queue:
            if packet["deliver_step"] <= self.current_step:
                r = packet["receiver"]
                s = packet["sender"]
                old = self.latest_packets[r][s]
                if old is None or packet["send_step"] >= old["send_step"]:
                    self.latest_packets[r][s] = packet
            else:
                remaining.append(packet)
        self.future_queue = remaining

        sender_intents = torch.zeros(1, n_agents, n_agents, intent_dim, device=device, dtype=dtype)
        sender_hidden = torch.zeros(1, n_agents, n_agents, hidden_dim, device=device, dtype=dtype)
        recv_mask = torch.zeros(1, n_agents, n_agents, device=device, dtype=dtype)
        time_lags = torch.zeros(1, n_agents, n_agents, device=device, dtype=dtype)

        for r in range(n_agents):
            for s in range(n_agents):
                pkt = self.latest_packets[r][s]
                if pkt is None or r == s:
                    continue
                sender_intents[0, r, s] = pkt["intent"].to(device=device, dtype=dtype)
                sender_hidden[0, r, s] = pkt["hidden"].to(device=device, dtype=dtype)
                recv_mask[0, r, s] = 1.0
                time_lags[0, r, s] = float(self.current_step - pkt["send_step"])

        self.current_step += 1
        return {
            "sender_intents": sender_intents,
            "sender_hidden": sender_hidden,
            "recv_mask": recv_mask,
            "time_lags": time_lags,
        }

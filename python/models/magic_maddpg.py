import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from models.gnn_layers import GraphAttention
from models.intent_encoder import IntentEncoder
from models.intent_decoder import IntentDecoder
from models.fusion_net import DualAlignmentFusion
from models.runtime_delay_buffer import RuntimeDelayBuffer
from utils.comm_utils import ensure_action_tensor
from models.leader_belief import LeaderBeliefDecoder, LeaderBeliefEncoder

class MAGICCoDeActor(nn.Module):
    """重构 Actor。

    设计原则:
    - 发送端: obs -> LSTM -> h_t -> intent
    - 路由端: 只保留 MAGIC 的 Scheduler，不再保留 sub_processor1 / sub_processor2
    - 接收端: 用 CoDe dual alignment 替代原 MAGIC 的 GAT Message Processor
    - 输出端: policy head 只吃 [h_t, intent, fused_msg]
    """
    def __init__(self, args):
        super(MAGICCoDeActor, self).__init__()
        """
        Initialization method for the MAGICCoDeActor communication protocol (2 rounds of communication)

        Arguements:
            args (Namespace): Parse arguments
        """

        self.args = args
        self.nagents = args.num_followers
        self.hid_size = args.hid_size
        self.action_dim = args.action_dim
        self.intent_dim = args.intent_dim
        self.value_dim = args.value_dim
        
        dropout = 0
        negative_slope = 0.2

        self.prev_action = None

        # initialize the gat encoder for the Scheduler
        self.gat_encoder = None
        if args.use_gat_encoder:
            self.gat_encoder = GraphAttention(args.hid_size, args.gat_encoder_out_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.ge_num_heads, self_loop_type=1, average=True, normalize=args.gat_encoder_normalize)

        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)

        # 发送方
        self.intent_encoder = IntentEncoder(args) # 意图编码器
        self.intent_decoder = IntentDecoder(args) # 意图解码器

        self.belief_dim = 32
        self.leader_belief_encoder = LeaderBeliefEncoder(args.hid_size, lidar_dim=41, belief_dim=self.belief_dim)
        self.leader_belief_decoder = LeaderBeliefDecoder(args.hid_size, pred_horizon=args.pred_horizon, belief_dim=self.belief_dim)

        # 接收方
        self.message_fusion = DualAlignmentFusion(args) # 消息融合器

        # self.init_hidden(args.batch_size)
        self.lstm_cell= nn.LSTMCell(args.hid_size, args.hid_size)

        sched_in_dim = args.hid_size + 1

        # initialize mlp layers for the sub-schedulers
        if not args.first_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear((args.gat_encoder_out_size + 1)*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(sched_in_dim*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))
                
        if args.learn_second_graph and not args.second_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear((args.gat_encoder_out_size + 1)*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(sched_in_dim*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))


        self.message_encoder = nn.Linear(args.hid_size, args.hid_size) if args.message_encoder else None
        self.message_decoder = nn.Linear(args.value_dim, args.value_dim) if args.message_decoder else None

        self.runtime_buffer = RuntimeDelayBuffer(args)
        self.runtime_prev_action = None

        # initialize weights as 0
        if args.comm_init == 'zeros':
            if args.message_encoder:
                self.message_encoder.weight.data.zero_()
            if args.message_decoder:
                self.message_decoder.weight.data.zero_()
            if not args.first_graph_complete:
                self.sub_scheduler_mlp1.apply(self.init_linear)
            if args.learn_second_graph and not args.second_graph_complete:
                self.sub_scheduler_mlp2.apply(self.init_linear)
                   
        # initialize the action head (in practice, one action head is used)
        # policy head 输入维度改为 h + intent + fused
        policy_in_dim = args.hid_size + args.intent_dim + args.value_dim + self.belief_dim

        self.action_head = nn.Sequential(
            nn.Linear(policy_in_dim, args.hid_size),
            nn.ReLU(),
            nn.Linear(args.hid_size, args.action_dim),
            nn.Tanh(),
        )

    def reset_runtime_state(self, batch_size: int = 1):
        self.runtime_buffer.reset(self.nagents)
        self.runtime_prev_action = None

    def _build_scheduler_mlp(self, feat_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 2),
        )

    def init_linear(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.fill_(0.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def init_hidden(self, batch_size: int, n: Optional[int] = None):
        n_agents = n if n is not None else self.nagents
        h = torch.zeros(batch_size, n_agents, self.hid_size)
        c = torch.zeros(batch_size, n_agents, self.hid_size)
        return h, c

    def _scheduler_features(self, sender_hidden: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        """
        利用智能体间的全局关系（尽管是虚拟的全连接）来学习通信拓扑。
        """
        if self.gat_encoder is None:
            return sender_hidden
        adj_comp = self.get_complete_graph(agent_mask)
        return self.gat_encoder(sender_hidden, adj_comp)

    def sub_scheduler(self, mlp, hidden_state, agent_mask, directed=True):
        """输出 sender->receiver 路由矩阵。

        agent_mask：标记哪些智能体活跃（非故障）
        directed=True：表示路由是单向的，即 sender 只能向 receiver 发送消息。
        directed=False：表示路由是双向的，即 sender 可以向 receiver 发送消息，也可以从 receiver 接收消息。

        输出表示 sender->receiver 的路由矩阵，“不发送”和“发送”，每个元素为 [0, 1] 之间的概率值。

        """
        b, n, f_dim = hidden_state.shape
        hs = hidden_state.unsqueeze(2).expand(b, n, n, f_dim)  # sender
        hr = hidden_state.unsqueeze(1).expand(b, n, n, f_dim)  # receiver
        pair_feat = torch.cat([hs, hr], dim=-1)
        logits = mlp(pair_feat)

        if not directed:
            logits = 0.5 * (logits + logits.transpose(1, 2))

        hard = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)[..., 1]
        soft = torch.softmax(logits, dim=-1)[..., 1]

        mask = torch.bmm(agent_mask, agent_mask.transpose(1, 2))
        eye = torch.eye(n, device=hidden_state.device, dtype=hidden_state.dtype).unsqueeze(0)
        hard = hard * mask * (1.0 - eye)
        soft = soft * mask * (1.0 - eye)
        return hard, soft

    def build_route_graph(self, sender_hidden: torch.Tensor, agent_mask: torch.Tensor, belief_entropy: torch.Tensor):

        """
        通过一步或两步调度器构建路由图，用于确定智能体之间的通信拓扑。
        """

        feat1 = self._scheduler_features(sender_hidden, agent_mask)
        sched_input1 = torch.cat([feat1, belief_entropy], dim=-1)
      
        adj1_hard, adj1_soft = self.sub_scheduler(self.sub_scheduler_mlp1, sched_input1, agent_mask, self.args.directed)

        if self.args.learn_second_graph:
            feat2 = self._scheduler_features(sender_hidden, agent_mask)
            sched_input2 = torch.cat([feat2, belief_entropy], dim=-1)
            adj2_hard, adj2_soft = self.sub_scheduler(self.sub_scheduler_mlp2, sched_input2, agent_mask, self.args.directed)
            # [MOD 6-4] 两轮 scheduler 现在做“粗筛 + 细筛”，最终路由为逐元素相乘
            final_hard = adj1_hard * adj2_hard
            final_soft = adj1_soft * adj2_soft
        else:
            final_hard, final_soft = adj1_hard, adj1_soft

        if self.args.comm_mask_zero:
            final_hard = torch.zeros_like(final_hard)
            final_soft = torch.zeros_like(final_soft)
        return final_hard, final_soft

    def get_complete_graph(self, agent_mask):
        """
        用于生成一个全连接邻接矩阵，但仅连接活跃的智能体。
        """
        return torch.bmm(agent_mask, agent_mask.transpose(1, 2))

    def _build_synchronous_receiver_inputs(
        self,
        intents: torch.Tensor,
        sender_hidden: torch.Tensor,
        route_hard: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        """
        在没有延迟的情况下，构造一个同步通信的接收端输入。
        """
        b, n, e = intents.shape
        _, _, h = sender_hidden.shape
        sender_intents = intents.unsqueeze(1).expand(b, n, n, e).clone()
        sender_hidden_rm = sender_hidden.unsqueeze(1).expand(b, n, n, h).clone()
        recv_mask = route_hard.transpose(1, 2).contiguous()  # receiver,row ; sender,col
        time_lags = torch.zeros(b, n, n, device=intents.device, dtype=intents.dtype)
        return {
            "sender_intents": sender_intents,
            "sender_hidden": sender_hidden_rm,
            "recv_mask": recv_mask,
            "time_lags": time_lags,
        }
    
    
    def decode_intent_for_training(
        self,
        intents: torch.Tensor,        # [B, N, E]
        history_h: torch.Tensor,      # [B, N, H]
        obs_decode: torch.Tensor,     # [B, N, K, obs_dim]
        prev_action: torch.Tensor,    # [B, N, A]
    ) -> torch.Tensor:
        """发送端解码支路，只用于训练 L_inf。"""
        b, n, k, obs_dim = obs_decode.shape

        flat_intents = intents.reshape(b * n, -1)
        flat_history_h = history_h.reshape(b * n, -1)
        flat_obs_decode = obs_decode.reshape(b * n, k, obs_dim)
        flat_prev_action = prev_action.reshape(b * n, -1)

        pred_seq = self.intent_decoder(
            intent=flat_intents,
            history_h_t=flat_history_h,
            obs_decode=flat_obs_decode,
            prev_action=flat_prev_action,
        )
        return pred_seq.reshape(b, n, k, self.action_dim)
    
    def _build_receiver_major_sender_tensors(
        self,
        intents: torch.Tensor,         # [B, N, E]
        sender_hidden: torch.Tensor,   # [B, N, H]
    ):
        """
        把当前 sender 侧张量扩成 receiver-major 形式:
        - sender_intents_rm: [B, N, N, E]
        - sender_hidden_rm : [B, N, N, H]
        """
        b, n, e = intents.shape
        _, _, h = sender_hidden.shape

        sender_intents_rm = intents.unsqueeze(1).expand(b, n, n, e).clone()
        sender_hidden_rm = sender_hidden.unsqueeze(1).expand(b, n, n, h).clone()

        return sender_intents_rm, sender_hidden_rm
    

    def forward(
        self,
        obs: torch.Tensor,
        extras: Tuple[torch.Tensor, torch.Tensor],
        prev_action: Optional[torch.Tensor] = None,
        runtime_mode: bool = False,
        external_comm: Optional[Dict[str, torch.Tensor]] = None,
    ):
        

        """ 
        MAGIC 的前向传播函数

        输入：
        obs : (batch_size, num_agents, obs_dim)
        extras : (h_in, c_in)  # LSTM hidden state
        prev_action : (batch_size, num_agents, action_dim)
        输出：
        action : (batch_size, num_agents, action_dim)
        recv_inputs : (Dict[str, torch.Tensor])
        recv_inputs 包含接收端的输入，用于计算接收端的路由矩阵。

        runtime_mode=True：表示当前处于在线交互模式，需要使用 runtime delay buffer
        external_comm：用于在训练或离线评估时向模型注入外部提供的通信数据。
         
        """
        h_in, c_in = extras
        b, n, _ = obs.shape
        device = obs.device
        dtype = obs.dtype

        encoded_obs = self.obs_encoder(obs)
        # 兼容死掉的 agent (全 1 代表全活)
        agent_mask = torch.ones(b, n, 1, device=device, dtype=dtype)

        # LSTM 时序更新 (展平 -> 计算 -> 还原)
        h_out, c_out = self.lstm_cell(
            encoded_obs.reshape(b * n, -1),
            (h_in.reshape(b * n, -1), c_in.reshape(b * n, -1)),
        )
        h_out = h_out.reshape(b, n, self.hid_size)
        c_out = c_out.reshape(b, n, self.hid_size)

        lidar_obs = obs[..., :41]                   # 雷达数据 (看墙)
        leader_rel = obs[..., 41:43]                # 老大相对坐标 (看老大的位置)
        leader_heading = obs[..., -3:]              # 老大航向差与角速度 (看老大的扭动姿态)

        b_mu, b_logvar, belief_z, belief_entropy = self.leader_belief_encoder(
            h_out,           # 记忆上下文
            lidar_obs,       # 戴上眼镜看墙
            leader_rel,      # 戴上眼镜看老大当前坐标
            leader_heading   # 戴上眼镜看老大当前车头朝向
        )

        if runtime_mode:
            prev_action = ensure_action_tensor(self.runtime_prev_action, b, n, self.action_dim, device, dtype)
        else:
            prev_action = ensure_action_tensor(prev_action, b, n, self.action_dim, device, dtype)

        # 意图编码器
        mu, logvar, intents = self.intent_encoder(h_out, prev_action)

        # 发送端路由
        sender_hidden = h_out
        if self.message_encoder is not None:
            sender_hidden = self.message_encoder(sender_hidden)
        sender_hidden = sender_hidden * agent_mask

        route_hard, route_soft = self.build_route_graph(h_out, agent_mask, belief_entropy)

        # ==========================================
        # 接收端输入：执行硬、训练软
        # ==========================================
        if runtime_mode:
            # -----------------------------
            # 运行时：用硬图真的发消息
            # -----------------------------
            if b != 1:
                raise ValueError("runtime_mode=True 目前只支持 batch_size=1")

            delay_mat = self.runtime_buffer.push_current_packets(
                sender_intents=intents[0],
                sender_hidden=sender_hidden[0],
                route_mask=route_hard[0],   # 运行时只用硬图
            )

            recv_inputs = self.runtime_buffer.collect_receiver_inputs(
                receiver_intents=intents,
                hidden_dim=self.hid_size,
            )

            # 运行时 fusion 也按硬可用图走
            fusion_sender_intents = recv_inputs["sender_intents"]
            fusion_sender_hidden = recv_inputs["sender_hidden"]
            available_mask = recv_inputs["recv_mask"]       # 消息是否已到达
            route_gate = recv_inputs["recv_mask"]           # 运行时就按硬图执行
            time_lags = recv_inputs["time_lags"]
            online_delay = delay_mat.unsqueeze(0)

        elif external_comm is not None:
            # -----------------------------
            # 训练时：当前 actor 重新算 sender 内容 + 当前 scheduler 的软门控
            # replay 只提供“消息是否到达”和“延迟是多少”
            # -----------------------------
            fusion_sender_intents, fusion_sender_hidden = self._build_receiver_major_sender_tensors(
                intents, sender_hidden
            )

            # available_mask = external_comm["recv_mask"]   # 这是硬可用性，不参与学习

            # 假装物理信道全通，把切断信道的生杀大权完全交给下面的 route_gate！
            eye_mask = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
            available_mask = (1.0 - eye_mask).expand(b, n, n).clone()

            time_lags = external_comm["time_lags"]

            # 训练时让 scheduler 收到梯度：
            # 方案 1：纯软门控
            # route_gate = route_soft.transpose(1, 2).contiguous()

            # 方案 2：STE 门控（推荐）
            route_gate_soft = route_soft.transpose(1, 2).contiguous()
            route_gate_hard = route_hard.transpose(1, 2).contiguous()
            route_gate = route_gate_hard.detach() - route_gate_soft.detach() + route_gate_soft

            online_delay = time_lags.transpose(1, 2).contiguous()

        else:
            # -----------------------------
            # 无延迟同步简化模式
            # 当前 sender 内容仍然用当前 actor 计算
            # route gate 用当前 scheduler 的软门控
            # -----------------------------
            fusion_sender_intents, fusion_sender_hidden = self._build_receiver_major_sender_tensors(
                intents, sender_hidden
            )

            eye_mask = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
            available_mask = (1.0 - eye_mask).expand(b, n, n).clone()
            time_lags = torch.zeros(b, n, n, device=device, dtype=dtype)

            route_gate_soft = route_soft.transpose(1, 2).contiguous()
            route_gate_hard = route_hard.transpose(1, 2).contiguous()
            route_gate = route_gate_hard.detach() - route_gate_soft.detach() + route_gate_soft

            online_delay = time_lags.transpose(1, 2).contiguous()

        # ==========================================
        # 接收端融合
        # ==========================================
        fusion_out = self.message_fusion(
            receiver_intents=intents,
            sender_intents=fusion_sender_intents,
            sender_hidden=fusion_sender_hidden,
            available_mask=available_mask,
            route_gate=route_gate,
            time_lags=time_lags,
        )

        fused_msg = fusion_out["combined"]
        if self.message_decoder is not None:
            fused_msg = self.message_decoder(fused_msg)

        policy_feat = torch.cat([h_out, intents, fused_msg, belief_z], dim=-1)
        
        actions_out = self.action_head(policy_feat)

        if runtime_mode:
            self.runtime_prev_action = actions_out.detach().clone()

        aux = {
            "mu": mu,
            "logvar": logvar,
            "intents": intents,
            "history_h": h_out,
            "sender_hidden": sender_hidden,
            "route_hard": route_hard,
            "route_soft": route_soft,
            "recv_mask": available_mask,
            "time_lags": time_lags,
            "L_e": fusion_out["L_e"],
            "alpha": fusion_out["alpha"],
            "alpha_hat": fusion_out["alpha_hat"],
            "online_delay": online_delay,
            "prev_action_used": prev_action,
        }

        return actions_out, route_hard, route_soft, (h_out, c_out), aux

    

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